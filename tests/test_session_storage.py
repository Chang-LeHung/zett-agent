from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from kcs_agent import (
    Agent,
    AgentConfig,
    AssistantMessage,
    CompactionExtension,
    ImageBytesSource,
    ImageContent,
    ModelEvent,
    ModelResponse,
    SessionPersistenceExtension,
    SQLiteSessionExtension,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from kcs_agent.compaction import CompactedMessage
from kcs_agent.storage import (
    ContextSnapshotModel,
    RawLogMessageModel,
    SQLiteSessionStorage,
    decode_messages,
    encode_messages,
)


@pytest.fixture
def storage(tmp_path):
    storage = SQLiteSessionStorage(tmp_path / "sessions.sqlite3")
    yield storage
    storage.close()


class Model:
    def __init__(self, answer="Answer"):
        self.answer = answer
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=self.answer)))


async def test_storage_reopens_without_application_database(tmp_path):
    path = tmp_path / "nested" / "agent.sqlite3"
    storage = SQLiteSessionStorage(path)
    await storage.append("session", "request", UserMessage(content="Persisted"))
    storage.close()
    reopened = SQLiteSessionStorage(path)
    try:
        view = await reopened.load("session")
        assert view.messages == [UserMessage(content="Persisted")]
        assert view.snapshot is None
        assert view.raw_tail[-1].sequence == 1
        assert (await reopened.load("other-session")).messages == []
    finally:
        reopened.close()


async def test_sqlite_session_extension_owns_storage_and_restores_history(tmp_path):
    path = tmp_path / "owned.sqlite3"
    first = SQLiteSessionExtension(path)
    try:
        agent = await Agent.create(Model("First answer"), config=AgentConfig("session"), extensions=[first])
        await agent.run("First question")
    finally:
        first.close()

    second = SQLiteSessionExtension(path)
    model = Model("Second answer")
    try:
        agent = await Agent.create(model, config=AgentConfig("session"), extensions=[second])
        await agent.run("Second question")
        assert [message.content for message in model.requests[0].messages] == [
            "You are a helpful assistant.",
            "First question",
            "First answer",
            "Second question",
        ]
    finally:
        second.close()


async def test_same_agent_reloads_snapshot_and_external_tail_every_request(storage):
    session_id = "test-session"
    model = Model()
    agent = await Agent.create(model, config=AgentConfig(session_id), extensions=[SessionPersistenceExtension(storage)])
    await agent.run("First")
    first = await storage.load(session_id)
    assert first.snapshot is None
    await storage.append(session_id, "external", UserMessage(content="External update"))
    await agent.run("Second")
    assert [m.content for m in model.requests[1].messages] == [
        "You are a helpful assistant.",
        "First",
        "Answer",
        "External update",
        "Second",
    ]
    second = await storage.load(session_id)
    assert second.snapshot is None
    assert second.raw_tail[-1].sequence == 5
    with Session(storage.engine) as session:
        rows = list(session.scalars(select(RawLogMessageModel).order_by(RawLogMessageModel.sequence)))
        assert len(rows) == 5
        assert all(
            UUID(row.id).version == 7 and UUID(row.request_id).version == 7
            for row in rows
            if row.request_id != "external"
        )
        assert all(row.updated_at == row.created_at for row in rows)
        assert list(session.scalars(select(ContextSnapshotModel))) == []
    assert await storage.append(session_id, "later", UserMessage(content="Next")) == 6


async def test_configured_request_id_is_persisted(storage):
    agent = await Agent.create(
        Model(),
        config=AgentConfig("test-session", request_id="request-from-application"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await agent.run("Hello")
    assert {record.request_id for record in storage.list_messages("test-session")} == {"request-from-application"}


async def test_compaction_snapshot_keeps_raw_log_and_restores_checkpoint(storage):
    session_id = "test-session"
    await storage.append(session_id, "old", UserMessage(content="Old " * 500))
    await storage.append(session_id, "old", AssistantMessage(content="Old answer"))
    agent = await Agent.create(
        Model(),
        config=AgentConfig(session_id),
        extensions=[
            SessionPersistenceExtension(storage),
            CompactionExtension(Model("Checkpoint"), max_tokens=100, keep_recent_tokens=1),
        ],
    )
    await agent.run("Latest")
    view = await storage.load(session_id)
    assert view.snapshot is not None
    assert view.snapshot.version == 1
    assert view.snapshot.created_at == view.snapshot.updated_at
    assert view.snapshot.created_at.tzinfo is not None
    assert isinstance(view.messages[0], CompactedMessage)
    assert view.snapshot.compacted_through_sequence == 2
    assert [record.sequence for record in view.raw_tail] == [3, 4]
    with Session(storage.engine) as session:
        rows = list(session.scalars(select(RawLogMessageModel).order_by(RawLogMessageModel.sequence)))
        assert len(rows) == 4
        assert rows[0].content == "Old " * 500
        snapshot = session.scalar(select(ContextSnapshotModel))
        assert UUID(snapshot.id).version == 7
        assert snapshot.compacted_through_sequence == 2
        assert decode_messages(snapshot.message_json) == [view.snapshot.compacted_message]
    assert view.messages[-1] == AssistantMessage(content="Answer")
    # A fresh Agent restores the checkpoint plus the later answer exactly once.
    restored_model = Model()
    restored = await Agent.create(
        restored_model, config=AgentConfig(session_id), extensions=[SessionPersistenceExtension(storage)]
    )
    assert restored.state.messages == []
    await restored.run("Continue")
    assert list(restored_model.requests[0].messages[1:-1]) == view.messages
    latest = await storage.load(session_id)
    assert latest.snapshot is not None
    assert latest.snapshot.version == 1
    assert latest.raw_tail[-1].sequence == 6


async def test_new_snapshot_advances_boundary_without_copying_raw_tail(storage):
    session_id = "test-session"
    await storage.append(session_id, "request", UserMessage(content="One"))
    await storage.append(session_id, "request", AssistantMessage(content="Two"))
    await storage.append(session_id, "request", UserMessage(content="Three"))

    first = await storage.snapshot(session_id, CompactedMessage(content="Checkpoint one"), 1, 0)
    second = await storage.snapshot(session_id, CompactedMessage(content="Checkpoint two"), 2, first.version)
    view = await storage.load(session_id)

    assert view.snapshot == second
    assert view.messages == [CompactedMessage(content="Checkpoint two"), UserMessage(content="Three")]
    assert [record.sequence for record in view.raw_tail] == [3]
    with Session(storage.engine) as session:
        snapshots = list(session.scalars(select(ContextSnapshotModel).order_by(ContextSnapshotModel.version)))
        assert [snapshot.compacted_through_sequence for snapshot in snapshots] == [1, 2]
        assert [decode_messages(snapshot.message_json) for snapshot in snapshots] == [
            [CompactedMessage(content="Checkpoint one")],
            [CompactedMessage(content="Checkpoint two")],
        ]


def test_message_codec_preserves_multimodal_and_tool_replay():
    messages = [
        SystemMessage(content="System"),
        UserMessage(content=[ImageContent(ImageBytesSource(b"\xff\x89PNG", "image/png"))]),
        AssistantMessage(
            content="Looking",
            reasoning="Reasoning",
            provider="anthropic",
            replay_blocks=({"type": "thinking", "signature": "signed"},),
            tool_calls=(ToolCall("id", "read", {"path": "notes.md"}),),
        ),
        ToolMessage(tool_call_id="id", name="read", content="Text"),
        CompactedMessage(content="Checkpoint"),
    ]
    assert decode_messages(encode_messages(messages)) == messages


async def test_lists_typed_raw_messages_and_deletes_one_session(storage):
    await storage.append("first", "request-1", UserMessage(content="Hello"))
    await storage.append("first", "request-1", AssistantMessage(content="Hi"))
    await storage.append("second", "request-2", UserMessage(content="Keep"))

    records = storage.list_messages("first", after_sequence=0, through_sequence=1)
    assert len(records) == 1
    assert records[0].request_id == "request-1"
    assert records[0].message == UserMessage(content="Hello")
    assert records[0].created_at.tzinfo is not None
    assert storage.count_messages("first") == 2

    sessions = storage.list_sessions()
    assert [(session.session_id, session.message_count) for session in sessions] == [("second", 1), ("first", 2)]
    assert all(session.created_at.tzinfo is not None and session.updated_at.tzinfo is not None for session in sessions)
    assert len(storage.list_sessions(limit=1)) == 1

    assert storage.delete_session("first") is True
    assert storage.list_messages("first") == []
    assert storage.count_messages("first") == 0
    assert [record.message for record in storage.list_messages("second")] == [UserMessage(content="Keep")]
    assert storage.delete_session("missing") is False


@pytest.mark.parametrize(
    "options,error",
    [
        ({"after_sequence": -1}, "after_sequence"),
        ({"through_sequence": 0}, "through_sequence"),
        ({"limit": 0}, "limit"),
    ],
)
def test_list_messages_rejects_invalid_ranges(storage, options, error):
    with pytest.raises(ValueError, match=error):
        storage.list_messages("session", **options)


def test_list_sessions_rejects_invalid_limit(storage):
    with pytest.raises(ValueError, match="limit"):
        storage.list_sessions(limit=0)


async def test_storage_rejects_invalid_checkpoint_writes(storage):
    await storage.append("session", "request", UserMessage(content="One"))

    with pytest.raises(ValueError, match="Checkpoints belong"):
        await storage.append("session", "request", CompactedMessage(content="Invalid"))

    snapshot = await storage.snapshot("session", CompactedMessage(content="First"), 1, 0)
    with pytest.raises(ValueError, match="Snapshot conflict"):
        await storage.snapshot("session", CompactedMessage(content="Stale"), 1, 0)
    with pytest.raises(ValueError, match="existing Raw Log"):
        await storage.snapshot("session", CompactedMessage(content="Outside"), 2, snapshot.version)
    with pytest.raises(ValueError, match="must advance"):
        await storage.snapshot("session", CompactedMessage(content="Repeated"), 1, snapshot.version)
