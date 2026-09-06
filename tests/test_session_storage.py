import json
from uuid import UUID

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from zett_agent import (
    Agent,
    AgentConfig,
    AssistantMessage,
    BaseSessionPersistenceExtension,
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
from zett_agent.extensions.compaction import CompactedMessage
from zett_agent.storage import (
    AgentSessionModel,
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


async def test_storage_persists_parent_identity_and_message_metadata(storage):
    await storage.append("parent", "parent-request", UserMessage(content="Parent"))
    await storage.append(
        "child",
        "child-request",
        UserMessage(content="Child"),
        parent_session_id="parent",
        title="Inspect session storage",
        agent_name="explore",
        metadata={"subagent_type": "explore", "depth": 1},
        tags={"domain": "code", "read_only": True},
    )

    view = await storage.load("child")
    summaries = {summary.session_id: summary for summary in storage.list_sessions()}

    assert view.parent_session_id == "parent"
    assert view.title == "Inspect session storage"
    assert view.agent_name == "explore"
    assert view.raw_tail[0].metadata == {"depth": 1, "subagent_type": "explore"}
    assert view.raw_tail[0].tags == {"domain": "code", "read_only": True}
    assert summaries["parent"].parent_session_id is None
    assert summaries["child"].parent_session_id == "parent"
    assert summaries["child"].title == "Inspect session storage"
    assert summaries["child"].agent_name == "explore"
    with Session(storage.engine) as session:
        child = session.get(AgentSessionModel, "child")
        assert child.parent_session_id == "parent"
        columns = {column["name"] for column in inspect(storage.engine).get_columns("agent_sessions")}
        assert "metadata_json" not in columns
        assert "tags_json" not in columns
        raw = session.scalar(select(RawLogMessageModel).where(RawLogMessageModel.session_id == "child"))
        assert json.loads(raw.metadata_json) == {"subagent_type": "explore", "depth": 1}
        assert json.loads(raw.tags_json) == {"domain": "code", "read_only": True}
        assert set(json.loads(raw.message_json)[0]["data"]) == {"content"}

    with pytest.raises(ValueError, match="parent cannot change"):
        await storage.append(
            "child",
            "another-request",
            AssistantMessage(content="Invalid"),
            parent_session_id="another-parent",
        )

    assert all(
        inspect(storage.engine).get_foreign_keys(table) == []
        for table in ("agent_sessions", "raw_messages", "session_snapshots")
    )
    assert storage.delete_session("parent") is True
    assert (await storage.load("child")).parent_session_id == "parent"
    assert storage.count_messages("child") == 1


async def test_storage_updates_mutable_session_identity_without_copying_it_to_raw_log(storage):
    await storage.append("session", "first", UserMessage(content="One"))
    await storage.append(
        "session",
        "second",
        AssistantMessage(content="Two"),
        title="Updated title",
        agent_name="coding",
        metadata={"workspace": "/tmp/project"},
        tags={"domain": "coding"},
    )

    view = await storage.load("session")
    assert view.title == "Updated title"
    assert view.agent_name == "coding"
    assert view.raw_tail[1].metadata == {"workspace": "/tmp/project"}
    assert view.raw_tail[1].tags == {"domain": "coding"}
    with Session(storage.engine) as session:
        assert session.scalar(select(RawLogMessageModel.message_json).limit(1)) is not None
        assert "Updated title" not in session.scalar(select(RawLogMessageModel.message_json).limit(1))

    with pytest.raises(ValueError, match="Session title"):
        await storage.append("invalid", "request", UserMessage(content="One"), title="")
    with pytest.raises(ValueError, match="Agent name"):
        await storage.append("invalid", "request", UserMessage(content="One"), agent_name="x" * 65)


async def test_raw_message_context_rejects_invalid_input_and_corrupt_rows(storage):
    with pytest.raises(ValueError, match="Message metadata"):
        await storage.append("invalid", "request", UserMessage(content="One"), metadata={"value": object()})
    with pytest.raises(ValueError, match="Message tags keys"):
        await storage.append("invalid", "request", UserMessage(content="One"), tags={"": True})

    await storage.append("corrupt", "request", UserMessage(content="One"), metadata={"valid": True})
    with Session(storage.engine) as session, session.begin():
        row = session.scalar(select(RawLogMessageModel).where(RawLogMessageModel.session_id == "corrupt"))
        row.metadata_json = "[]"

    with pytest.raises(ValueError, match="must be a JSON object"):
        storage.list_raw_messages("corrupt")


async def test_persistence_restores_a_child_parent_when_config_omits_it(storage):
    first = await Agent.create(
        Model("Child answer"),
        config=AgentConfig(
            "child",
            parent_session_id="parent",
        ),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await first.run("Child question", metadata={"scope": "storage"}, tags={"domain": "code"})

    restored = await Agent.create(
        Model("Continued"),
        config=AgentConfig("child"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await restored.run("Continue")

    assert restored.state.parent_session_id == "parent"
    restored_record = storage.list_raw_messages("child")[0]
    assert restored_record.metadata == {"scope": "storage"}
    assert restored_record.tags == {"domain": "code"}
    assert (await storage.load("child")).parent_session_id == "parent"


async def test_persistence_rejects_changing_an_existing_root_into_a_child(storage):
    await storage.append("root", "request", UserMessage(content="Existing root"))
    agent = await Agent.create(
        Model(),
        config=AgentConfig("root", parent_session_id="parent"),
        extensions=[SessionPersistenceExtension(storage)],
    )

    with pytest.raises(ValueError, match="different parent"):
        await agent.run("Invalid continuation")


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


async def test_custom_persistence_extension_only_wires_its_storage(tmp_path):
    class CustomSessionExtension(BaseSessionPersistenceExtension[SQLiteSessionStorage]):
        def __init__(self, path):
            super().__init__(SQLiteSessionStorage(path))

        def close(self) -> None:
            self.storage.close()

    extension = CustomSessionExtension(tmp_path / "custom.sqlite3")
    try:
        first = await Agent.create(Model("Stored"), config=AgentConfig("session"), extensions=[extension])
        await first.run("Remember this")
        second_model = Model("Continued")
        second = await Agent.create(second_model, config=AgentConfig("session"), extensions=[extension])
        await second.run("Continue")

        assert [message.content for message in second_model.requests[0].messages[1:-1]] == [
            "Remember this",
            "Stored",
        ]
    finally:
        extension.close()


async def test_raw_log_persists_model_output_timing(storage):
    class StreamingModel:
        async def stream(self, request):
            message = AssistantMessage(content="Answer", reasoning="Think")
            yield ModelEvent.reasoning("Think")
            yield ModelEvent.text("Answer")
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(
        StreamingModel(),
        config=AgentConfig("timed-session"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await agent.run("Question")

    user, assistant = storage.list_raw_messages("timed-session")
    assert user.duration_ns == 0
    assert user.started_at == user.completed_at
    assert assistant.duration_ns >= 0
    assert assistant.reasoning_duration_ns is not None
    assert assistant.content_duration_ns is not None
    assert assistant.reasoning_started_at is not None
    assert assistant.reasoning_completed_at is not None
    assert assistant.content_started_at is not None
    assert assistant.content_completed_at is not None
    assert assistant.started_at.tzinfo is not None
    assert assistant.completed_at.tzinfo is not None


async def test_sqlite_session_extension_lists_paginated_raw_messages(tmp_path):
    extension = SQLiteSessionExtension(tmp_path / "owned.sqlite3")
    try:
        await extension.storage.append("session", "request", UserMessage(content="First"))
        await extension.storage.append("session", "request", AssistantMessage(content="Second"))

        records = extension.list_raw_messages("session", limit=1, offset=1)

        assert [record.sequence for record in records] == [2]
        assert [record.message for record in records] == [AssistantMessage(content="Second")]

        await extension.storage.append("newer", "request", UserMessage(content="Third"))
        sessions = extension.list_sessions(limit=1, offset=1)
        assert [summary.session_id for summary in sessions] == ["session"]
    finally:
        extension.close()


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
    assert {record.request_id for record in storage.list_raw_messages("test-session")} == {"request-from-application"}


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


def test_message_codec_rejects_values_outside_the_supported_envelope():
    message = AssistantMessage(replay_blocks=({"unsupported": object()},))
    with pytest.raises(TypeError, match="Unsupported message value"):
        encode_messages([message])


async def test_load_rejects_a_corrupted_snapshot_payload(storage):
    await storage.append("session", "request", UserMessage(content="One"))
    snapshot = await storage.snapshot("session", CompactedMessage(content="Checkpoint"), 1, 0)
    with Session(storage.engine) as session, session.begin():
        row = session.get(ContextSnapshotModel, snapshot.id)
        row.message_json = encode_messages([UserMessage(content="Not a checkpoint")])

    with pytest.raises(ValueError, match="exactly one CompactedMessage"):
        await storage.load("session")


async def test_lists_typed_raw_messages_and_deletes_one_session(storage):
    await storage.append("first", "request-1", UserMessage(content="Hello"))
    await storage.append("first", "request-1", AssistantMessage(content="Hi"))
    await storage.append("second", "request-2", UserMessage(content="Keep"))

    records = storage.list_raw_messages("first", after_sequence=0, through_sequence=1)
    assert len(records) == 1
    assert records[0].request_id == "request-1"
    assert records[0].message == UserMessage(content="Hello")
    assert records[0].created_at.tzinfo is not None
    assert storage.count_messages("first") == 2
    second_page = storage.list_raw_messages("first", limit=1, offset=1)
    assert [record.sequence for record in second_page] == [2]
    assert [record.message for record in second_page] == [AssistantMessage(content="Hi")]
    assert storage.list_raw_messages("first", limit=1, offset=2) == []

    sessions = storage.list_sessions()
    assert [(session.session_id, session.message_count) for session in sessions] == [("second", 1), ("first", 2)]
    assert all(session.created_at.tzinfo is not None and session.updated_at.tzinfo is not None for session in sessions)
    assert len(storage.list_sessions(limit=1)) == 1
    assert storage.list_sessions(limit=1, offset=0) == sessions[:1]
    assert storage.list_sessions(limit=1, offset=1) == sessions[1:]
    assert storage.list_sessions(limit=1, offset=2) == []

    assert storage.delete_session("first") is True
    assert storage.list_raw_messages("first") == []
    assert storage.count_messages("first") == 0
    assert [record.message for record in storage.list_raw_messages("second")] == [UserMessage(content="Keep")]
    assert storage.delete_session("missing") is False


@pytest.mark.parametrize(
    "options,error",
    [
        ({"after_sequence": -1}, "after_sequence"),
        ({"through_sequence": 0}, "through_sequence"),
        ({"limit": 0}, "limit"),
        ({"offset": -1}, "offset"),
    ],
)
def test_list_raw_messages_rejects_invalid_query_options(storage, options, error):
    with pytest.raises(ValueError, match=error):
        storage.list_raw_messages("session", **options)


@pytest.mark.parametrize("options,error", [({"limit": 0}, "limit"), ({"offset": -1}, "offset")])
def test_list_sessions_rejects_invalid_query_options(storage, options, error):
    with pytest.raises(ValueError, match=error):
        storage.list_sessions(**options)


def test_create_session_persists_an_empty_conversation(storage):
    created = storage.create_session(session_id="empty-session", title="Empty", agent_name="Zett Agent")

    assert created.session_id == "empty-session"
    assert created.title == "Empty"
    assert created.agent_name == "Zett Agent"
    assert created.message_count == 0
    assert storage.get_session("empty-session") == created
    assert storage.list_raw_messages("empty-session") == []
    with pytest.raises(ValueError, match="already exists"):
        storage.create_session(session_id="empty-session")


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
