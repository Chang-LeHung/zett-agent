import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import UUID

import pytest

from zett_agent import (
    Agent,
    AgentExtension,
    AgentRunConfig,
    AssistantMessage,
    BaseSessionPersistenceExtension,
    CompactionExtension,
    ImageBytesSource,
    ImageContent,
    ModelEvent,
    ModelResponse,
    ModelUsage,
    SessionPersistenceExtension,
    SQLiteSessionExtension,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from zett_agent.extensions.compaction import CompactedMessage
from zett_agent.storage import (
    SQLiteSessionStorage,
    decode_messages,
    encode_messages,
)


@contextmanager
def sqlite_file(path: Path) -> Iterator[sqlite3.Connection]:
    """Read the storage file with the driver-independent stdlib API.

    The asynchronous engine owns an aiosqlite DBAPI that has no synchronous
    counterpart, so schema and row assertions use their own connection.
    """
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            # Assertions may also rewrite rows to simulate corrupt storage.
            connection.commit()


def table_columns(path: Path, table: str) -> set[str]:
    with sqlite_file(path) as connection:
        return {row[1] for row in connection.execute(f"pragma table_info({table})")}


@pytest.fixture
async def storage(tmp_path):
    storage = SQLiteSessionStorage(tmp_path / "sessions.sqlite3")
    yield storage
    await storage.close()


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
    await storage.close()
    reopened = SQLiteSessionStorage(path)
    try:
        view = await reopened.load("session")
        assert view.messages == [UserMessage(content="Persisted")]
        assert view.snapshot is None
        assert view.raw_tail[-1].sequence == 1
        assert (await reopened.load("other-session")).messages == []
    finally:
        await reopened.close()


async def test_reopening_old_database_adds_missing_indexes(tmp_path):
    path = tmp_path / "agent.sqlite3"
    storage = SQLiteSessionStorage(path)
    await storage.append("session", "request", UserMessage(content="Persisted"))
    await storage.close()
    with sqlite_file(path) as connection:
        connection.execute("drop index if exists ix_agent_sessions_activity")

    reopened = SQLiteSessionStorage(path)
    try:
        await reopened.list_sessions()
    finally:
        await reopened.close()

    with sqlite_file(path) as connection:
        indexes = {
            row[0] for row in connection.execute("select name from sqlite_master where type = 'index'") if row[0]
        }
    assert "ix_agent_sessions_activity" in indexes


async def test_reopening_old_database_adds_session_type_column(tmp_path):
    path = tmp_path / "agent.sqlite3"
    storage = SQLiteSessionStorage(path)
    await storage.append("session", "request", UserMessage(content="Persisted"))
    await storage.close()
    with sqlite_file(path) as connection:
        connection.execute("alter table agent_sessions drop column session_type")

    reopened = SQLiteSessionStorage(path)
    try:
        summaries = await reopened.list_sessions()
    finally:
        await reopened.close()

    assert summaries[0].session_type == 0
    assert "session_type" in table_columns(path, "agent_sessions")


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
    summaries = {summary.session_id: summary for summary in await storage.list_sessions()}

    assert view.parent_session_id == "parent"
    assert view.title == "Inspect session storage"
    assert view.agent_name == "explore"
    assert view.raw_tail[0].metadata == {"depth": 1, "subagent_type": "explore"}
    assert view.raw_tail[0].tags == {"domain": "code", "read_only": True}
    assert summaries["parent"].parent_session_id is None
    assert summaries["child"].parent_session_id == "parent"
    assert summaries["child"].title == "Inspect session storage"
    assert summaries["child"].agent_name == "explore"
    with sqlite_file(storage.path) as connection:
        parent_of_child = connection.execute(
            "select parent_session_id from agent_sessions where id = ?", ("child",)
        ).fetchone()
        assert parent_of_child["parent_session_id"] == "parent"
        columns = table_columns(storage.path, "agent_sessions")
        assert "metadata_json" not in columns
        assert "tags_json" not in columns
        raw = connection.execute(
            "select metadata_json, tags_json, message_json from raw_messages where session_id = ?", ("child",)
        ).fetchone()
        assert json.loads(raw["metadata_json"]) == {"subagent_type": "explore", "depth": 1}
        assert json.loads(raw["tags_json"]) == {"domain": "code", "read_only": True}
        assert set(json.loads(raw["message_json"])[0]["data"]) == {
            "attributes",
            "content",
            "include_in_messages",
            "persist",
        }

    with pytest.raises(ValueError, match="parent cannot change"):
        await storage.append(
            "child",
            "another-request",
            AssistantMessage(content="Invalid"),
            parent_session_id="another-parent",
        )

    with sqlite_file(storage.path) as connection:
        assert all(
            list(connection.execute(f"pragma foreign_key_list({table})")) == []
            for table in ("agent_sessions", "raw_messages", "session_snapshots")
        )
    assert await storage.delete_session("parent") is True
    assert (await storage.load("child")).parent_session_id == "parent"
    assert await storage.count_messages("child") == 1


async def test_agent_persists_provider_usage_on_each_assistant_raw_message(storage):
    class UsageModel:
        async def stream(self, request):
            yield ModelEvent.completed(
                ModelResponse(
                    AssistantMessage(content="Measured", provider="test"),
                    usage=ModelUsage(
                        input_tokens=100,
                        output_tokens=25,
                        cache_read_tokens=80,
                        cache_write_tokens=5,
                        reasoning_tokens=10,
                    ),
                )
            )

    agent = await Agent.create(
        UsageModel(),
        config=AgentRunConfig("usage-session"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await agent.run("Measure this request")

    records = await storage.list_raw_messages("usage-session")
    system, user, assistant = records
    assert isinstance(system.message, SystemMessage)
    assert user.input_tokens is None
    assert user.total_tokens is None
    assert user.cache_hit_rate is None
    assert assistant.input_tokens == 100
    assert assistant.output_tokens == 25
    assert assistant.cache_read_tokens == 80
    assert assistant.cache_write_tokens == 5
    assert assistant.reasoning_tokens == 10
    assert assistant.total_tokens == 125
    assert assistant.cache_hit_rate == pytest.approx(0.8)

    assert {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    } <= table_columns(storage.path, "raw_messages")


async def test_storage_rejects_model_usage_on_non_assistant_messages(storage):
    with pytest.raises(ValueError, match="only to assistant"):
        await storage.append(
            "invalid-usage",
            "request",
            UserMessage(content="Not a model response"),
            usage=ModelUsage(input_tokens=1),
        )


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
    with sqlite_file(storage.path) as connection:
        stored_message = connection.execute("select message_json from raw_messages limit 1").fetchone()[0]
        assert stored_message is not None
        assert "Updated title" not in stored_message

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
    with sqlite_file(storage.path) as connection:
        connection.execute("update raw_messages set metadata_json = '[]' where session_id = 'corrupt'")

    with pytest.raises(ValueError, match="must be a JSON object"):
        await storage.list_raw_messages("corrupt")


async def test_persistence_restores_a_child_parent_when_config_omits_it(storage):
    first = await Agent.create(
        Model("Child answer"),
        config=AgentRunConfig(
            "child",
            parent_session_id="parent",
        ),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await first.run("Child question", metadata={"scope": "storage"}, tags={"domain": "code"})

    restored = await Agent.create(
        Model("Continued"),
        config=AgentRunConfig("child"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await restored.run("Continue")

    assert restored.state.parent_session_id == "parent"
    restored_record = (await storage.list_raw_messages("child"))[0]
    assert restored_record.metadata == {"scope": "storage"}
    assert restored_record.tags == {"domain": "code"}
    assert (await storage.load("child")).parent_session_id == "parent"


async def test_persistence_rejects_changing_an_existing_root_into_a_child(storage):
    await storage.append("root", "request", UserMessage(content="Existing root"))
    agent = await Agent.create(
        Model(),
        config=AgentRunConfig("root", parent_session_id="parent"),
        extensions=[SessionPersistenceExtension(storage)],
    )

    with pytest.raises(ValueError, match="different parent"):
        await agent.run("Invalid continuation")


async def test_sqlite_session_extension_owns_storage_and_restores_history(tmp_path):
    path = tmp_path / "owned.sqlite3"
    first = SQLiteSessionExtension(path)
    try:
        agent = await Agent.create(Model("First answer"), config=AgentRunConfig("session"), extensions=[first])
        await agent.run("First question")
    finally:
        await first.close()

    second = SQLiteSessionExtension(path)
    model = Model("Second answer")
    try:
        agent = await Agent.create(model, config=AgentRunConfig("session"), extensions=[second])
        await agent.run("Second question")
        assert [message.content for message in model.requests[0].messages] == [
            "You are a helpful assistant.",
            "First question",
            "First answer",
            "Second question",
        ]
    finally:
        await second.close()


async def test_system_messages_are_stored_but_not_replayed(storage):
    class DynamicInstructions(AgentExtension):
        async def on_state(self, context):
            context.add_message(SystemMessage(content="Dynamic instructions"), index=0)

    first = await Agent.create(
        Model("First answer"),
        system_prompt="Base instructions",
        config=AgentRunConfig("system-log"),
        extensions=[SessionPersistenceExtension(storage), DynamicInstructions()],
    )
    await first.run("First question")

    records = await storage.list_raw_messages("system-log")
    assert [record.message.content for record in records] == [
        "Dynamic instructions",
        "Base instructions",
        "First question",
        "First answer",
    ]

    second_model = Model("Second answer")
    second = await Agent.create(
        second_model,
        system_prompt="Current instructions",
        config=AgentRunConfig("system-log"),
        extensions=[SessionPersistenceExtension(storage), DynamicInstructions()],
    )
    await second.run("Second question")

    instructions = [
        message.content for message in second_model.requests[0].messages if isinstance(message, SystemMessage)
    ]
    assert instructions == ["Dynamic instructions", "Current instructions"]
    assert sum(content == "Dynamic instructions" for content in instructions) == 1


async def test_session_view_filters_raw_messages_marked_context_only(storage):
    await storage.append("context-filter", "request", SystemMessage(content="Raw system"))
    await storage.append(
        "context-filter",
        "request",
        UserMessage(content="Raw only", include_in_messages=False),
    )
    await storage.append("context-filter", "request", UserMessage(content="Replayable"))

    view = await storage.load("context-filter")

    assert [record.message.content for record in view.raw_tail] == [
        "Raw system",
        "Raw only",
        "Replayable",
    ]
    assert [message.content for message in view.messages] == ["Replayable"]


async def test_custom_persistence_extension_only_wires_its_storage(tmp_path):
    class CustomSessionExtension(BaseSessionPersistenceExtension[SQLiteSessionStorage]):
        def __init__(self, path):
            super().__init__(SQLiteSessionStorage(path))

        async def close(self) -> None:
            await self.storage.close()

    extension = CustomSessionExtension(tmp_path / "custom.sqlite3")
    try:
        first = await Agent.create(Model("Stored"), config=AgentRunConfig("session"), extensions=[extension])
        await first.run("Remember this")
        second_model = Model("Continued")
        second = await Agent.create(second_model, config=AgentRunConfig("session"), extensions=[extension])
        await second.run("Continue")

        assert [message.content for message in second_model.requests[0].messages[1:-1]] == [
            "Remember this",
            "Stored",
        ]
    finally:
        await extension.close()


async def test_raw_log_persists_model_output_timing(storage):
    class StreamingModel:
        async def stream(self, request):
            message = AssistantMessage(content="Answer", reasoning="Think")
            yield ModelEvent.reasoning("Think")
            yield ModelEvent.text("Answer")
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(
        StreamingModel(),
        config=AgentRunConfig("timed-session"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await agent.run("Question")

    system, user, assistant = await storage.list_raw_messages("timed-session")
    assert isinstance(system.message, SystemMessage)
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

        records = await extension.list_raw_messages("session", limit=1, offset=1)

        assert [record.sequence for record in records] == [2]
        assert [record.message for record in records] == [AssistantMessage(content="Second")]

        await extension.storage.append("newer", "request", UserMessage(content="Third"))
        sessions = await extension.list_sessions(limit=1, offset=1)
        assert [summary.session_id for summary in sessions] == ["session"]
    finally:
        await extension.close()


async def test_same_agent_reloads_snapshot_and_external_tail_every_request(storage):
    session_id = "test-session"
    model = Model()
    agent = await Agent.create(
        model, config=AgentRunConfig(session_id), extensions=[SessionPersistenceExtension(storage)]
    )
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
    assert [record.message.role for record in second.raw_tail] == [
        "system",
        "user",
        "assistant",
        "user",
        "system",
        "user",
        "assistant",
    ]
    assert second.raw_tail[-1].sequence == 7
    with sqlite_file(storage.path) as connection:
        rows = connection.execute(
            "select id, request_id, created_at, updated_at from raw_messages order by sequence"
        ).fetchall()
        assert len(rows) == 7
        assert all(
            UUID(row["id"]).version == 7 and UUID(row["request_id"]).version == 7
            for row in rows
            if row["request_id"] != "external"
        )
        assert all(row["updated_at"] == row["created_at"] for row in rows)
        assert connection.execute("select count(*) from session_snapshots").fetchone()[0] == 0
    assert await storage.append(session_id, "later", UserMessage(content="Next")) == 8


async def test_configured_request_id_is_persisted(storage):
    agent = await Agent.create(
        Model(),
        config=AgentRunConfig("test-session", request_id="request-from-application"),
        extensions=[SessionPersistenceExtension(storage)],
    )
    await agent.run("Hello")
    records = await storage.list_raw_messages("test-session")
    assert {record.request_id for record in records} == {"request-from-application"}


async def test_compaction_snapshot_keeps_raw_log_and_restores_checkpoint(storage):
    session_id = "test-session"
    await storage.append(session_id, "old", UserMessage(content="Old " * 500))
    await storage.append(session_id, "old", AssistantMessage(content="Old answer"))
    agent = await Agent.create(
        Model(),
        config=AgentRunConfig(session_id),
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
    assert [record.sequence for record in view.raw_tail] == [3, 4, 5]
    with sqlite_file(storage.path) as connection:
        rows = connection.execute("select content, message_json from raw_messages order by sequence").fetchall()
        assert len(rows) == 5
        assert rows[0]["content"] == "Old " * 500
        snapshot = connection.execute(
            "select id, compacted_through_sequence, message_json from session_snapshots"
        ).fetchone()
        assert UUID(snapshot["id"]).version == 7
        assert snapshot["compacted_through_sequence"] == 2
        assert decode_messages(snapshot["message_json"]) == [view.snapshot.compacted_message]
    assert view.messages[-1] == AssistantMessage(content="Answer")
    # A fresh Agent restores the checkpoint plus the later answer exactly once.
    restored_model = Model()
    restored = await Agent.create(
        restored_model, config=AgentRunConfig(session_id), extensions=[SessionPersistenceExtension(storage)]
    )
    assert restored.state.messages == ()
    await restored.run("Continue")
    assert list(restored_model.requests[0].messages[1:-1]) == view.messages
    latest = await storage.load(session_id)
    assert latest.snapshot is not None
    assert latest.snapshot.version == 1
    assert latest.raw_tail[-1].sequence == 8


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
    with sqlite_file(storage.path) as connection:
        snapshots = connection.execute(
            "select compacted_through_sequence, message_json from session_snapshots order by version"
        ).fetchall()
        assert [snapshot["compacted_through_sequence"] for snapshot in snapshots] == [1, 2]
        assert [decode_messages(snapshot["message_json"]) for snapshot in snapshots] == [
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
            attributes={"trace": "model-step-1", "attempt": 1},
            provider="anthropic",
            model="claude-test",
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
    corrupt = encode_messages([UserMessage(content="Not a checkpoint")])
    with sqlite_file(storage.path) as connection:
        connection.execute("update session_snapshots set message_json = ? where id = ?", (corrupt, snapshot.id))

    with pytest.raises(ValueError, match="exactly one CompactedMessage"):
        await storage.load("session")


async def test_lists_typed_raw_messages_and_deletes_one_session(storage):
    await storage.append("first", "request-1", UserMessage(content="Hello"))
    await storage.append("first", "request-1", AssistantMessage(content="Hi"))
    await storage.append("second", "request-2", UserMessage(content="Keep"))

    records = await storage.list_raw_messages("first", after_sequence=0, through_sequence=1)
    assert len(records) == 1
    assert records[0].request_id == "request-1"
    assert records[0].message == UserMessage(content="Hello")
    assert records[0].created_at.tzinfo is not None
    assert await storage.count_messages("first") == 2
    second_page = await storage.list_raw_messages("first", limit=1, offset=1)
    assert [record.sequence for record in second_page] == [2]
    assert [record.message for record in second_page] == [AssistantMessage(content="Hi")]
    assert await storage.list_raw_messages("first", limit=1, offset=2) == []

    sessions = await storage.list_sessions()
    assert [(session.session_id, session.message_count) for session in sessions] == [("second", 1), ("first", 2)]
    assert all(session.created_at.tzinfo is not None and session.updated_at.tzinfo is not None for session in sessions)
    assert len(await storage.list_sessions(limit=1)) == 1
    assert await storage.list_sessions(limit=1, offset=0) == sessions[:1]
    assert await storage.list_sessions(limit=1, offset=1) == sessions[1:]
    assert await storage.list_sessions(limit=1, offset=2) == []

    assert await storage.delete_session("first") is True
    assert await storage.list_raw_messages("first") == []
    assert await storage.count_messages("first") == 0
    kept = await storage.list_raw_messages("second")
    assert [record.message for record in kept] == [UserMessage(content="Keep")]
    assert await storage.delete_session("missing") is False


@pytest.mark.parametrize(
    "options,error",
    [
        ({"after_sequence": -1}, "after_sequence"),
        ({"through_sequence": 0}, "through_sequence"),
        ({"limit": 0}, "limit"),
        ({"offset": -1}, "offset"),
    ],
)
async def test_list_raw_messages_rejects_invalid_query_options(storage, options, error):
    with pytest.raises(ValueError, match=error):
        await storage.list_raw_messages("session", **options)


@pytest.mark.parametrize("options,error", [({"limit": 0}, "limit"), ({"offset": -1}, "offset")])
async def test_list_sessions_rejects_invalid_query_options(storage, options, error):
    with pytest.raises(ValueError, match=error):
        await storage.list_sessions(**options)


async def test_create_session_persists_an_empty_conversation(storage):
    created = await storage.create_session(session_id="empty-session", title="Empty", agent_name="Zett Agent")

    assert created.session_id == "empty-session"
    assert created.session_type == 0
    assert created.title == "Empty"
    assert created.agent_name == "Zett Agent"
    assert created.message_count == 0
    assert await storage.get_session("empty-session") == created
    assert await storage.list_raw_messages("empty-session") == []
    with pytest.raises(ValueError, match="already exists"):
        await storage.create_session(session_id="empty-session")


async def test_sessions_can_be_filtered_by_integer_type_code(storage):
    normal = await storage.create_session(session_id="normal-session")
    scheduled = await storage.create_session(
        session_id="scheduled-session",
        session_type=1,
    )
    # The runtime stores application-defined codes without owning the vocabulary.
    custom = await storage.create_session(session_id="custom-session", session_type=99)

    assert normal.session_type == 0
    assert scheduled.session_type == 1
    assert custom.session_type == 99
    assert [item.session_id for item in await storage.list_sessions(session_types=(1,))] == ["scheduled-session"]
    assert [item.session_id for item in await storage.list_sessions(session_types=(99,))] == ["custom-session"]
    assert await storage.get_session("custom-session") == custom
    assert [item.session_id for item in await storage.list_sessions()] == [
        "custom-session",
        "scheduled-session",
        "normal-session",
    ]


@pytest.mark.parametrize(
    ("session_type", "error"),
    [
        (-1, "cannot be negative"),
        (1.5, "must be an integer code"),
        ("standard", "must be an integer code"),
        (True, "must be an integer code"),
    ],
)
async def test_create_session_rejects_invalid_type_codes(storage, session_type, error):
    with pytest.raises(ValueError, match=error):
        await storage.create_session(session_id="invalid-session", session_type=session_type)


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
