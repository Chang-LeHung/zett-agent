"""Short integration scenarios against a disposable real SQLite database."""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zett_agent import (
    Agent,
    AgentConfig,
    AgentExtension,
    AssistantMessage,
    CompactionExtension,
    ModelEvent,
    ModelResponse,
    SQLiteSessionExtension,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool,
)
from zett_agent.extensions.compaction import CompactedMessage
from zett_agent.storage import AgentSessionModel, ContextSnapshotModel, MessageKind, RawLogMessageModel


class AnswerModel:
    """Return one deterministic answer through the real Agent lifecycle."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def stream(self, request):
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=self.answer)))


class ToolRoundModel:
    """Request one deterministic tool call, then produce a final answer."""

    async def stream(self, request):
        if any(isinstance(message, ToolMessage) for message in request.messages):
            message = AssistantMessage(content="The result is 5")
        else:
            message = AssistantMessage(
                reasoning="A calculation tool will provide the exact result.",
                tool_calls=(ToolCall("add-1", "add", {"left": 2, "right": 3}),),
            )
        yield ModelEvent.completed(ModelResponse(message))


@tool
def add(left: int, right: int) -> int:
    """Add two integers.

    Args:
        left: First integer.
        right: Second integer.

    Snippet:
        add(left=2, right=3)

    Guidelines:
        - Use this tool when an exact integer sum is required.
    """
    return left + right


@pytest.fixture
def sqlite_extension(tmp_path: Path):
    """Own one real SQLite file and remove it after every scenario."""
    path = tmp_path / "persistence-scenario.sqlite3"
    extension = SQLiteSessionExtension(path)
    try:
        yield extension
    finally:
        extension.close()
        path.unlink(missing_ok=True)


async def test_successful_turn_writes_valid_session_and_raw_log_rows(sqlite_extension):
    class Classifier(AgentExtension):
        async def on_message(self, context):
            context.metadata["classified_by"] = "extension"
            context.tags["reviewed"] = True

    agent = await Agent.create(
        AnswerModel("Stored answer"),
        config=AgentConfig("session"),
        extensions=[Classifier(), sqlite_extension],
    )

    user_message = UserMessage(content="Store this question")
    await agent.run(
        "Store this question",
        metadata={"source": "integration-test"},
        tags={"kind": "test"},
    )
    sqlite_extension.update_session("session", title="SQLite scenario", agent_name="test-agent")

    view = await sqlite_extension.storage.load("session")
    assert view.messages == [user_message, AssistantMessage(content="Stored answer")]
    assert [(record.metadata, record.tags) for record in view.raw_tail] == [
        (
            {"source": "integration-test", "classified_by": "extension"},
            {"kind": "test", "reviewed": True},
        ),
        (
            {"source": "integration-test", "classified_by": "extension"},
            {"kind": "test", "reviewed": True},
        ),
    ]
    assert (view.title, view.agent_name) == ("SQLite scenario", "test-agent")
    with Session(sqlite_extension.storage.engine) as session:
        rows = list(session.scalars(select(RawLogMessageModel).order_by(RawLogMessageModel.sequence)))
        assert [row.sequence for row in rows] == [1, 2]
        assert all(UUID(row.id).version == 7 and row.created_at == row.updated_at for row in rows)


async def test_complete_tool_turn_preserves_roles_and_session_view(sqlite_extension):
    agent = await Agent.create(
        ToolRoundModel(),
        config=AgentConfig("tool-session"),
        tools=[add],
        extensions=[sqlite_extension],
    )

    result = await agent.run("Calculate 2 + 3")

    expected_messages = [
        UserMessage(content="Calculate 2 + 3"),
        AssistantMessage(
            reasoning="A calculation tool will provide the exact result.",
            tool_calls=(ToolCall("add-1", "add", {"left": 2, "right": 3}),),
        ),
        ToolMessage(tool_call_id="add-1", name="add", content="5"),
        AssistantMessage(content="The result is 5"),
    ]
    records = sqlite_extension.list_raw_messages("tool-session")
    view = await sqlite_extension.storage.load("tool-session")

    assert result == expected_messages[-1]
    assert [record.message for record in records] == expected_messages
    assert view.snapshot is None
    assert view.raw_tail == records
    assert view.messages == expected_messages
    with Session(sqlite_extension.storage.engine) as session:
        rows = list(session.scalars(select(RawLogMessageModel).order_by(RawLogMessageModel.sequence)))
        assert [row.role for row in rows] == [
            int(MessageKind.USER),
            int(MessageKind.ASSISTANT),
            int(MessageKind.TOOL),
            int(MessageKind.ASSISTANT),
        ]
        assert [row.sequence for row in rows] == [1, 2, 3, 4]


async def test_compaction_writes_snapshot_without_rewriting_raw_log(sqlite_extension):
    storage = sqlite_extension.storage
    original = UserMessage(content="Old context " * 200)
    await storage.append("session", "old-request", original)
    await storage.append("session", "old-request", AssistantMessage(content="Old answer"))
    agent = await Agent.create(
        AnswerModel("Current answer"),
        config=AgentConfig("session"),
        extensions=[
            sqlite_extension,
            CompactionExtension(AnswerModel("Compact checkpoint"), max_tokens=50, keep_recent_tokens=1),
        ],
    )

    await agent.run("Current question")

    with Session(storage.engine) as session:
        raw_rows = list(session.scalars(select(RawLogMessageModel).order_by(RawLogMessageModel.sequence)))
        snapshots = list(session.scalars(select(ContextSnapshotModel)))
        assert [row.content for row in raw_rows] == [
            original.content,
            "Old answer",
            "Current question",
            "Current answer",
        ]
        assert len(snapshots) == 1
        assert snapshots[0].compacted_through_sequence == 2


async def test_delete_session_explicitly_cleans_all_sqlite_rows(sqlite_extension):
    storage = sqlite_extension.storage
    await storage.append("session", "request", UserMessage(content="Delete me"))
    await storage.snapshot("session", CompactedMessage(content="Checkpoint"), 1, 0)

    assert sqlite_extension.delete_session("session") is True

    with Session(storage.engine) as session:
        assert session.scalar(select(func.count(AgentSessionModel.id))) == 0
        assert session.scalar(select(func.count(RawLogMessageModel.id))) == 0
        assert session.scalar(select(func.count(ContextSnapshotModel.id))) == 0


async def test_update_session_title_returns_summary_and_preserves_history(sqlite_extension):
    storage = sqlite_extension.storage
    await storage.append(
        "session",
        "request",
        UserMessage(content="Keep this message"),
        title="Original title",
        agent_name="coding-agent",
        metadata={"owner": "test"},
        tags={"kind": "root"},
    )
    original = storage.list_raw_messages("session")

    updated = sqlite_extension.update_session(
        "session",
        title="  Renamed session  ",
        agent_name="  review-agent  ",
    )

    assert updated is not None
    assert updated.title == "Renamed session"
    assert updated.agent_name == "review-agent"
    assert updated.message_count == 1
    assert sqlite_extension.get_session("session") == updated
    assert (await storage.load("session")).title == "Renamed session"
    assert storage.list_raw_messages("session") == original
    assert sqlite_extension.get_session("missing") is None
    assert sqlite_extension.update_session("missing", title="Valid title") is None


@pytest.mark.parametrize("title", ["", "   ", "x" * 201])
def test_update_session_rejects_invalid_titles(sqlite_extension, title):
    with pytest.raises(ValueError, match="Session title"):
        sqlite_extension.update_session("session", title=title)


@pytest.mark.parametrize("agent_name", ["", "   ", "x" * 65])
def test_update_session_rejects_invalid_agent_names(sqlite_extension, agent_name):
    with pytest.raises(ValueError, match="Agent name"):
        sqlite_extension.update_session("session", agent_name=agent_name)


def test_update_session_requires_at_least_one_field(sqlite_extension):
    with pytest.raises(ValueError, match="At least one"):
        sqlite_extension.update_session("session")
