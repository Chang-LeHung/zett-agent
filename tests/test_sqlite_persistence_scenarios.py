"""Short integration scenarios against a disposable real SQLite database."""

import asyncio
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
from zett_agent.storage import MessageKind


@contextmanager
def sqlite_file(path: Path) -> Iterator[sqlite3.Connection]:
    """Read the storage file directly; the async DBAPI has no sync counterpart."""
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        yield connection


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
async def sqlite_extension(tmp_path: Path):
    """Own one real SQLite file and remove it after every scenario."""
    path = tmp_path / "persistence-scenario.sqlite3"
    extension = SQLiteSessionExtension(path)
    try:
        yield extension
    finally:
        await extension.close()
        path.unlink(missing_ok=True)


async def test_successful_turn_writes_valid_session_and_raw_log_rows(sqlite_extension):
    class Classifier(AgentExtension):
        async def on_message(self, context):
            context.metadata["classified_by"] = "extension"
            context.tags["reviewed"] = True

    agent = await Agent.create(
        AnswerModel("Stored answer"),
        config=AgentRunConfig("session"),
        extensions=[Classifier(), sqlite_extension],
    )

    user_message = UserMessage(content="Store this question")
    await agent.run(
        "Store this question",
        metadata={"source": "integration-test"},
        tags={"kind": "test"},
    )
    await sqlite_extension.update_session("session", title="SQLite scenario", agent_name="test-agent")

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
    with sqlite_file(sqlite_extension.storage.path) as connection:
        rows = connection.execute("select id, sequence, created_at, updated_at from raw_messages").fetchall()
        assert [row["sequence"] for row in rows] == [1, 2]
        assert all(UUID(row["id"]).version == 7 and row["created_at"] == row["updated_at"] for row in rows)


async def test_complete_tool_turn_preserves_roles_and_session_view(sqlite_extension):
    agent = await Agent.create(
        ToolRoundModel(),
        config=AgentRunConfig("tool-session"),
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
    records = await sqlite_extension.list_raw_messages("tool-session")
    view = await sqlite_extension.storage.load("tool-session")

    assert result == expected_messages[-1]
    assert [record.message for record in records] == expected_messages
    assert view.snapshot is None
    assert view.raw_tail == records
    assert view.messages == expected_messages
    with sqlite_file(sqlite_extension.storage.path) as connection:
        rows = connection.execute("select role, sequence from raw_messages").fetchall()
        assert [row["role"] for row in rows] == [
            int(MessageKind.USER),
            int(MessageKind.ASSISTANT),
            int(MessageKind.TOOL),
            int(MessageKind.ASSISTANT),
        ]
        assert [row["sequence"] for row in rows] == [1, 2, 3, 4]


async def test_cancelled_tool_turn_is_persisted_as_provider_complete_and_can_resume(sqlite_extension):
    entered = asyncio.Event()

    @tool(guidelines="Wait until the request is cancelled.")
    async def wait_for_cancel() -> str:
        """Block so the test can cancel an active tool."""
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    class CancelThenAnswerModel:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(tool_calls=(ToolCall("wait-1", "wait_for_cancel"),))
                if len(self.requests) == 1
                else AssistantMessage(content="Resumed safely")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = CancelThenAnswerModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("cancel-resume"),
        tools=[wait_for_cancel],
        extensions=[sqlite_extension],
    )
    first = asyncio.create_task(agent.run("Start a tool"))
    await asyncio.wait_for(entered.wait(), 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    records = await sqlite_extension.list_raw_messages("cancel-resume")
    assert [type(record.message) for record in records] == [UserMessage, AssistantMessage, ToolMessage]
    cancelled_result = records[-1].message
    assert isinstance(cancelled_result, ToolMessage)
    assert cancelled_result.tool_call_id == "wait-1"
    assert cancelled_result.success is False
    assert cancelled_result.content == "Request cancelled before tool completion"

    result = await agent.run("Continue after stopping")

    assert result.content == "Resumed safely"
    restored = model.requests[-1].messages
    assistant_index = next(
        index for index, message in enumerate(restored) if isinstance(message, AssistantMessage) and message.tool_calls
    )
    assert isinstance(restored[assistant_index + 1], ToolMessage)
    assert restored[assistant_index + 1].tool_call_id == "wait-1"


async def test_restore_omits_legacy_incomplete_tool_batch_without_mutating_raw_log(sqlite_extension):
    storage = sqlite_extension.storage
    await storage.append("legacy", "request-1", UserMessage(content="Original request"))
    await storage.append(
        "legacy",
        "request-1",
        AssistantMessage(tool_calls=(ToolCall("missing-result", "add", {"left": 1, "right": 2}),)),
    )
    await storage.append("legacy", "request-2", UserMessage(content="Previously failed retry"))

    class CapturingModel(AnswerModel):
        def __init__(self) -> None:
            super().__init__("Recovered")
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            async for event in super().stream(request):
                yield event

    model = CapturingModel()
    agent = await Agent.create(model, config=AgentRunConfig("legacy"), extensions=[sqlite_extension])

    await agent.run("Try again")

    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls for message in model.requests[0].messages
    )
    records = await sqlite_extension.list_raw_messages("legacy")
    assert [record.message for record in records][:3] == [
        UserMessage(content="Original request"),
        AssistantMessage(tool_calls=(ToolCall("missing-result", "add", {"left": 1, "right": 2}),)),
        UserMessage(content="Previously failed retry"),
    ]


async def test_compaction_writes_snapshot_without_rewriting_raw_log(sqlite_extension):
    storage = sqlite_extension.storage
    original = UserMessage(content="Old context " * 200)
    await storage.append("session", "old-request", original)
    await storage.append("session", "old-request", AssistantMessage(content="Old answer"))
    agent = await Agent.create(
        AnswerModel("Current answer"),
        config=AgentRunConfig("session"),
        extensions=[
            sqlite_extension,
            CompactionExtension(AnswerModel("Compact checkpoint"), max_tokens=50, keep_recent_tokens=1),
        ],
    )

    await agent.run("Current question")

    with sqlite_file(storage.path) as connection:
        raw_rows = connection.execute("select content from raw_messages order by sequence").fetchall()
        snapshots = connection.execute(
            "select compacted_through_sequence from session_snapshots order by version"
        ).fetchall()
        assert [row["content"] for row in raw_rows] == [
            original.content,
            "Old answer",
            "Current question",
            "Current answer",
        ]
        assert len(snapshots) == 1
        assert snapshots[0]["compacted_through_sequence"] == 2


async def test_delete_session_explicitly_cleans_all_sqlite_rows(sqlite_extension):
    storage = sqlite_extension.storage
    await storage.append("session", "request", UserMessage(content="Delete me"))
    await storage.snapshot("session", CompactedMessage(content="Checkpoint"), 1, 0)

    assert await sqlite_extension.delete_session("session") is True

    with sqlite_file(storage.path) as connection:
        assert connection.execute("select count(*) from agent_sessions").fetchone()[0] == 0
        assert connection.execute("select count(*) from raw_messages").fetchone()[0] == 0
        assert connection.execute("select count(*) from session_snapshots").fetchone()[0] == 0


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
    original = await storage.list_raw_messages("session")

    updated = await sqlite_extension.update_session(
        "session",
        title="  Renamed session  ",
        agent_name="  review-agent  ",
    )

    assert updated is not None
    assert updated.title == "Renamed session"
    assert updated.agent_name == "review-agent"
    assert updated.message_count == 1
    assert await sqlite_extension.get_session("session") == updated
    assert (await storage.load("session")).title == "Renamed session"
    assert await storage.list_raw_messages("session") == original
    assert await sqlite_extension.get_session("missing") is None
    assert await sqlite_extension.update_session("missing", title="Valid title") is None


@pytest.mark.parametrize("title", ["", "   ", "x" * 201])
async def test_update_session_rejects_invalid_titles(sqlite_extension, title):
    with pytest.raises(ValueError, match="Session title"):
        await sqlite_extension.update_session("session", title=title)


@pytest.mark.parametrize("agent_name", ["", "   ", "x" * 65])
async def test_update_session_rejects_invalid_agent_names(sqlite_extension, agent_name):
    with pytest.raises(ValueError, match="Agent name"):
        await sqlite_extension.update_session("session", agent_name=agent_name)


async def test_update_session_requires_at_least_one_field(sqlite_extension):
    with pytest.raises(ValueError, match="At least one"):
        await sqlite_extension.update_session("session")
