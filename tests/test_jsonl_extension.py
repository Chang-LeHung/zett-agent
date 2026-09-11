import json

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentEventType,
    AssistantMessage,
    JSONLExtension,
    ModelEvent,
    ModelResponse,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool,
)
from zett_agent.storage import decode_messages


@tool(guidelines="Use once to produce the requested value.")
def lookup() -> dict[str, str]:
    """Return one deterministic tool result."""
    return {"value": "found"}


class ToolModel:
    async def stream(self, request):
        if isinstance(request.messages[-1], ToolMessage):
            message = AssistantMessage(content="Final answer")
        else:
            message = AssistantMessage(tool_calls=(ToolCall("call-1", "lookup", {}),))
        yield ModelEvent.completed(ModelResponse(message))


def records(extension: JSONLExtension, session_id: str) -> list[dict]:
    return [json.loads(line) for line in extension.session_path(session_id).read_text().splitlines()]


async def test_jsonl_writes_one_line_per_turn_without_repeating_history(tmp_path):
    extension = JSONLExtension(tmp_path / "turns")
    config = AgentConfig("session", request_id="request-1")
    agent = await Agent.create(ToolModel(), tools=[lookup], extensions=[extension], config=config)

    await agent.run(
        UserMessage(content="First", attributes={"client_id": "message-1"}),
        config=config,
        metadata={"source": "editor", "attempt": 1},
        tags={"intent": "lookup"},
    )
    await agent.run("Second", config=AgentConfig("session", request_id="request-2"))

    saved = records(extension, "session")
    assert len(saved) == 2
    assert saved[0]["metadata"] == {"source": "editor", "attempt": 1}
    assert saved[0]["tags"] == {"intent": "lookup"}
    assert saved[0]["status"] == "completed"
    first_messages = decode_messages(json.dumps(saved[0]["messages"]))
    assert [message.role.value for message in first_messages] == ["user", "assistant", "tool", "assistant"]
    assert first_messages[0].attributes == {"client_id": "message-1"}
    assert [message.content for message in first_messages] == [
        "First",
        "",
        '{"value": "found"}',
        "Final answer",
    ]
    second_messages = decode_messages(json.dumps(saved[1]["messages"]))
    assert second_messages[0] == UserMessage(content="Second")
    assert all(message.content != "First" for message in second_messages)


async def test_jsonl_uses_one_file_per_session_and_safe_file_names(tmp_path):
    extension = JSONLExtension(tmp_path)
    first = await Agent.create(ToolModel(), tools=[lookup], extensions=[extension], config=AgentConfig("../first"))
    second = await Agent.create(ToolModel(), tools=[lookup], extensions=[extension], config=AgentConfig("second"))

    await first.run("One")
    await second.run("Two")

    first_path = extension.session_path("../first")
    second_path = extension.session_path("second")
    assert first_path.parent == tmp_path
    assert first_path != second_path
    assert records(extension, "../first")[0]["session_id"] == "../first"
    assert records(extension, "second")[0]["session_id"] == "second"


async def test_jsonl_records_failed_turn_once_without_serializing_exception(tmp_path):
    class FailingModel:
        async def stream(self, request):
            raise RuntimeError("provider unavailable")
            yield

    extension = JSONLExtension(tmp_path)
    agent = await Agent.create(FailingModel(), extensions=[extension], config=AgentConfig("failed"))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.run("Try")

    saved = records(extension, "failed")
    assert len(saved) == 1
    assert saved[0]["status"] == "failed"
    assert saved[0]["error"] == {"type": "RuntimeError", "message": "provider unavailable"}
    assert decode_messages(json.dumps(saved[0]["messages"])) == [UserMessage(content="Try")]


async def test_jsonl_records_cancelled_turn_once(tmp_path):
    extension = JSONLExtension(tmp_path)
    agent = await Agent.create(ToolModel(), tools=[lookup], extensions=[extension], config=AgentConfig("cancelled"))
    stream = agent.stream("Stop")
    async for event in stream:
        if event.type is AgentEventType.MODEL_STARTED:
            break
    await stream.aclose()

    saved = records(extension, "cancelled")
    assert len(saved) == 1
    assert saved[0]["status"] == "cancelled"
    assert decode_messages(json.dumps(saved[0]["messages"])) == [UserMessage(content="Stop")]
