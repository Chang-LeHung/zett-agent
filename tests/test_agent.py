import asyncio

import pytest

from kcs_agent import (
    Agent,
    AgentEventType,
    AgentIterationLimitError,
    AgentProtocolError,
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ReasoningEffort,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool,
)


@tool(guidelines="Use for exact addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


class ScriptedModel:
    def __init__(self, *messages: AssistantMessage):
        self.messages = list(messages)
        self.requests: list[ModelRequest] = []

    async def stream(self, request):
        self.requests.append(request)
        message = self.messages.pop(0)
        if message.content:
            yield ModelEvent.reasoning("Checking.")
            yield ModelEvent.text(message.content)
        yield ModelEvent.completed(ModelResponse(message))


async def test_model_tool_model_loop_preserves_order_and_usage_response():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 2, "right": 3}),)),
        AssistantMessage(content="5"),
    )
    events = [event async for event in Agent(model, tools=[add]).stream(UserMessage(content="Add 2 and 3"))]
    assert [event.type for event in events] == [
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.REASONING_DELTA,
        AgentEventType.TEXT_DELTA,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.RUN_COMPLETED,
    ]
    assert events[-1].message.content == "5"
    assert isinstance(model.requests[1].messages[-1], ToolMessage)
    assert model.requests[1].messages[-1].content == "5"
    assert model.requests[1].messages[-2].tool_calls[0].id == "c1"


async def test_run_returns_a_plain_assistant_message():
    model = ScriptedModel(AssistantMessage(content="Hello"))
    reply = await Agent(model).run(UserMessage(content="Hi"), reasoning_effort=ReasoningEffort.LOW)
    assert reply.content == "Hello"
    assert model.requests[0].reasoning_effort == ReasoningEffort.LOW


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("c1", "missing", {}),
        ToolCall("c1", "add", {"left": "invalid", "right": 1}),
    ],
)
async def test_tool_error_returns_to_model(call):
    model = ScriptedModel(AssistantMessage(tool_calls=(call,)), AssistantMessage(content="Please clarify."))
    events = [event async for event in Agent(model, tools=[add]).stream(UserMessage(content="Do it"))]
    assert any(event.type == AgentEventType.TOOL_FAILED for event in events)
    assert model.requests[1].messages[-1].success is False
    assert events[-1].message.content == "Please clarify."


async def test_history_is_explicit_and_not_mutated_or_retained_between_runs():
    history = [UserMessage(content="Old question"), AssistantMessage(content="Old reply")]
    model = ScriptedModel(AssistantMessage(content="New reply"), AssistantMessage(content="Other reply"))
    agent = Agent(model, system_prompt="")
    await agent.run(UserMessage(content="Latest"), history=history)
    await agent.run(UserMessage(content="Other session"))
    assert [item.content for item in model.requests[0].messages] == ["Old question", "Old reply", "Latest"]
    assert [item.content for item in model.requests[1].messages] == ["Other session"]
    assert len(history) == 2


async def test_tool_guidance_follows_all_instructions_without_duplication():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 1, "right": 2}),)),
        AssistantMessage(content="3"),
    )
    await Agent(model, system_prompt="Base", tools=[add]).run(
        UserMessage(content="Add"),
        history=[SystemMessage(content="Snapshot")],
    )
    for request in model.requests:
        prompt = request.messages[0].content
        assert prompt.startswith("Base\n\nSnapshot")
        assert prompt.endswith("- add: Use for exact addition.")
        assert prompt.count("# Tool guidelines") == 1


async def test_iteration_limit_is_explicit():
    model = ScriptedModel(AssistantMessage(tool_calls=(ToolCall("c", "add", {"left": 1, "right": 2}),)))
    with pytest.raises(AgentIterationLimitError):
        await Agent(model, tools=[add], max_iterations=1).run(UserMessage(content="Add"))


@pytest.mark.parametrize("extra_response", [False, True])
async def test_invalid_model_stream_fails(extra_response):
    class InvalidModel:
        async def stream(self, request):
            if extra_response:
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))
            yield ModelEvent.text("unexpected")

    with pytest.raises(AgentProtocolError):
        await Agent(InvalidModel()).run(UserMessage(content="Hi"))


async def test_task_cancellation_closes_provider_stream():
    waiting, closed = asyncio.Event(), asyncio.Event()

    class SlowModel:
        async def stream(self, request):
            try:
                yield ModelEvent.text("partial")
                waiting.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

    task = asyncio.create_task(Agent(SlowModel()).run(UserMessage(content="Hi")))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_duplicate_tools_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        Agent(ScriptedModel(), tools=[add, add])
