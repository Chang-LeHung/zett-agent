"""Verify that turn hooks bracket every model/tool cycle inside one request."""

import pytest

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.events import SteeringMessageEvent
from zett_agent.messages import (
    AssistantMessage,
    ToolCall,
    UserMessage,
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)
from zett_agent.tools.base import tool


class ScriptedModel:
    def __init__(self, *messages: AssistantMessage) -> None:
        self.messages = list(messages)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(self.messages.pop(0)))


class Recorder(AgentExtension):
    """Record lifecycle order and the assistant message reported per turn."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.turns: list[AssistantMessage] = []
        self.errors: list[Exception] = []

    async def before_run(self, context):
        self.order.append("before_run")

    async def before_turn(self, context):
        self.order.append("before_turn")

    async def before_model(self, context, request):
        self.order.append("before_model")

    async def after_model(self, context, response):
        self.order.append("after_model")

    async def before_tool(self, context, call):
        self.order.append("before_tool")

    async def after_tool(self, context, call, result, error):
        self.order.append("after_tool")

    async def after_turn(self, context, result):
        self.order.append("after_turn")
        self.turns.append(result)

    async def after_run(self, context, result):
        self.order.append("after_run")

    async def on_success(self, context, result):
        self.order.append("on_success")

    async def on_error(self, context, error):
        self.errors.append(error)


@tool(guidelines="Echo one required argument.")
async def echo(text: str) -> str:
    """Return the supplied text unchanged.

    Args:
        text: Text to return.
    """
    return text


async def test_turn_hooks_bracket_each_model_and_tool_cycle():
    recorder = Recorder()
    call = ToolCall("call-1", "echo", {"text": "hi"})
    model = ScriptedModel(AssistantMessage(tool_calls=(call,)), AssistantMessage(content="done"))
    agent = await Agent.create(
        model,
        config=AgentRunConfig("turn-order"),
        extensions=[recorder],
        tools=[echo],
    )

    result = await agent.run("start")

    assert result.content == "done"
    assert recorder.order == [
        "before_run",
        "before_turn",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "after_turn",
        "before_turn",
        "before_model",
        "after_model",
        "after_turn",
        "after_run",
        "on_success",
    ]
    tool_turn, answer_turn = recorder.turns
    assert [requested.name for requested in tool_turn.tool_calls] == ["echo"]
    assert answer_turn.content == "done"


async def test_failed_tool_call_still_ends_its_turn():
    @tool(guidelines="Fail with a normal tool error.")
    async def explode() -> str:
        """Raise a tool failure."""
        raise RuntimeError("tool failed")

    recorder = Recorder()
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("call-1", "explode"),)),
        AssistantMessage(content="recovered"),
    )
    agent = await Agent.create(
        model,
        config=AgentRunConfig("turn-tool-failure"),
        extensions=[recorder],
        tools=[explode],
    )

    result = await agent.run("start")

    assert result.content == "recovered"
    assert recorder.order.count("after_turn") == 2
    assert "on_success" in recorder.order


async def test_steering_continues_one_request_with_a_second_turn():
    class Steer(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "first":
                await context.publish(SteeringMessageEvent(UserMessage(content="follow up")))

    recorder = Recorder()
    model = ScriptedModel(AssistantMessage(content="first"), AssistantMessage(content="second"))
    agent = await Agent.create(
        model,
        config=AgentRunConfig("turn-steering"),
        extensions=[Steer(), recorder],
    )

    result = await agent.run("original")

    assert result.content == "second"
    assert [message.content for message in recorder.turns] == ["first", "second"]
    # One request still owns both turns: one before_run wraps two turn pairs.
    assert recorder.order == [
        "before_run",
        "before_turn",
        "before_model",
        "after_model",
        "after_turn",
        "before_turn",
        "before_model",
        "after_model",
        "after_turn",
        "after_run",
        "on_success",
    ]


async def test_steering_that_skips_a_tool_batch_still_closes_its_turn():
    class Steer(AgentExtension):
        async def after_model(self, context, response):
            if response.message.tool_calls:
                await context.publish(SteeringMessageEvent(UserMessage(content="instead")))

    recorder = Recorder()
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("call-1", "echo", {"text": "hi"}),)),
        AssistantMessage(content="redirected"),
    )
    agent = await Agent.create(
        model,
        config=AgentRunConfig("turn-skipped-tools"),
        extensions=[Steer(), recorder],
        tools=[echo],
    )

    result = await agent.run("start")

    assert result.content == "redirected"
    # The skipped batch never ran, so no tool hooks fire, but the turn still ends
    # once before the steering turn begins.
    assert recorder.order == [
        "before_run",
        "before_turn",
        "before_model",
        "after_model",
        "after_turn",
        "before_turn",
        "before_model",
        "after_model",
        "after_turn",
        "after_run",
        "on_success",
    ]
    skipped = [message for message in agent.state.messages if message.role.value == "tool"]
    assert len(skipped) == 1
    assert skipped[0].success is False


async def test_failed_request_reports_no_turn_completion():
    class FailingModel:
        async def stream(self, request):
            yield ModelEvent.text("partial")
            raise ValueError("model failed")

    recorder = Recorder()
    agent = await Agent.create(FailingModel(), config=AgentRunConfig("turn-failure"), extensions=[recorder])

    with pytest.raises(ValueError, match="model failed"):
        await agent.run("start")

    assert recorder.order == ["before_run", "before_turn", "before_model"]
    assert [str(error) for error in recorder.errors] == ["model failed"]
