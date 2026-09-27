"""Ordinary lifecycle hooks emit through the request-owned event queue."""

import asyncio

import pytest

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.events import (
    AgentEvent,
    AgentEventType,
    AgentPhase,
)
from zett_agent.exceptions import AgentProtocolError
from zett_agent.extensions.base import AgentExtension
from zett_agent.messages import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)
from zett_agent.tools.base import tool


class Model:
    async def stream(self, request):
        message = (
            AssistantMessage(content="done")
            if isinstance(request.messages[-1], ToolMessage)
            else AssistantMessage(tool_calls=(ToolCall("c", "work", {}),))
        )
        yield ModelEvent.completed(ModelResponse(message))


@tool(guidelines="Return a test result.")
def work() -> str:
    """Return a test result."""
    return "done"


@pytest.mark.parametrize("failed", [False, True])
async def test_lifecycle_hooks_emit_in_priority_order(failed):
    class Observer(AgentExtension):
        def __init__(self, name, priority):
            self.name, self.priority = name, priority

        async def after_model(self, context, response):
            assert context.state.messages[-1] is response.message
            await context.emit(AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name=f"model-{self.name}"))

        async def after_tool(self, context, call, result, error):
            assert result.success is not failed
            assert (error is not None) is failed
            assert context.state.phase is AgentPhase.READY
            await context.emit(AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name=f"tool-{self.name}"))

    agent = await Agent.create(
        Model(),
        config=AgentRunConfig("s"),
        tools=[] if failed else [work],
        extensions=[Observer("late", 200), Observer("early", 10)],
    )
    events = [event async for event in agent.stream("go")]
    for index, event in enumerate(events):
        if event.type is AgentEventType.MODEL_COMPLETED:
            assert [item.name for item in events[index - 2 : index]] == ["model-early", "model-late"]
        if event.type in (AgentEventType.TOOL_COMPLETED, AgentEventType.TOOL_FAILED):
            assert [item.name for item in events[index - 2 : index]] == ["tool-early", "tool-late"]


@pytest.mark.parametrize("hook", ["after_model", "after_tool"])
@pytest.mark.parametrize("mode", ["close", "error", "wrong_session"])
async def test_emitting_hook_failures_and_close_release_resources(hook, mode):
    finalized = []
    blocker = asyncio.Event()

    class Observer(AgentExtension):
        enabled = True

    async def produce(self, context, *args):
        if not self.enabled:
            return
        self.enabled = False
        try:
            if mode == "error":
                raise RuntimeError("hook failed")
            await context.emit(
                AgentEvent(
                    AgentEventType.CUSTOM,
                    "other" if mode == "wrong_session" else context.config.session_id,
                    name="post",
                )
            )
            if mode == "close":
                await blocker.wait()
        finally:
            finalized.append(True)

    setattr(Observer, hook, produce)
    agent = await Agent.create(Model(), config=AgentRunConfig("s"), tools=[work], extensions=[Observer()])
    stream = agent.stream("go")
    if mode == "close":
        async for event in stream:
            if event.name == "post":
                break
        await stream.aclose()
        assert agent.state.phase is AgentPhase.CANCELLED
    else:
        with pytest.raises(RuntimeError if mode == "error" else AgentProtocolError):
            _ = [event async for event in stream]
        assert agent.state.phase is AgentPhase.FAILED
    assert finalized == [True]
    assert agent._active_configs == {}
    assert (await agent.run("retry")).content == "done"
