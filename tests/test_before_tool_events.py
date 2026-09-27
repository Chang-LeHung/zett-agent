"""Pre-tool queue emission ordering, validation, and cancellation cleanup."""

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
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)
from zett_agent.tools.base import tool


class ToolModel:
    def __init__(self):
        self.steps = 0

    async def stream(self, request):
        self.steps += 1
        message = (
            AssistantMessage(tool_calls=(ToolCall("a", "record"), ToolCall("b", "record")))
            if self.steps == 1
            else AssistantMessage(content="done")
        )
        yield ModelEvent.completed(ModelResponse(message))


def recording_tool(executed):
    @tool(guidelines="Record one invocation.")
    def record() -> str:
        """Record a tool invocation."""
        executed.append(True)
        return "ok"

    return record


async def test_hooks_run_for_each_call_in_order_before_tool_execution():
    order, executed = [], []

    class Extension(AgentExtension):
        def __init__(self, name):
            self.name = name

        async def before_tool(self, context, call):
            order.append((call.id, self.name, "hook"))
            assert context.state.phase == AgentPhase.READY
            order.append((call.id, self.name, "events"))
            await context.emit(
                AgentEvent(
                    AgentEventType.CUSTOM,
                    context.config.session_id,
                    name=self.name,
                    payload={"call": call.id},
                )
            )

    agent = await Agent.create(
        ToolModel(),
        config=AgentRunConfig("tools"),
        tools=[recording_tool(executed)],
        extensions=[Extension("one"), Extension("two")],
    )
    events = [event async for event in agent.stream("run")]
    assert order == [
        (call, name, kind) for call in ("a", "b") for name in ("one", "two") for kind in ("hook", "events")
    ]
    relevant = [event for event in events if event.type in (AgentEventType.CUSTOM, AgentEventType.TOOL_STARTED)]
    assert [event.type for event in relevant] == [
        AgentEventType.CUSTOM,
        AgentEventType.CUSTOM,
        AgentEventType.CUSTOM,
        AgentEventType.CUSTOM,
        AgentEventType.TOOL_STARTED,
    ]
    assert all(event.phase == AgentPhase.READY for event in relevant if event.type == AgentEventType.CUSTOM)
    tool_events = [
        event for event in events if event.type in (AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED)
    ]
    assert [[call.id for call in event.tool_calls] for event in tool_events] == [["a", "b"], ["a"], ["b"]]
    assert executed == [True, True]


def test_agent_event_tool_call_lists_are_not_shared_between_events():
    first = AgentEvent(AgentEventType.TOOL_STARTED, "session")
    second = AgentEvent(AgentEventType.TOOL_STARTED, "session")

    first.tool_calls.append(ToolCall("call-1", "record"))

    assert [call.id for call in first.tool_calls] == ["call-1"]
    assert second.tool_calls == []


@pytest.mark.parametrize("mode", ["close", "invalid", "error"])
async def test_pre_tool_interruption_closes_hook_without_executing_tool(mode):
    executed, closed = [], []
    blocker = asyncio.Event()

    class Extension(AgentExtension):
        async def before_tool(self, context, call):
            try:
                if mode == "error":
                    raise ValueError("preparation failed")
                await context.emit(
                    AgentEvent(
                        AgentEventType.MODEL_STARTED if mode == "invalid" else AgentEventType.CUSTOM,
                        context.config.session_id,
                        name=None if mode == "invalid" else "preparation",
                    )
                )
                if mode == "close":
                    await blocker.wait()
            finally:
                closed.append(True)

    agent = await Agent.create(
        ToolModel(), config=AgentRunConfig("tools"), tools=[recording_tool(executed)], extensions=[Extension()]
    )
    stream = agent.stream("run")
    if mode == "close":
        async for event in stream:
            if event.type == AgentEventType.CUSTOM:
                break
        await stream.aclose()
        assert agent.state.phase == AgentPhase.CANCELLED
    else:
        with pytest.raises(AgentProtocolError if mode == "invalid" else ValueError):
            _ = [event async for event in stream]
        assert agent.state.phase == AgentPhase.FAILED
    assert closed == [True]
    assert executed == []
