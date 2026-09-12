"""Pre-tool event ordering, protocol validation, and cancellation cleanup."""

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentPhase,
    AgentProtocolError,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    ToolCall,
    tool,
)


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

        async def before_tool_events(self, context, call):
            assert context.state.phase == AgentPhase.READY
            order.append((call.id, self.name, "events"))
            yield AgentEvent(
                AgentEventType.CUSTOM,
                context.config.session_id,
                name=self.name,
                payload={"call": call.id},
            )

    agent = await Agent.create(
        ToolModel(),
        config=AgentConfig("tools"),
        tools=[recording_tool(executed)],
        extensions=[Extension("one"), Extension("two")],
    )
    events = [event async for event in agent.stream("run")]
    assert order == [
        (call, name, kind) for call in ("a", "b") for kind in ("hook", "events") for name in ("one", "two")
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

    class Extension(AgentExtension):
        async def before_tool_events(self, context, call):
            try:
                if mode == "error":
                    raise ValueError("preparation failed")
                yield AgentEvent(
                    AgentEventType.MODEL_STARTED if mode == "invalid" else AgentEventType.CUSTOM,
                    context.config.session_id,
                    name="preparation" if mode != "invalid" else None,
                )
            finally:
                closed.append(True)

    agent = await Agent.create(
        ToolModel(), config=AgentConfig("tools"), tools=[recording_tool(executed)], extensions=[Extension()]
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
