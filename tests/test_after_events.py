"""Post-operation event hooks preserve ordering, validation, and cleanup."""

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
    ToolMessage,
    tool,
)


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
async def test_post_hooks_follow_completion_in_priority_order(failed):
    class Observer(AgentExtension):
        def __init__(self, name, priority):
            self.name, self.priority = name, priority

        async def after_model_events(self, context, response):
            assert context.state.messages[-1] is response.message
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name=f"model-{self.name}")

        async def after_tool_events(self, context, call, result, error):
            assert result.success is not failed
            assert (error is not None) is failed
            assert context.state.phase is AgentPhase.READY
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name=f"tool-{self.name}")

    agent = await Agent.create(
        Model(),
        config=AgentConfig("s"),
        tools=[] if failed else [work],
        extensions=[Observer("late", 200), Observer("early", 10)],
    )
    events = [event async for event in agent.stream("go")]
    for i, event in enumerate(events):
        if event.type is AgentEventType.MODEL_COMPLETED:
            assert [e.name for e in events[i + 1 : i + 3]] == ["model-early", "model-late"]
        if event.type in (AgentEventType.TOOL_COMPLETED, AgentEventType.TOOL_FAILED):
            assert [e.name for e in events[i + 1 : i + 3]] == ["tool-early", "tool-late"]


@pytest.mark.parametrize("hook", ["after_model_events", "after_tool_events"])
@pytest.mark.parametrize("mode", ["close", "error", "wrong_type", "wrong_session"])
async def test_post_hook_failures_and_close_release_resources(hook, mode):
    finalized = []

    class Observer(AgentExtension):
        enabled = True

    async def produce(self, context, *args):
        if not self.enabled:
            return
        self.enabled = False
        try:
            if mode == "error":
                raise RuntimeError("hook failed")
            yield AgentEvent(
                AgentEventType.MODEL_COMPLETED if mode == "wrong_type" else AgentEventType.CUSTOM,
                "other" if mode == "wrong_session" else context.config.session_id,
                name="post",
            )
        finally:
            finalized.append(True)

    setattr(Observer, hook, produce)
    agent = await Agent.create(Model(), config=AgentConfig("s"), tools=[work], extensions=[Observer()])
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
