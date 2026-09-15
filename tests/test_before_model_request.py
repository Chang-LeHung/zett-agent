"""Before-model hooks observe fresh request snapshots without stale context."""

from dataclasses import FrozenInstanceError

import pytest

from zett_agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentRunConfig,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    ReasoningEffort,
    SystemMessage,
    tool,
)


@tool
def probe() -> str:
    """Return a fixed observation.

    Snippet:
        probe()

    Guidelines:
        - Use to inspect the test environment.
    """
    return "observed"


@pytest.mark.parametrize("effort", [ReasoningEffort.LOW, ReasoningEffort.HIGH])
async def test_each_hook_and_provider_receive_latest_messages_and_tools(effort):
    seen = []

    class First(AgentExtension):
        priority = 10

        async def before_model(self, context, request):
            seen.append(request)
            assert request.reasoning_effort == effort
            assert not request.tools
            with pytest.raises(FrozenInstanceError):
                request.reasoning_effort = ReasoningEffort.LOW
            context.state.messages.append(SystemMessage(content="before-model change"))
            context.register_tool(probe)

    class Second(AgentExtension):
        priority = 20

        async def before_model(self, context, request):
            seen.append(request)
            assert request.messages[-1].content == "before-model change"
            assert [item.name for item in request.tools] == ["probe"]
            await context.emit(AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name="preparing"))
            context.state.messages[:] = [
                SystemMessage(content="replacement context"),
                SystemMessage(content="last hook change"),
            ]
            context.tools.clear()

    class Model:
        async def stream(self, request):
            seen.append(request)
            assert [item.content for item in request.messages] == ["replacement context", "last hook change"]
            assert request.reasoning_effort == effort
            assert not request.tools
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    agent = await Agent.create(Model(), config=AgentRunConfig("fresh"), extensions=[Second(), First()])
    events = [event async for event in agent.stream("hello", reasoning_effort=effort)]
    assert events[-1].message.content == "done"
    assert len(seen) == 3
    assert len({id(request) for request in seen}) == 3
    assert seen[0].messages[-1].content == "hello"
    assert not seen[0].tools


@pytest.mark.parametrize("streaming", [False, True])
async def test_failed_preprocessing_never_calls_provider(streaming):
    class Reject(AgentExtension):
        async def before_model(self, context, request):
            assert request.messages[-1].content == "hello"
            if streaming:
                await context.emit(AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name="checking"))
            raise ValueError("rejected request")

    class Model:
        async def stream(self, request):
            pytest.fail("Provider must not run after preprocessing fails")
            if False:
                yield

    agent = await Agent.create(Model(), config=AgentRunConfig("failure"), extensions=[Reject()])
    with pytest.raises(ValueError, match="rejected request"):
        await agent.run("hello")
