"""Custom extension events are streamed without changing request state."""

import pytest

from kcs_agent import (
    Agent,
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentPhase,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
)


@pytest.mark.parametrize("data", [None, {}, {"name": "progress", "items": [1, 2], "done": False}])
@pytest.mark.parametrize("compacting", [False, True])
async def test_custom_events_preserve_payload_order_and_phase(data, compacting):
    custom = AgentEvent(AgentEventType.CUSTOM, "session", data=data)

    class Extension(AgentExtension):
        async def before_model_events(self, context):
            if compacting:
                yield AgentEvent(AgentEventType.COMPACTION_STARTED, "session")
            yield custom
            if compacting:
                yield AgentEvent(AgentEventType.COMPACTION_COMPLETED, "session", applied=False)

    class Model:
        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    agent = await Agent.create(Model(), config=AgentConfig("session"), extensions=[Extension()])
    events = [event async for event in agent.stream("hello")]
    assert events[1 if compacting else 0] is custom
    assert custom.data is data
    assert custom.phase == (AgentPhase.COMPACTING if compacting else AgentPhase.READY)
    assert events[-1].type == AgentEventType.RUN_COMPLETED
    assert agent.state.phase == AgentPhase.COMPLETED


def test_builtin_events_have_no_custom_payload_by_default():
    assert AgentEvent(AgentEventType.MODEL_STARTED, "session").data is None
