"""Custom extension events are streamed without changing request state."""

import pytest

from zett_agent import (
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


@pytest.mark.parametrize("payload", [None, {}, {"items": [1, 2], "done": False}])
@pytest.mark.parametrize("compacting", [False, True])
async def test_custom_events_preserve_name_payload_order_and_phase(payload, compacting):
    custom = AgentEvent(AgentEventType.CUSTOM, "session", name="progress", payload=payload)

    class Extension(AgentExtension):
        async def before_model_events(self, context, request):
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
    assert custom.name == "progress"
    assert custom.payload is payload
    assert custom.phase == (AgentPhase.COMPACTING if compacting else AgentPhase.READY)
    assert events[-1].type == AgentEventType.RUN_COMPLETED
    assert agent.state.phase == AgentPhase.COMPLETED


def test_builtin_events_have_no_custom_name_or_payload_by_default():
    event = AgentEvent(AgentEventType.MODEL_STARTED, "session")
    assert event.name is None
    assert event.payload is None


@pytest.mark.parametrize("name", [None, "", "   "])
def test_custom_events_require_a_non_empty_name(name):
    with pytest.raises(ValueError, match="requires a non-empty name"):
        AgentEvent(AgentEventType.CUSTOM, "session", name=name)
