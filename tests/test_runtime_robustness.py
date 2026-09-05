from datetime import UTC, datetime

import pytest

from kcs_agent import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentPhase,
    AgentPhaseTransitionMixin,
    AgentProtocolError,
    AgentState,
    AssistantMessage,
    ContentCompletedEvent,
    ContentStartedEvent,
    MessageTiming,
    ModelEvent,
    ModelEventType,
    ModelOutputTracker,
    ModelResponse,
    ReasoningCompletedEvent,
    ReasoningStartedEvent,
    ToolCall,
    ToolCallDelta,
)

CONFIG = AgentConfig("robustness-session")


class EventModel:
    """Emit an exact event sequence for protocol-error tests."""

    def __init__(self, *events: ModelEvent) -> None:
        self.events = events

    async def stream(self, request):
        for event in self.events:
            yield event


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (ModelEvent(ModelEventType.TOOL_CALL_DELTA), "Missing tool-call delta"),
        (ModelEvent(ModelEventType.RESPONSE), "Missing model response"),
    ],
)
async def test_agent_rejects_incomplete_terminal_model_events(event, message):
    agent = await Agent.create(EventModel(event), config=CONFIG, extensions=[])

    with pytest.raises(AgentProtocolError, match=message):
        await agent.run("Trigger malformed event")

    assert agent.state.phase == AgentPhase.FAILED


async def test_agent_rejects_extension_events_outside_the_pre_model_protocol():
    class InvalidExtension(AgentExtension):
        async def before_model_events(self, context):
            yield AgentEvent(AgentEventType.MODEL_STARTED, context.config.session_id)

    agent = await Agent.create(
        EventModel(ModelEvent.completed(ModelResponse(AssistantMessage(content="unused")))),
        config=CONFIG,
        extensions=[InvalidExtension()],
    )

    with pytest.raises(AgentProtocolError, match="unsupported pre-model event"):
        await agent.run("Trigger invalid extension event")

    assert agent.state.phase == AgentPhase.FAILED


async def test_phase_guards_reject_wrong_phase_and_do_not_recancel_completion():
    machine = AgentPhaseTransitionMixin()
    context = AgentContext(CONFIG, AgentState(), {}, ())

    with pytest.raises(AgentProtocolError, match="must be 'generating'"):
        machine._require_phase(context.state, AgentPhase.GENERATING)

    await machine._start_context_loading(context)
    await machine._finish_context_loading(context)
    await machine._complete_request(context)
    await machine._cancel_request(context)
    assert context.state.phase == AgentPhase.COMPLETED


async def test_output_tracker_ignores_empty_deltas_and_closes_reasoning_for_a_tool_call():
    published = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            published.append(event)

    context = AgentContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, (Observer(),))
    tracker = ModelOutputTracker()
    await tracker.observe(context, ModelEvent.reasoning(""))
    await tracker.observe(context, ModelEvent.text(""))
    assert published == []

    await tracker.observe(context, ModelEvent.reasoning("first"))
    await tracker.observe(context, ModelEvent.reasoning("second"))
    await tracker.observe(
        context,
        ModelEvent.tool_call(ToolCallDelta(index=0, id_delta="call-1", name_delta="read")),
    )
    await tracker.observe(
        context, ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(ToolCall("call-1", "read"),))))
    )

    assert [type(event) for event in published] == [ReasoningStartedEvent, ReasoningCompletedEvent]
    assert tracker.content_started is None
    assert tracker.content_completed is None


async def test_output_tracker_publishes_each_content_boundary_only_once():
    published = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            published.append(event)

    context = AgentContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, (Observer(),))
    tracker = ModelOutputTracker()
    await tracker.observe(context, ModelEvent.text("one"))
    await tracker.observe(context, ModelEvent.text("two"))
    response = ModelEvent.completed(ModelResponse(AssistantMessage(content="onetwo")))
    await tracker.observe(context, response)
    await tracker.observe(context, response)

    assert [type(event) for event in published] == [ContentStartedEvent, ContentCompletedEvent]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"duration_ns": -1}, "duration_ns cannot be negative"),
        ({"reasoning_started_at": datetime.now(UTC)}, "reasoning timing fields must be supplied together"),
        (
            {
                "content_started_at": datetime.now(UTC),
                "content_completed_at": datetime.now(UTC),
                "content_duration_ns": -1,
            },
            "content_duration_ns cannot be negative",
        ),
    ],
)
def test_message_timing_rejects_incomplete_or_negative_measurements(changes, message):
    now = datetime.now(UTC)
    values = {"started_at": now, "completed_at": now, "duration_ns": 0, **changes}
    with pytest.raises(ValueError, match=message):
        MessageTiming(**values)
