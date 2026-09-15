from datetime import UTC, datetime

import pytest

from zett_agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentPhase,
    AgentPhaseTransitionMixin,
    AgentProtocolError,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AssistantMessage,
    ContentCompletedEvent,
    ContentStartedEvent,
    MessageTiming,
    ModelEvent,
    ModelEventType,
    ModelOutputTracker,
    ModelResponse,
    PhaseTransitionEvent,
    ReasoningCompletedEvent,
    ReasoningStartedEvent,
    RunCancelledEvent,
    ToolCall,
    ToolCallDelta,
)

CONFIG = AgentRunConfig("robustness-session")


@pytest.mark.parametrize(
    "phase",
    [
        AgentPhase.LOADING_CONTEXT,
        AgentPhase.READY,
        AgentPhase.COMPACTING,
        AgentPhase.GENERATING,
        AgentPhase.RUNNING_TOOL,
    ],
)
async def test_cancellation_publishes_a_dedicated_event_once(phase):
    received = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            assert context.state.phase == AgentPhase.CANCELLED
            received.append(event)

    context = AgentRunContext(CONFIG, AgentState(phase=phase), {}, (Observer(),))
    machine = AgentPhaseTransitionMixin()
    await machine._cancel_request(context)
    await machine._cancel_request(context)
    assert [type(event) for event in received] == [PhaseTransitionEvent, RunCancelledEvent]
    transition, cancelled = received
    assert cancelled.previous_phase == phase
    assert cancelled.occurred_at == transition.occurred_at
    assert cancelled.monotonic_ns == transition.monotonic_ns


async def test_closing_agent_stream_publishes_cancellation():
    received = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, RunCancelledEvent):
                received.append(event)

    agent = await Agent.create(EventModel(), config=CONFIG, extensions=[Observer()])
    stream = agent.stream("Hello")
    assert (await anext(stream)).type == AgentEventType.MODEL_STARTED
    await stream.aclose()
    assert agent.state.phase == AgentPhase.CANCELLED
    assert len(received) == 1
    assert received[0].previous_phase == AgentPhase.GENERATING


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
        async def before_model(self, context, request):
            await context.emit(AgentEvent(AgentEventType.CUSTOM, "another-session", name="invalid"))

    agent = await Agent.create(
        EventModel(ModelEvent.completed(ModelResponse(AssistantMessage(content="unused")))),
        config=CONFIG,
        extensions=[InvalidExtension()],
    )

    with pytest.raises(AgentProtocolError, match="another session"):
        await agent.run("Trigger invalid extension event")

    assert agent.state.phase == AgentPhase.FAILED


async def test_phase_guards_reject_wrong_phase_and_do_not_recancel_completion():
    machine = AgentPhaseTransitionMixin()
    context = AgentRunContext(CONFIG, AgentState(), {}, ())

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

    context = AgentRunContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, (Observer(),))
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

    context = AgentRunContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, (Observer(),))
    tracker = ModelOutputTracker()
    await tracker.observe(context, ModelEvent.text("one"))
    await tracker.observe(context, ModelEvent.text("two"))
    response = ModelEvent.completed(ModelResponse(AssistantMessage(content="onetwo")))
    await tracker.observe(context, response)
    with pytest.raises(AgentProtocolError, match="after its final response"):
        await tracker.observe(context, response)

    assert [type(event) for event in published] == [ContentStartedEvent, ContentCompletedEvent]


@pytest.mark.parametrize("segment", ["reasoning", "content"])
async def test_output_completion_requires_an_open_segment(segment):
    context = AgentRunContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, ())
    tracker = ModelOutputTracker()
    complete = getattr(tracker, f"_complete_{segment}")
    with pytest.raises(AgentProtocolError, match="requires an open"):
        await complete(context)
    delta = ModelEvent.reasoning("thinking") if segment == "reasoning" else ModelEvent.text("answer")
    await tracker.observe(context, delta)
    await complete(context)
    with pytest.raises(AgentProtocolError, match="requires an open"):
        await complete(context)


@pytest.mark.parametrize("prefix", ["content", "tool", "reasoning-content", "reasoning-tool"])
async def test_reasoning_cannot_resume_after_other_output(prefix):
    context = AgentRunContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, ())
    tracker = ModelOutputTracker()
    if prefix.startswith("reasoning-"):
        await tracker.observe(context, ModelEvent.reasoning("thinking"))
    delta = (
        ModelEvent.text("answer")
        if prefix.endswith("content")
        else ModelEvent.tool_call(ToolCallDelta(index=0, id_delta="call-1", name_delta="read"))
    )
    await tracker.observe(context, delta)
    with pytest.raises(AgentProtocolError, match="cannot start or resume"):
        await tracker.observe(context, ModelEvent.reasoning("late thinking"))


@pytest.mark.parametrize("phase", [phase for phase in AgentPhase if phase != AgentPhase.GENERATING])
async def test_output_tracker_rejects_non_generating_phase(phase):
    context = AgentRunContext(CONFIG, AgentState(phase=phase), {}, ())
    tracker = ModelOutputTracker()
    with pytest.raises(AgentProtocolError, match="must be 'generating'"):
        await tracker.observe(context, ModelEvent.text("answer"))
    assert tracker.content_started is None


@pytest.mark.parametrize("kind", [ModelEventType.RESPONSE, ModelEventType.TOOL_CALL_DELTA])
async def test_malformed_events_do_not_complete_reasoning(kind):
    context = AgentRunContext(CONFIG, AgentState(phase=AgentPhase.GENERATING), {}, ())
    tracker = ModelOutputTracker()
    await tracker.observe(context, ModelEvent.reasoning("thinking"))
    with pytest.raises(AgentProtocolError, match="Missing"):
        await tracker.observe(context, ModelEvent(kind))
    assert tracker.reasoning_completed is None


@pytest.mark.parametrize("output", ["empty", "reasoning", "content", "both"])
async def test_stream_emits_paired_output_boundaries_in_order(output):
    class Model:
        async def stream(self, request):
            yield ModelEvent.reasoning("")
            yield ModelEvent.text("")
            if output in ("reasoning", "both"):
                yield ModelEvent.reasoning("think")
                yield ModelEvent.reasoning(" more")
            if output in ("content", "both"):
                yield ModelEvent.text("answer")
                yield ModelEvent.text(" more")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="final")))

    agent = await Agent.create(Model(), config=CONFIG)
    events = [event async for event in agent.stream("hello")]
    expected = [AgentEventType.MODEL_STARTED, AgentEventType.REASONING_DELTA, AgentEventType.TEXT_DELTA]
    if output in ("reasoning", "both"):
        expected += [
            AgentEventType.REASONING_STARTED,
            AgentEventType.REASONING_DELTA,
            AgentEventType.REASONING_DELTA,
            AgentEventType.REASONING_COMPLETED,
        ]
    if output in ("content", "both"):
        expected += [
            AgentEventType.CONTENT_STARTED,
            AgentEventType.TEXT_DELTA,
            AgentEventType.TEXT_DELTA,
            AgentEventType.CONTENT_COMPLETED,
        ]
    expected += [AgentEventType.MODEL_COMPLETED, AgentEventType.RUN_COMPLETED]
    assert [event.type for event in events] == expected


@pytest.mark.parametrize("reasoning", [True, False])
@pytest.mark.parametrize("cancel", [True, False])
async def test_interrupted_stream_does_not_emit_successful_output_completion(reasoning, cancel):
    from contextlib import aclosing

    class Model:
        async def stream(self, request):
            yield ModelEvent.reasoning("think") if reasoning else ModelEvent.text("answer")
            raise ValueError("stream disconnected")

    agent = await Agent.create(Model(), config=CONFIG)
    events = []
    async with aclosing(agent.stream("hello")) as stream:
        if cancel:
            async for event in stream:
                events.append(event)
                if event.type in (AgentEventType.REASONING_DELTA, AgentEventType.TEXT_DELTA):
                    break
        else:
            with pytest.raises(ValueError, match="disconnected"):
                async for event in stream:
                    events.append(event)
    assert not any(
        event.type in (AgentEventType.CONTENT_COMPLETED, AgentEventType.REASONING_COMPLETED) for event in events
    )


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
