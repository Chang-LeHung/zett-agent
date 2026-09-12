"""Exercise state transitions under overlapping lifecycle and failure conditions."""

import asyncio

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
    ModelEvent,
    ModelResponse,
    PhaseTransitionEvent,
    RunCancelledEvent,
    ToolCall,
    tool,
)


class FinalModel:
    async def stream(self, request):
        yield ModelEvent.text("answer")
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="answer")))


@pytest.mark.parametrize("source", list(AgentPhase))
@pytest.mark.parametrize(
    ("method", "allowed", "target"),
    [
        ("_start_context_loading", {AgentPhase.CREATED}, AgentPhase.LOADING_CONTEXT),
        ("_finish_context_loading", {AgentPhase.LOADING_CONTEXT}, AgentPhase.READY),
        ("_start_compaction", {AgentPhase.READY}, AgentPhase.COMPACTING),
        ("_finish_compaction", {AgentPhase.COMPACTING}, AgentPhase.READY),
        ("_start_model_generation", {AgentPhase.READY}, AgentPhase.GENERATING),
        ("_finish_model_generation", {AgentPhase.GENERATING}, AgentPhase.READY),
        ("_start_tool_execution", {AgentPhase.READY}, AgentPhase.RUNNING_TOOL),
        ("_finish_tool_execution", {AgentPhase.RUNNING_TOOL}, AgentPhase.READY),
        ("_complete_request", {AgentPhase.READY}, AgentPhase.COMPLETED),
        (
            "_fail_request",
            {
                AgentPhase.LOADING_CONTEXT,
                AgentPhase.READY,
                AgentPhase.COMPACTING,
                AgentPhase.GENERATING,
                AgentPhase.RUNNING_TOOL,
            },
            AgentPhase.FAILED,
        ),
    ],
)
async def test_transition_matrix_rejects_without_mutation_or_notification(source, method, allowed, target):
    received = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            received.append(event)

    context = AgentRunContext(AgentRunConfig("matrix"), AgentState(phase=source), {}, (Observer(),))
    operation = getattr(AgentPhaseTransitionMixin(), method)
    if source not in allowed:
        with pytest.raises(AgentProtocolError, match="Invalid agent phase transition"):
            await operation(context)
        assert context.state.phase == source
        assert received == []
    else:
        await operation(context)
        assert context.state.phase == target
        assert len(received) == 1
        assert received[0].previous_phase == source
        assert received[0].current_phase == target


@pytest.mark.parametrize(
    "phase",
    [
        AgentPhase.LOADING_CONTEXT,
        AgentPhase.READY,
        AgentPhase.GENERATING,
    ],
)
async def test_task_cancel_during_transition_subscription_then_reuse_agent(phase):
    entered = asyncio.Event()
    cancelled = []
    errors = []

    class BlockingObserver(AgentExtension):
        block = True

        async def on_event(self, context, event):
            if isinstance(event, PhaseTransitionEvent) and event.current_phase == phase and self.block:
                self.block = False
                entered.set()
                await asyncio.Event().wait()
            if isinstance(event, RunCancelledEvent):
                cancelled.append(event)

        async def on_error(self, context, error):
            errors.append(error)

    agent = await Agent.create(FinalModel(), config=AgentRunConfig("cancel"), extensions=[BlockingObserver()])
    task = asyncio.create_task(agent.run("first"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert agent.state.phase == AgentPhase.CANCELLED
    assert len(cancelled) == 1
    assert cancelled[0].previous_phase == phase
    assert errors == []
    assert (await agent.run("second")).content == "answer"
    assert agent.state.phase == AgentPhase.COMPLETED


async def test_failure_notifications_do_not_mask_the_original_model_exception():
    original = ValueError("model failure")

    class BrokenModel:
        async def stream(self, request):
            yield ModelEvent.text("partial")
            raise original

    class BrokenObserver(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, PhaseTransitionEvent) and event.current_phase == AgentPhase.FAILED:
                raise RuntimeError("failure subscriber")

        async def on_error(self, context, error):
            assert error is original
            raise RuntimeError("error hook")

    agent = await Agent.create(BrokenModel(), config=AgentRunConfig("failure"), extensions=[BrokenObserver()])
    with pytest.raises(ValueError) as caught:
        await agent.run("hello")
    assert caught.value is original
    assert agent.state.phase == AgentPhase.FAILED
    assert len(original.__notes__) == 2


async def test_completed_observer_failure_preserves_terminal_state_and_original_error():
    original = RuntimeError("completed observer")

    class BrokenObserver(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, PhaseTransitionEvent) and event.current_phase == AgentPhase.COMPLETED:
                raise original

    agent = await Agent.create(FinalModel(), config=AgentRunConfig("completion"), extensions=[BrokenObserver()])
    with pytest.raises(RuntimeError) as caught:
        await agent.run("hello")
    assert caught.value is original
    assert agent.state.phase == AgentPhase.COMPLETED


async def test_completed_subscription_cannot_reenter_agent_before_stream_finishes():
    rejected = []

    class ReentrantObserver(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, PhaseTransitionEvent) and event.current_phase == AgentPhase.COMPLETED:
                with pytest.raises(AgentProtocolError) as caught:
                    await agent.run("nested")
                rejected.append(caught.value)

    agent = await Agent.create(FinalModel(), config=AgentRunConfig("reentry"), extensions=[ReentrantObserver()])
    state = None
    stream = agent.stream("hello")
    async for event in stream:
        if event.type == AgentEventType.RUN_COMPLETED:
            state = agent.state
            with pytest.raises(AgentProtocolError):
                await agent.run("before generator exits")
    assert len(rejected) == 1
    assert agent.state is state
    assert (await agent.run("after generator exits")).content == "answer"


async def test_cancel_notification_failure_does_not_break_generator_close():
    class BrokenObserver(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, RunCancelledEvent):
                raise RuntimeError("cancel subscriber")

    agent = await Agent.create(FinalModel(), config=AgentRunConfig("close"), extensions=[BrokenObserver()])
    stream = agent.stream("hello")
    assert (await anext(stream)).type == AgentEventType.MODEL_STARTED
    await stream.aclose()
    assert agent.state.phase == AgentPhase.CANCELLED
    assert (await agent.run("next")).content == "answer"


async def test_close_during_compaction_synchronously_closes_extension_generator():
    closed = []
    cancelled = []

    class Compaction(AgentExtension):
        async def before_model_events(self, context, request):
            try:
                yield AgentEvent(AgentEventType.COMPACTION_STARTED, context.config.session_id)
                yield AgentEvent(AgentEventType.COMPACTION_COMPLETED, context.config.session_id)
            finally:
                closed.append(True)

        async def on_event(self, context, event):
            if isinstance(event, RunCancelledEvent):
                cancelled.append(event)

    agent = await Agent.create(FinalModel(), config=AgentRunConfig("compact"), extensions=[Compaction()])
    stream = agent.stream("hello")
    assert (await anext(stream)).type == AgentEventType.COMPACTION_STARTED
    await stream.aclose()
    assert closed == [True]
    assert agent.state.phase == AgentPhase.CANCELLED
    assert cancelled[0].previous_phase == AgentPhase.COMPACTING


async def test_cancel_running_tool_cleans_up_and_does_not_execute_the_next_tool():
    entered = asyncio.Event()
    closed = []
    executed = []
    cancelled = []

    @tool(guidelines="Wait until cancelled.")
    async def wait_tool() -> str:
        """Wait for external cancellation."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)
        return "done"

    @tool(guidelines="Record an execution.")
    def next_tool() -> str:
        """Record that a subsequent tool ran."""
        executed.append(True)
        return "done"

    class ToolModel:
        async def stream(self, request):
            yield ModelEvent.completed(
                ModelResponse(
                    AssistantMessage(
                        tool_calls=(
                            ToolCall("first", "wait_tool"),
                            ToolCall("second", "next_tool"),
                        )
                    )
                )
            )

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, RunCancelledEvent):
                cancelled.append(event)

    agent = await Agent.create(
        ToolModel(), config=AgentRunConfig("tools"), tools=[wait_tool, next_tool], extensions=[Observer()]
    )
    task = asyncio.create_task(agent.run("run tools"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert closed == [True]
    # Both parallel handlers start before cancellation reaches the batch. The
    # blocking handler is cancelled; the synchronous sibling may already finish.
    assert executed == [True]
    assert agent.state.phase == AgentPhase.CANCELLED
    assert len(cancelled) == 1
    assert cancelled[0].previous_phase == AgentPhase.RUNNING_TOOL
