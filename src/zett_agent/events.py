from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, Protocol

from .exceptions import AgentProtocolError
from .messages import AgentMessage, AssistantMessage, ToolCall, ToolMessage, UserMessage
from .model import ModelEvent, ModelEventType, ModelResponse, ToolCallDelta

if TYPE_CHECKING:
    from .extensions.events import (
        CompactionEvent,
        ContentCompletedEvent,
        ContentStartedEvent,
        ExtensionEvent,
        MessageTiming,
        PhaseTransitionEvent,
        ReasoningCompletedEvent,
        ReasoningStartedEvent,
    )


class AgentEventType(StrEnum):
    """Events consumed by a streaming UI or application."""

    COMPACTION_STARTED = "compaction_started"
    COMPACTION_TEXT_DELTA = "compaction_text_delta"
    COMPACTION_REASONING_DELTA = "compaction_reasoning_delta"
    COMPACTION_COMPLETED = "compaction_completed"
    MODEL_STARTED = "model_started"
    CONTENT_STARTED = "content_started"
    CONTENT_COMPLETED = "content_completed"
    REASONING_STARTED = "reasoning_started"
    REASONING_COMPLETED = "reasoning_completed"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    #: An unexecuted call has a persisted skipped result because steering took over.
    TOOL_SKIPPED = "tool_skipped"
    STEERING_STARTED = "steering_started"
    STEERING_COMPLETED = "steering_completed"
    STEERING_INTERRUPTED = "steering_interrupted"
    INTERNAL_MESSAGE_STARTED = "internal_message_started"
    INTERNAL_MESSAGE_COMPLETED = "internal_message_completed"
    INTERNAL_MESSAGE_INTERRUPTED = "internal_message_interrupted"
    RUN_COMPLETED = "run_completed"
    CUSTOM = "custom"


class AgentPhase(StrEnum):
    """Exclusive execution phase of one request-scoped AgentState.

    Complete request state machine. The enclosing ACTIVE REQUEST box gives
    every active phase the same failure and cancellation exits. Bidirectional
    arrows enter an operation to the right and return to READY on completion::

            +-----------------------+
            | CREATED               |
            +-----------------------+
                        |
                        |
        +---------------+-------------------------------------------------------------+
        |               |                  ACTIVE REQUEST                             |
        |               |                  shared error / cancellation exits          |
        |               v                                                             |
        |   +-----------------------+                                                 |
        |   | LOADING_CONTEXT       |                                                 |
        |   | on_tool()             |                                                 |
        |   | on_state()            |                                                 |
        |   | on_message()          |                                                 |
        |   +-----------------------+                                                 |
        |               |                                                             +---------------+
        |               |                                                             |               |
        |               v                                                             |               |
        |   +-----------------------+                  +-----------------------+      |               |
        |   | READY                 |<- compact/done ->| COMPACTING            |      |               |
        |   |                       |                  +-----------------------+      |               |
        |   |                       |                                                 |               |
        |   |                       |                  +-----------------------+      |               |
        |   |                       |<- generate/done >| GENERATING            |      |               |
        |   |                       |                  +-----------------------+      |               |
        |   |                       |                                                 |               |
        |   |                       |                  +-----------------------+      |               |
        |   +-----------------------+<- tool/done ---->| RUNNING_TOOL          |      |               |
        |               |                              +-----------------------+      |               |
        |               |                                                             |               |
        |               |                                                             |               | cancellation
        +---------------+-------------------------------------------------------------+               |
                        |                                         +- exception -----------------------+
                        |                                         |                                   |
                        v                                         v                                   v
            +-----------------------+                +------------------------+         +---------------------------+
            | COMPLETED             |                | FAILED                 |         | CANCELLED                 |
            | after_run()           |                | on_error()             |         | RunCancelledEvent         |
            | on_success()          |                | re-raise error         |         | re-raise cancellation     |
            +-----------------------+                +------------------------+         +---------------------------+

    Active phases are LOADING_CONTEXT, READY, COMPACTING, GENERATING, and
    RUNNING_TOOL. COMPLETED, FAILED, and CANCELLED are terminal for the current
    request. A later request receives a fresh AgentState beginning at CREATED.
    """

    CREATED = "created"
    LOADING_CONTEXT = "loading_context"
    READY = "ready"
    COMPACTING = "compacting"
    GENERATING = "generating"
    RUNNING_TOOL = "running_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def accepts_new_request(self) -> bool:
        """Return whether an agent may start another request from this phase."""
        match self:
            case AgentPhase.CREATED | AgentPhase.COMPLETED | AgentPhase.FAILED | AgentPhase.CANCELLED:
                return True
            case _:
                return False


class PhaseState(Protocol):
    """Minimal mutable state required by phase transitions."""

    phase: AgentPhase


class PhaseContext(Protocol):
    """Context operations required to transition and publish a phase."""

    state: PhaseState

    async def publish(self, event: ExtensionEvent) -> None:
        """Publish one extension event."""


class AgentPhaseTransitionMixin:
    """Own and validate every transition in the Agent request state machine."""

    _ACTIVE_PHASES = frozenset(
        {
            AgentPhase.LOADING_CONTEXT,
            AgentPhase.READY,
            AgentPhase.COMPACTING,
            AgentPhase.GENERATING,
            AgentPhase.RUNNING_TOOL,
        }
    )

    @staticmethod
    async def _transition_phase(
        context: PhaseContext,
        target: AgentPhase,
        *,
        expected: Collection[AgentPhase],
    ) -> PhaseTransitionEvent:
        """Validate, apply, and publish one phase transition."""
        from .extensions.events import PhaseTransitionEvent

        state = context.state
        if state.phase not in expected:
            allowed = ", ".join(sorted(phase.value for phase in expected))
            raise AgentProtocolError(
                f"Invalid agent phase transition from {state.phase.value!r} to {target.value!r}; "
                f"expected one of: {allowed}"
            )
        previous = state.phase
        state.phase = target
        transition = PhaseTransitionEvent(
            previous_phase=previous,
            current_phase=target,
            occurred_at=datetime.now(UTC),
            monotonic_ns=monotonic_ns(),
        )
        await context.publish(transition)
        return transition

    async def _start_context_loading(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.LOADING_CONTEXT, expected=(AgentPhase.CREATED,))

    async def _finish_context_loading(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.LOADING_CONTEXT,))

    async def _start_compaction(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.COMPACTING, expected=(AgentPhase.READY,))

    async def _finish_compaction(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.COMPACTING,))

    async def _start_model_generation(self, context: PhaseContext) -> PhaseTransitionEvent:
        return await self._transition_phase(context, AgentPhase.GENERATING, expected=(AgentPhase.READY,))

    async def _finish_model_generation(self, context: PhaseContext) -> PhaseTransitionEvent:
        return await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.GENERATING,))

    async def _start_tool_execution(self, context: PhaseContext) -> PhaseTransitionEvent:
        return await self._transition_phase(context, AgentPhase.RUNNING_TOOL, expected=(AgentPhase.READY,))

    async def _finish_tool_execution(self, context: PhaseContext) -> PhaseTransitionEvent:
        return await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.RUNNING_TOOL,))

    async def _complete_request(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.COMPLETED, expected=(AgentPhase.READY,))

    async def _fail_request(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.FAILED, expected=self._ACTIVE_PHASES)

    async def _cancel_request(self, context: PhaseContext) -> None:
        """Enter CANCELLED and broadcast its dedicated notification once."""
        from .extensions.events import RunCancelledEvent

        state = context.state
        if state.phase in (AgentPhase.COMPLETED, AgentPhase.CANCELLED):
            return
        transition = await self._transition_phase(context, AgentPhase.CANCELLED, expected=self._ACTIVE_PHASES)
        await context.publish(
            RunCancelledEvent(
                previous_phase=transition.previous_phase,
                occurred_at=transition.occurred_at,
                monotonic_ns=transition.monotonic_ns,
            )
        )

    @staticmethod
    def _require_phase(state: PhaseState, expected: AgentPhase) -> None:
        """Reject an event emitted outside the phase in which it is valid."""
        if state.phase != expected:
            raise AgentProtocolError(f"Agent phase must be {expected.value!r}, found {state.phase.value!r}")


@dataclass(slots=True)
class ModelOutputTracker:
    """Validate and publish output boundaries for one model call.

    Nonempty deltas open a segment once. Content or tool arguments close
    reasoning; reasoning cannot resume afterwards. The final response closes
    content and seals the tracker. Completion requires a matching open segment.
    Failure and cancellation do not synthesize successful completion events.
    """

    reasoning_started: ReasoningStartedEvent | None = None
    reasoning_completed: ReasoningCompletedEvent | None = None
    content_started: ContentStartedEvent | None = None
    content_completed: ContentCompletedEvent | None = None
    _finished: bool = False
    _tools_started: bool = False

    async def observe(self, context: PhaseContext, event: ModelEvent) -> tuple[AgentEventType, ...]:
        """Publish extension boundaries and return their ordered UI event types."""
        from .extensions.events import (
            ContentStartedEvent,
            ReasoningStartedEvent,
        )

        AgentPhaseTransitionMixin._require_phase(context.state, AgentPhase.GENERATING)
        if self._finished:
            raise AgentProtocolError("Model emitted events after its final response")
        boundaries: list[AgentEventType] = []
        match event.type:
            case ModelEventType.REASONING_DELTA if event.delta:
                if self.reasoning_completed is not None or self.content_started is not None or self._tools_started:
                    raise AgentProtocolError("Reasoning cannot start or resume after content or tool arguments")
                if self.reasoning_started is None:
                    self.reasoning_started = ReasoningStartedEvent.now()
                    await context.publish(self.reasoning_started)
                    boundaries.append(AgentEventType.REASONING_STARTED)
            case ModelEventType.TEXT_DELTA if event.delta:
                if self.content_completed is not None:
                    raise AgentProtocolError("Content cannot resume after completion")
                if self.reasoning_started is not None and self.reasoning_completed is None:
                    await self._complete_reasoning(context)
                    boundaries.append(AgentEventType.REASONING_COMPLETED)
                if self.content_started is None:
                    self.content_started = ContentStartedEvent.now()
                    await context.publish(self.content_started)
                    boundaries.append(AgentEventType.CONTENT_STARTED)
            case ModelEventType.TOOL_CALL_DELTA:
                if event.tool_call_delta is None:
                    raise AgentProtocolError("Missing tool-call delta")
                self._tools_started = True
                if self.reasoning_started is not None and self.reasoning_completed is None:
                    await self._complete_reasoning(context)
                    boundaries.append(AgentEventType.REASONING_COMPLETED)
            case ModelEventType.RESPONSE:
                if event.response is None:
                    raise AgentProtocolError("Missing model response")
                if self.reasoning_started is not None and self.reasoning_completed is None:
                    await self._complete_reasoning(context)
                    boundaries.append(AgentEventType.REASONING_COMPLETED)
                if self.content_started is not None and self.content_completed is None:
                    await self._complete_content(context)
                    boundaries.append(AgentEventType.CONTENT_COMPLETED)
                self._finished = True
            case _:
                pass
        return tuple(boundaries)

    async def _complete_reasoning(self, context: PhaseContext) -> None:
        from .extensions.events import ReasoningCompletedEvent

        AgentPhaseTransitionMixin._require_phase(context.state, AgentPhase.GENERATING)
        if self.reasoning_started is None or self.reasoning_completed is not None:
            raise AgentProtocolError("Reasoning completion requires an open reasoning segment")
        self.reasoning_completed = ReasoningCompletedEvent.now()
        await context.publish(self.reasoning_completed)

    async def _complete_content(self, context: PhaseContext) -> None:
        """Close content only after its start and at most once."""
        from .extensions.events import ContentCompletedEvent

        AgentPhaseTransitionMixin._require_phase(context.state, AgentPhase.GENERATING)
        if self.content_started is None or self.content_completed is not None:
            raise AgentProtocolError("Content completion requires an open content segment")
        self.content_completed = ContentCompletedEvent.now()
        await context.publish(self.content_completed)

    def message_timing(
        self,
        started: PhaseTransitionEvent,
        completed: PhaseTransitionEvent,
    ) -> MessageTiming:
        """Build persistable durations from monotonic lifecycle boundaries."""
        from .extensions.events import MessageTiming

        reasoning = self._span(self.reasoning_started, self.reasoning_completed)
        content = self._span(self.content_started, self.content_completed)
        return MessageTiming(
            started_at=started.occurred_at,
            completed_at=completed.occurred_at,
            duration_ns=max(0, completed.monotonic_ns - started.monotonic_ns),
            reasoning_started_at=reasoning[0],
            reasoning_completed_at=reasoning[1],
            reasoning_duration_ns=reasoning[2],
            content_started_at=content[0],
            content_completed_at=content[1],
            content_duration_ns=content[2],
        )

    @staticmethod
    def _span(started, completed) -> tuple[datetime | None, datetime | None, int | None]:
        if started is None or completed is None:
            return None, None, None
        return started.occurred_at, completed.occurred_at, max(0, completed.monotonic_ns - started.monotonic_ns)


@dataclass(slots=True)
class AgentEvent:
    """One UI-facing event emitted by :meth:`Agent.stream`.

    Optional fields are populated according to ``type``. Consumers should first
    match the event type and only then read the corresponding payload.

    .. code-block:: text

        event type                 relevant fields
        -------------------------  -------------------------------------------
        TEXT_DELTA                 delta
        TOOL_STARTED               tool_calls
        TOOL_COMPLETED/FAILED      tool_calls, message, optional error
        MODEL_COMPLETED            response
        RUN_COMPLETED              message
        CUSTOM                     name, payload

    Examples:
        Handle a mixed stream with structural pattern matching::

            async for event in agent.stream("Inspect this project"):
                match event.type:
                    case AgentEventType.TEXT_DELTA:
                        print(event.delta, end="", flush=True)
                    case AgentEventType.TOOL_STARTED:
                        print("tool:", event.tool_calls[0].name)
                    case AgentEventType.CUSTOM:
                        print(event.name, event.payload)

    .. note::
        Model failures and cancellation propagate as exceptions. They are not
        converted into a synthetic terminal ``AgentEvent``.

    .. seealso::
        :class:`~zett_agent.AgentEventDispatcher` provides typed application
        callbacks, and :doc:`/concepts/events` compares all event channels.
    """

    type: AgentEventType
    #: Session that owns the run emitting this event.
    session_id: str
    #: Exclusive request phase when this event was emitted.
    phase: AgentPhase | None = None
    #: Incremental text or reasoning.
    delta: str = ""
    #: Incomplete tool arguments for display, never for execution.
    tool_call_delta: ToolCallDelta | None = None
    #: Complete invocations associated with a tool event. Current execution emits
    #: one item per event; the list shape also supports future concurrent batches.
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Final model response, including usage.
    response: ModelResponse | None = None
    #: Tool result or final assistant answer.
    message: AssistantMessage | ToolMessage | None = None
    #: Internal instruction associated with INTERNAL_MESSAGE_STARTED/COMPLETED.
    internal_message: AgentMessage | None = None
    #: User input associated with STEERING_STARTED/COMPLETED/INTERRUPTED.
    steering_message: UserMessage | None = None
    #: Tool failure; skipped calls have success=False but no execution exception.
    #: Model and cancellation exceptions propagate to the caller.
    error: Exception | None = None
    #: Applied compaction details, populated only by COMPACTION_COMPLETED.
    compaction: CompactionEvent | None = None
    #: Whether a completed compaction replaced context; null for other events.
    applied: bool | None = None
    #: Stable event name used by consumers to match one kind of CUSTOM event.
    #: This field is required and must be non-empty when type is CUSTOM.
    name: str | None = None
    #: Extension-owned data for CUSTOM events. Extensions targeting JSON/SSE
    #: clients must supply JSON-serializable keys and values.
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type == AgentEventType.CUSTOM and (self.name is None or not self.name.strip()):
            raise ValueError("A custom AgentEvent requires a non-empty name")
