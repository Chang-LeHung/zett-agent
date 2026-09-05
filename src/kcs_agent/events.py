from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from .exceptions import AgentProtocolError
from .messages import AssistantMessage, ToolCall, ToolMessage
from .model import ModelResponse, ToolCallDelta

if TYPE_CHECKING:
    from .extension_events import CompactionEvent, ExtensionEvent


class AgentEventType(StrEnum):
    """Events consumed by a streaming UI or application."""

    COMPACTION_STARTED = "compaction_started"
    COMPACTION_TEXT_DELTA = "compaction_text_delta"
    COMPACTION_REASONING_DELTA = "compaction_reasoning_delta"
    COMPACTION_COMPLETED = "compaction_completed"
    MODEL_STARTED = "model_started"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    RUN_COMPLETED = "run_completed"


class AgentPhase(StrEnum):
    """Exclusive execution phase of one request-scoped AgentState.

    Complete transition diagram::

        +---------+     +-----------------+     +-------+   final answer   +-----------+
        | CREATED | --> | LOADING_CONTEXT | --> | READY | --------------> | COMPLETED |
        +---------+     +-----------------+     +---+---+                 +-----------+
                                                   |
                         +-------------------------+-------------------------+
                         | compact                 | generate                | tool call
                         v                         v                         v
                  +------------+            +------------+           +--------------+
                  | COMPACTING |            | GENERATING |           | RUNNING_TOOL |
                  +------+-----+            +------+-----+           +-------+------+
                         |                         |                         |
                         +-------------------------+-------------------------+
                                                   |
                                                   | operation completed
                                                   v
                                               +-------+
                                               | READY |
                                               +-------+

                                         +------------------+
                                         | ANY ACTIVE PHASE |
                                         +----+--------+----+
                                              |        |
                                    exception |        | cancellation
                                              v        v
                                         +--------+  +-----------+
                                         | FAILED |  | CANCELLED |
                                         +--------+  +-----------+

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
    ) -> None:
        """Validate, apply, and publish one phase transition."""
        from .extension_events import PhaseTransitionEvent

        state = context.state
        if state.phase not in expected:
            allowed = ", ".join(sorted(phase.value for phase in expected))
            raise AgentProtocolError(
                f"Invalid agent phase transition from {state.phase.value!r} to {target.value!r}; "
                f"expected one of: {allowed}"
            )
        previous = state.phase
        state.phase = target
        await context.publish(PhaseTransitionEvent(previous_phase=previous, current_phase=target))

    async def _start_context_loading(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.LOADING_CONTEXT, expected=(AgentPhase.CREATED,))

    async def _finish_context_loading(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.LOADING_CONTEXT,))

    async def _start_compaction(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.COMPACTING, expected=(AgentPhase.READY,))

    async def _finish_compaction(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.COMPACTING,))

    async def _start_model_generation(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.GENERATING, expected=(AgentPhase.READY,))

    async def _finish_model_generation(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.GENERATING,))

    async def _start_tool_execution(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.RUNNING_TOOL, expected=(AgentPhase.READY,))

    async def _finish_tool_execution(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.READY, expected=(AgentPhase.RUNNING_TOOL,))

    async def _complete_request(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.COMPLETED, expected=(AgentPhase.READY,))

    async def _fail_request(self, context: PhaseContext) -> None:
        await self._transition_phase(context, AgentPhase.FAILED, expected=self._ACTIVE_PHASES)

    async def _cancel_request(self, context: PhaseContext) -> None:
        """Cancel an active request while preserving an already completed state."""
        state = context.state
        if state.phase == AgentPhase.COMPLETED:
            return
        await self._transition_phase(context, AgentPhase.CANCELLED, expected=self._ACTIVE_PHASES)

    @staticmethod
    def _require_phase(state: PhaseState, expected: AgentPhase) -> None:
        """Reject an event emitted outside the phase in which it is valid."""
        if state.phase != expected:
            raise AgentProtocolError(f"Agent phase must be {expected.value!r}, found {state.phase.value!r}")


@dataclass(slots=True)
class AgentEvent:
    """One event; fields are populated only when relevant to its type."""

    type: AgentEventType
    # Session that owns the run emitting this event.
    session_id: str
    # Exclusive request phase when this event was emitted.
    phase: AgentPhase | None = None
    # Incremental text or reasoning.
    delta: str = ""
    # Incomplete tool arguments for display, never for execution.
    tool_call_delta: ToolCallDelta | None = None
    # Complete invocation associated with a tool event.
    call: ToolCall | None = None
    # Final model response, including usage.
    response: ModelResponse | None = None
    # Tool result or final assistant answer.
    message: AssistantMessage | ToolMessage | None = None
    # Tool failure; model and cancellation exceptions propagate to the caller.
    error: Exception | None = None
    # Applied compaction details, populated only by COMPACTION_COMPLETED.
    compaction: CompactionEvent | None = None
    # Whether a completed compaction replaced context; null for other events.
    applied: bool | None = None
