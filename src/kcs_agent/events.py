from dataclasses import dataclass
from enum import StrEnum

from .extension_events import CompactionEvent
from .messages import AssistantMessage, ToolCall, ToolMessage
from .model import ModelResponse, ToolCallDelta


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

    State transitions::

        +---------+     +-----------------+     +-------+
        | CREATED | --> | LOADING_CONTEXT | --> | READY |
        +---------+     +-----------------+     +-------+
                                                 |   ^
                              +------------------+   +------------------+
                              v                                         |
                       +------------+     +-------+              +------------+
                       | COMPACTING | --> | READY | ------------>| GENERATING |
                       +------------+     +-------+              +------------+
                                                                    |     |
                                             final answer           |     | tool calls
                              +-----------+ <-----------------------+     v
                              | COMPLETED |                    +--------------+
                              +-----------+                    | RUNNING_TOOL |
                                                               +--------------+
                                                                      |
                                                                      +----> READY

        Any active phase may terminate as FAILED or CANCELLED.
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
