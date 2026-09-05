from dataclasses import dataclass
from enum import StrEnum

from .messages import AssistantMessage, ToolCall, ToolMessage
from .model import ModelResponse, ToolCallDelta


class AgentEventType(StrEnum):
    """Events consumed by a streaming UI or application."""

    MODEL_STARTED = "model_started"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    RUN_COMPLETED = "run_completed"


@dataclass(slots=True)
class AgentEvent:
    """One event; fields are populated only when relevant to its type."""

    type: AgentEventType
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
