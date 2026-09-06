from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .messages import AnyMessage, AssistantMessage


class ReasoningEffort(StrEnum):
    """Provider-neutral levels for controlling model reasoning effort."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Provider-neutral tool metadata supplied with a model request."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete input for one model step in an agent run."""

    messages: Sequence[AnyMessage]
    tools: Sequence[ToolDefinition] = ()
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM

    # Optional name of the single tool the provider must call for schema-bound output.
    tool_choice: str | None = None


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Normalized token and cache measurements reported by a provider adapter.

    ``input_tokens`` is the complete model input, including cache reads and
    cache writes. ``output_tokens`` includes reasoning tokens. This invariant
    makes totals and cache-hit ratios comparable across provider schemas.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.reasoning_tokens,
        )
        if any(value < 0 for value in values):
            raise ValueError("Model usage counters cannot be negative")
        if self.cache_read_tokens > self.input_tokens:
            raise ValueError("cache_read_tokens cannot exceed input_tokens")
        if self.cache_write_tokens > self.input_tokens:
            raise ValueError("cache_write_tokens cannot exceed input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        """Return cached input divided by total input, or None without input."""
        if self.input_tokens == 0:
            return None
        return self.cache_read_tokens / self.input_tokens


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Final normalized response for one model step."""

    message: AssistantMessage
    finish_reason: str | None = None
    usage: ModelUsage = field(default_factory=ModelUsage)


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One streamed fragment of a model-requested tool call."""

    index: int
    id_delta: str = ""
    name_delta: str = ""
    arguments_delta: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Tool call delta index cannot be negative")
        if not (self.id_delta or self.name_delta or self.arguments_delta):
            raise ValueError("Tool call delta must contain an ID, name, or arguments fragment")


class ModelEventType(StrEnum):
    """Streaming events emitted by a model adapter."""

    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    RESPONSE = "response"


@dataclass(frozen=True, slots=True)
class ModelEvent:
    """One provider-neutral model stream event."""

    type: ModelEventType
    delta: str = ""
    tool_call_delta: ToolCallDelta | None = None
    response: ModelResponse | None = None

    @classmethod
    def text(cls, delta: str) -> ModelEvent:
        return cls(ModelEventType.TEXT_DELTA, delta=delta)

    @classmethod
    def reasoning(cls, delta: str) -> ModelEvent:
        return cls(ModelEventType.REASONING_DELTA, delta=delta)

    @classmethod
    def tool_call(cls, delta: ToolCallDelta) -> ModelEvent:
        return cls(ModelEventType.TOOL_CALL_DELTA, tool_call_delta=delta)

    @classmethod
    def completed(cls, response: ModelResponse) -> ModelEvent:
        return cls(ModelEventType.RESPONSE, response=response)


@runtime_checkable
class AgentModel(Protocol):
    """Model adapter boundary required by the core agent loop."""

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream deltas and finish with exactly one response event."""
        ...
