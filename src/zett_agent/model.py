from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
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

    #: Unique provider-visible function name.
    name: str
    #: Natural-language purpose used by the model to select this tool.
    description: str
    #: JSON Schema describing the tool's keyword arguments.
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete input for one model step in an agent run."""

    #: Fully assembled context, including instructions and tool round trips.
    messages: Sequence[AnyMessage]
    #: Tools exposed for this model step; empty means no tool definitions.
    tools: Sequence[ToolDefinition] = ()
    #: Desired reasoning level; provider capabilities determine its mapping.
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM

    #: Optional name of the single tool the provider must call for schema-bound output.
    tool_choice: str | None = None


@dataclass(frozen=True, slots=True)
class RetryOptions:
    """Model-owned exponential backoff, without jitter.

    Wait base_delay seconds before the first retry, then double the delay
    until max_delay. max_retries counts additional attempts, not the initial
    request; zero disables retries. No sleep occurs after the final failure.

    Examples:
        Usage::

            RetryOptions(base_delay=1.0, max_delay=5.0, max_retries=4)
            # At most five requests, separated by 1, 2, 4, and 5 seconds.
    """

    #: Initial retry delay in seconds; zero allows immediate retries.
    base_delay: float = 0.25
    #: Hard upper bound in seconds, including the first retry delay.
    max_delay: float = 8.0
    #: Maximum additional requests after transient failures.
    max_retries: int = 3

    def __post_init__(self) -> None:
        for name in ("base_delay", "max_delay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")


DEFAULT_RETRY_OPTIONS = RetryOptions()


def validate_retry(retry: RetryOptions) -> None:
    """Reject invalid model configuration before opening network clients."""
    if not isinstance(retry, RetryOptions):
        raise ValueError("retry must be a RetryOptions instance")


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Normalized token and cache measurements reported by a provider adapter.

    ``input_tokens`` is the complete model input, including cache reads and
    cache writes. ``output_tokens`` includes reasoning tokens. This invariant
    makes totals and cache-hit ratios comparable across provider schemas.
    """

    #: Total input tokens, including cached input and cache writes.
    input_tokens: int = 0
    #: Total generated tokens, including reasoning tokens when reported.
    output_tokens: int = 0
    #: Input tokens reused from the provider cache, not extra input tokens.
    cache_read_tokens: int = 0
    #: Input tokens written to the provider cache, not extra input tokens.
    cache_write_tokens: int = 0
    #: Reasoning subset of output_tokens; zero when not reported.
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
        """Return input plus output without double-counting cache or reasoning."""
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

    #: Complete assistant output, including any tool calls and replay blocks.
    message: AssistantMessage
    #: Provider completion reason, or None when unavailable.
    finish_reason: str | None = None
    #: Normalized usage for this model call, not the entire conversation.
    usage: ModelUsage = field(default_factory=ModelUsage)


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One streamed fragment of a model-requested tool call."""

    #: Zero-based tool-call index used to assemble parallel streamed fragments.
    index: int
    #: Incremental call identifier; may be empty after the initial fragment.
    id_delta: str = ""
    #: Incremental function name; concatenate before executing the tool.
    name_delta: str = ""
    #: Partial serialized arguments; not necessarily valid standalone JSON.
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

    #: Discriminator indicating which optional event field is populated.
    type: ModelEventType
    #: Incremental answer or reasoning text for the corresponding delta type.
    delta: str = ""
    #: Partial tool identifier/name/arguments for TOOL_CALL_DELTA events.
    tool_call_delta: ToolCallDelta | None = None
    #: Complete final response, present only for RESPONSE events.
    response: ModelResponse | None = None

    @classmethod
    def text(cls, delta: str) -> ModelEvent:
        """Create one answer-text fragment event."""
        return cls(ModelEventType.TEXT_DELTA, delta=delta)

    @classmethod
    def reasoning(cls, delta: str) -> ModelEvent:
        """Create one provider-reported reasoning fragment event."""
        return cls(ModelEventType.REASONING_DELTA, delta=delta)

    @classmethod
    def tool_call(cls, delta: ToolCallDelta) -> ModelEvent:
        """Create one tool-call fragment event; this does not execute a tool."""
        return cls(ModelEventType.TOOL_CALL_DELTA, tool_call_delta=delta)

    @classmethod
    def completed(cls, response: ModelResponse) -> ModelEvent:
        """Create the single terminal response event for a successful model call."""
        return cls(ModelEventType.RESPONSE, response=response)


@runtime_checkable
class AgentModel(Protocol):
    """Model adapter boundary required by the core agent loop."""

    #: Model-owned backoff policy; max_retries=0 disables retries.
    #: This configuration never belongs to ModelRequest.
    retry: RetryOptions = RetryOptions()

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream deltas and finish with exactly one response event.

        The model owns and applies its retry configuration for every request.
        Transient failures may be retried only before the first emitted event.
        """
        ...
