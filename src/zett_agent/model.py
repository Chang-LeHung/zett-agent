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
    #: Whether a Responses API provider may defer loading this definition until
    #: tool search selects it. Other provider protocols ignore this flag.
    deferred: bool = False


@dataclass(frozen=True, slots=True)
class ServerToolDefinition:
    """Opaque provider-hosted tool configuration for one model request.

    Unlike :class:`ToolDefinition`, a server tool is executed by the selected
    provider or gateway. The Agent must not dispatch it through its local tool
    loop or append a client-authored ``ToolMessage`` for it.

    ``type`` is the exact provider protocol discriminator, such as
    ``web_search``, ``web_search_20260318``, or ``openrouter:web_fetch``.
    ``configuration`` contains the remaining provider-specific fields. Keeping
    that configuration opaque avoids pretending that differently shaped and
    versioned server tools are interchangeable across providers.

    Examples:
        Configure OpenRouter web fetch::

            ServerToolDefinition(
                type="openrouter:web_fetch",
                configuration={
                    "parameters": {
                        "max_uses": 4,
                        "allowed_domains": ["docs.python.org"],
                    }
                },
            )

        Configure Anthropic web search::

            ServerToolDefinition(
                type="web_search_20260318",
                configuration={"name": "web_search", "max_uses": 5},
            )

    .. note::
        Provider adapters validate whether they understand the requested type.
        A compatible gateway may accept types that the first-party provider
        does not, so this core model intentionally does not use a closed enum.
    """

    #: Exact tool type expected by the selected provider protocol.
    type: str
    #: Provider-specific fields merged beside ``type`` in the wire payload.
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type.strip():
            raise ValueError("Server tool type cannot be empty")
        if "type" in self.configuration:
            raise ValueError("Server tool configuration cannot override type")


@dataclass(frozen=True, slots=True)
class ServerToolCall:
    """Provider-executed tool invocation exposed for observation only.

    Unlike a local :class:`ToolCall`, this call must never be dispatched by the
    Agent or answered with a client-authored ``ToolMessage``. ``input`` may be
    incomplete on the start event when the provider streams its JSON later.
    """

    #: Provider-assigned invocation identifier, or a stable adapter-generated ID.
    id: str
    #: Provider tool name such as ``web_fetch``, ``web_search``, or ``url_context``.
    name: str
    #: Complete input when already available; None while input is still streaming.
    input: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Server tool call ID cannot be empty")
        if not self.name.strip():
            raise ValueError("Server tool call name cannot be empty")


@dataclass(frozen=True, slots=True)
class ServerToolInputDelta:
    """One streamed input fragment for a provider-executed tool invocation."""

    #: Invocation identifier matching :attr:`ServerToolCall.id`.
    call_id: str
    #: Partial serialized JSON; it is not necessarily valid by itself.
    delta: str

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("Server tool input call ID cannot be empty")
        if not self.delta:
            raise ValueError("Server tool input delta cannot be empty")


@dataclass(frozen=True, slots=True)
class ServerToolResult:
    """Terminal output from a tool executed inside the provider."""

    #: Invocation identifier matching :attr:`ServerToolCall.id`.
    call_id: str
    #: Provider tool name matching the corresponding start event.
    name: str
    #: Provider result payload; its shape remains tool-specific.
    output: Any = None
    #: Stable provider error code when the hosted tool failed.
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("Server tool result call ID cannot be empty")
        if not self.name.strip():
            raise ValueError("Server tool result name cannot be empty")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete provider-neutral input for one model step.

    The runtime rebuilds this object before every model call. ``messages``
    therefore includes the current system instructions, restored context, and
    completed tool round trips in chronological order.

    Before-model hooks also receive this type. Each hook receives a fresh
    shallow view of context at entry; the provider receives a final rebuilt
    request after preprocessing. Frozen fields do not freeze nested messages.

    Examples:
        Inspect the shape received by a custom adapter::

            request = ModelRequest(
                messages=[
                    SystemMessage(content="Be concise."),
                    UserMessage(content="What is 20 + 22?"),
                ],
                tools=[add.definition],
                server_tools=[ServerToolDefinition(type="web_search")],
                reasoning_effort=ReasoningEffort.LOW,
            )

    .. note::
        Retry policy belongs to the model adapter, not to this request. One
        ``ModelRequest`` describes intent; it does not control transport attempts.

    .. seealso::
        :class:`~zett_agent.AgentModel` defines the receiving adapter, and
        :class:`~zett_agent.ToolDefinition` defines exposed model tools.
    """

    #: Fully assembled context, including instructions and tool round trips.
    messages: Sequence[AnyMessage]
    #: Tools exposed for this model step; empty means no tool definitions.
    tools: Sequence[ToolDefinition] = ()
    #: Provider-hosted tools that bypass the Agent's local tool executor.
    server_tools: Sequence[ServerToolDefinition] = ()
    #: Desired reasoning level; provider capabilities determine its mapping.
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    #: Optional local function name or registered server-tool type the provider must call.
    tool_choice: str | None = None
    #: Whether the provider may return more than one local tool call in one response.
    #:
    #: Adapters map this preference when their protocol exposes an equivalent
    #: switch. The Agent still decides whether returned calls are safe to run
    #: concurrently; this flag alone never makes local execution concurrent.
    parallel_tool_call: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.parallel_tool_call, bool):
            raise ValueError("parallel_tool_call must be a boolean")


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


def validate_response(response: bool) -> None:
    """Require an explicit boolean for selecting the model transport API."""
    if not isinstance(response, bool):
        raise ValueError("response must be a boolean")


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
    """Final normalized response for one model step.

    Exactly one response terminates a successful model stream. Its message must
    contain complete tool calls even if partial tool-call deltas were emitted.

    Examples:
        Complete a text-only stream::

            response = ModelResponse(
                message=AssistantMessage(content="The answer is 42."),
                finish_reason="stop",
                usage=ModelUsage(input_tokens=18, output_tokens=6),
            )
            yield ModelEvent.completed(response)

    .. seealso::
        :class:`~zett_agent.ModelUsage` normalizes provider token counters.
    """

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
    SERVER_TOOL_STARTED = "server_tool_started"
    SERVER_TOOL_INPUT_DELTA = "server_tool_input_delta"
    SERVER_TOOL_COMPLETED = "server_tool_completed"
    SERVER_TOOL_FAILED = "server_tool_failed"
    RESPONSE = "response"


@dataclass(frozen=True, slots=True)
class ModelEvent:
    """One provider-neutral event emitted by an :class:`AgentModel`.

    .. zett-diagram:: model-stream

        +---------------------+
        | Text deltas         |----------+
        +---------------------+          |
                                         |
        +---------------------+          |
        | Reasoning deltas    |----------+
        +---------------------+          |
                                         |     +------------------------------+
        +---------------------+          +---->| terminal ModelResponse       |
        | Tool-call deltas    |----------+     | message + usage + finish     |
        +---------------------+          |     +------------------------------+
                                         |
        +---------------------+          |
        | Server-tool events  |----------+
        +---------------------+

    Examples:
        A minimal deterministic model can stream text and then return the same
        complete text in its terminal message::

            async def stream(self, request):
                yield ModelEvent.text("Hello")
                yield ModelEvent.completed(
                    ModelResponse(AssistantMessage(content="Hello"))
                )

    .. warning::
        Deltas are incremental display data. Do not append them as independent
        conversation messages, and do not omit the terminal response.

    .. seealso::
        :class:`~zett_agent.AgentEvent` is the higher-level stream consumed by
        applications after runtime lifecycle events are added.
    """

    #: Discriminator indicating which optional event field is populated.
    type: ModelEventType
    #: Incremental answer or reasoning text for the corresponding delta type.
    delta: str = ""
    #: Partial tool identifier/name/arguments for TOOL_CALL_DELTA events.
    tool_call_delta: ToolCallDelta | None = None
    #: Hosted invocation identity for a server-tool start event.
    server_tool_call: ServerToolCall | None = None
    #: Hosted incremental input associated with an earlier start event.
    server_tool_input_delta: ServerToolInputDelta | None = None
    #: Hosted terminal result for server-tool completion or failure events.
    server_tool_result: ServerToolResult | None = None
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
    def server_tool_started(cls, call: ServerToolCall) -> ModelEvent:
        """Expose the start of one provider-executed tool invocation."""
        return cls(ModelEventType.SERVER_TOOL_STARTED, server_tool_call=call)

    @classmethod
    def server_tool_input(cls, delta: ServerToolInputDelta) -> ModelEvent:
        """Expose an incremental hosted-tool JSON input fragment."""
        return cls(ModelEventType.SERVER_TOOL_INPUT_DELTA, server_tool_input_delta=delta)

    @classmethod
    def server_tool_completed(cls, result: ServerToolResult) -> ModelEvent:
        """Expose successful provider-side tool completion."""
        return cls(ModelEventType.SERVER_TOOL_COMPLETED, server_tool_result=result)

    @classmethod
    def server_tool_failed(cls, result: ServerToolResult) -> ModelEvent:
        """Expose provider-side tool failure without converting it to a transport error."""
        return cls(ModelEventType.SERVER_TOOL_FAILED, server_tool_result=result)

    @classmethod
    def completed(cls, response: ModelResponse) -> ModelEvent:
        """Create the single terminal response event for a successful model call."""
        return cls(ModelEventType.RESPONSE, response=response)


@runtime_checkable
class AgentModel(Protocol):
    """Protocol implemented by every provider or custom model adapter.

    An adapter translates :class:`ModelRequest` into one provider request and
    maps the provider stream back into :class:`ModelEvent` objects. The Agent
    remains independent of SDK-specific chunks.

    Examples:
        Implement a complete offline adapter::

            class EchoModel:
                retry = RetryOptions(max_retries=0)

                async def stream(self, request):
                    text = request.messages[-1].text
                    yield ModelEvent.text(text)
                    yield ModelEvent.completed(
                        ModelResponse(AssistantMessage(content=text))
                    )

            model: AgentModel = EchoModel()

    .. note::
        ``stream`` is declared as a regular protocol method returning an
        ``AsyncIterator``. Implementations normally use ``async def`` with
        ``yield``. Runtime protocol checks validate attribute presence, not the
        implementation's async semantics; integration tests must consume it.

    .. seealso::
        :doc:`/extending/model-adapter` provides a complete adapter walkthrough;
        :class:`~zett_agent.RetryOptions` defines the model-owned retry policy.
    """

    #: Model-owned backoff policy; max_retries=0 disables retries.
    #: This configuration never belongs to ModelRequest.
    retry: RetryOptions = RetryOptions()
    #: Whether this adapter uses a Responses-style endpoint instead of its
    #: provider's traditional message or chat-completions endpoint.
    response: bool = False

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream deltas and finish with exactly one response event.

        The model owns and applies its retry configuration for every request.
        Transient failures may be retried only before the first emitted event.
        """
        ...
