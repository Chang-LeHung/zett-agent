"""The Messages wire format, and the base class of every provider that speaks it.

Several services answer the Anthropic Messages protocol, and each of them bends it a little: one
authenticates with a bearer token, one ignores the cache markers Anthropic needs, one takes the
reasoning level in a field of its own and rejects the token budget. This module splits the work the
way :mod:`~zett_agent.providers.openai_compatible` does for Chat Completions::

    AnthropicCompatibleProvider         one request, one stream, the shared protocol
        |
        |  hooks (all optional, all public)
        |    vendor_headers()        headers the service wants on every request
        |    request_fields()        reasoning and output-budget fields of this request
        |    mark_cache()            where, and whether, to put cache markers
        |    mark_stable_cache()     the stable prefix alone, for CacheMode.READ
        |    start_usage()           token counts at the start of the response
        |    finish_usage()          token counts at its end
        |    message_stop()          why the response ended
        |    status_error()          what an HTTP failure means
        v
    AnthropicProvider  DeepSeekAnthropicProvider  KimiAnthropicProvider  GLMAnthropicProvider

A provider overrides only the hooks where its service differs; a hook that is not overridden is the
Anthropic behavior. The wire is JSON, typed with :data:`~zett_agent.json_types.JsonValue`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import ssl
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, cast

from ..json_types import JsonValue
from ..messages import (
    AgentMessage,
    AnyMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
    SystemMessage,
    TextContent,
    ToolMessage,
    UserMessage,
)
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    NOT_REPORTED_STATUS,
    CacheMode,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    ServerToolCall,
    ServerToolDefinition,
    ServerToolInputDelta,
    ServerToolResult,
    TimeoutOptions,
    ToolCallDelta,
    ToolDefinition,
    validate_retry,
    validate_timeout,
)
from ..stop import StopInfo, StopReason, classify_stop
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderResponseError,
    RetryingProvider,
    ToolCallAccumulator,
    describe_status_error,
    empty_stream_message,
    guard_idle,
    http_error,
    httpx_timeout,
    parse_tool_calls,
    require_event_stream,
    retry_model_stream,
    validate_extra_body,
    validate_extra_headers,
    validate_extra_query,
)
from .responses import local_tool_search_name
from .stop_reasons import withhold_replay_blocks
from .tool_arguments import ToolArgumentsError, decode_tool_arguments
from .tool_images import expand_tool_images
from .wire import JsonObject, read_int, read_object, read_str

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx
    from anthropic import AnthropicError, AsyncStream
    from anthropic.types import RawMessageStreamEvent

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "AnthropicCompatibleProvider",
    "image_block",
    "messages_server_tools",
    "messages_tools",
    "search_result_blocks",
    "to_content_blocks",
]

#: Output budget per response. The original 4096 truncated any tool call that wrote a
#: sizeable file, leaving its JSON arguments incomplete.
DEFAULT_MAX_TOKENS = 16_384


# --- the wire format ---------------------------------------------------------------------------


def messages_tools(
    tools: Sequence[ToolDefinition], server_tools: Sequence[ServerToolDefinition] = ()
) -> list[JsonObject]:
    """Render local tools, deferring the ones tool search is meant to load.

    A tool marked ``deferred`` is declared with ``defer_loading``: Anthropic
    keeps it out of the model's context until a ``tool_reference`` names it.
    That only works when something can produce the reference, so the flag is
    honored only when the request carries a search: the local search tool, or
    Anthropic's hosted ``tool_search_tool_*``. Without one the tool is declared
    normally, because a deferred tool nothing can load would be unusable.

    The local search tool is an ordinary function here, since Anthropic has no
    client-side search endpoint; the model calls it and its result carries the
    references (see :func:`search_result_blocks`).
    """
    if sum(tool.local_tool_search for tool in tools) > 1:
        raise ValueError("Only one local tool search tool can be registered")
    can_load = any(tool.local_tool_search for tool in tools) or any(
        tool.type.startswith("tool_search_tool_") for tool in server_tools
    )
    rendered: list[JsonObject] = []
    for tool in tools:
        payload: JsonObject = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": cast(JsonObject, dict(tool.parameters)),
        }
        if tool.deferred and can_load:
            payload["defer_loading"] = True
        rendered.append(payload)
    return rendered


def messages_server_tools(tools: Sequence[ServerToolDefinition]) -> list[JsonObject]:
    """Render server tools using their versioned native schemas."""
    return [{"type": tool.type, **cast(JsonObject, dict(tool.configuration))} for tool in tools]


def search_result_blocks(message: ToolMessage) -> list[JsonObject]:
    """Turn a local tool search result into ``tool_reference`` blocks.

    The search tool returns the definitions it found; Anthropic needs only their
    names, because each deferred tool is already declared in the request. A
    search that finds nothing is reported as text, since an empty reference list
    is not a valid result.
    """
    try:
        found = json.loads(message.text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ProviderResponseError("A local tool search tool must return JSON tool definitions") from error
    if not isinstance(found, list):
        raise ProviderResponseError("A local tool search tool must return a list of definitions")
    names = [item["name"] for item in found if isinstance(item, Mapping) and isinstance(item.get("name"), str)]
    if len(names) != len(found):
        raise ProviderResponseError("Every definition a local tool search tool returns needs a name")
    if not names:
        return [{"type": "text", "text": "No matching tools found."}]
    return [{"type": "tool_reference", "tool_name": name} for name in names]


def image_block(source: ImageUrlSource | ImageBytesSource) -> JsonObject:
    """Convert a typed image source into a Messages content block."""
    match source:
        case ImageUrlSource(url=url):
            if url.startswith("data:"):
                header, separator, data = url.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ProviderResponseError("Anthropic image data URLs must contain base64 data")
                return {"type": "image", "source": {"type": "base64", "media_type": header[5:-7], "data": data}}
            return {"type": "image", "source": {"type": "url", "url": url}}
        case ImageBytesSource(data=data, media_type=media_type):
            encoded = base64.b64encode(data).decode("ascii")
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}
        case _:
            raise ProviderResponseError(f"Unsupported image content source: {source!r}")


logger = logging.getLogger(__name__)

#: How many refused thinking blocks a provider remembers.
_STALE_LIMIT = 512
#: What the 400 for a thinking block says when the history before it changed. A tampered signature
#: is a different failure, without this sentence, and dropping blocks does not cure it.
_BOUND_ELSEWHERE = re.compile(r"bound to a different conversation", re.IGNORECASE)


def messages_effort(effort: ReasoningEffort) -> str:
    """Name the ``output_config.effort`` value for a reasoning level.

    Messages has five levels, ``low``, ``medium``, ``high``, ``xhigh`` and ``max``. ``OFF`` and
    ``MINIMAL`` share ``low``, and ``ULTRA`` shares ``max``. A service that takes the same field uses
    this to spell the level.
    """
    match effort:
        case ReasoningEffort.OFF | ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
            return "low"
        case ReasoningEffort.MEDIUM:
            return "medium"
        case ReasoningEffort.HIGH:
            return "high"
        case ReasoningEffort.XHIGH:
            return "xhigh"
        case ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return "max"


def _thinking_key(block: Mapping[str, object]) -> str | None:
    """Return what identifies a signed thinking block: its signature, or the data of a redacted one."""
    match block.get("type"):
        case "thinking":
            key = block.get("signature")
        case "redacted_thinking":
            key = block.get("data")
        case _:
            return None
    return key if isinstance(key, str) and key else None


def to_content_blocks(
    message: AnyMessage,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    search_tool: str | None = None,
    stale: Collection[str] = (),
) -> list[JsonObject]:
    """Convert a message into Messages content blocks.

    Args:
        message: The message to render.
        provider: Who produced the replay blocks a message may carry. They are sent back only to the
            provider and model that wrote them.
        model: The model about to receive the message.
        search_tool: Name of the local tool search tool, whose result becomes ``tool_reference``
            blocks; None when the endpoint cannot load deferred tools.
        stale: Signatures of thinking blocks the service refused. They are left out of a replay.
    """
    blocks: list[JsonObject] = []
    match message:
        case UserMessage() | AgentMessage():
            content = message.content
            if isinstance(content, str):
                return [{"type": "text", "text": content}] if content else []
            for part in content:
                match part:
                    case TextContent(text=text):
                        blocks.append({"type": "text", "text": text})
                    case ImageContent(source=source):
                        blocks.append(image_block(source))
            return blocks
        case AssistantMessage():
            if message.provider == provider and message.model == model and message.replay_blocks:
                return [
                    cast(JsonObject, dict(block))
                    for block in message.replay_blocks
                    if _thinking_key(block) not in stale
                ]
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": cast(JsonObject, dict(call.arguments)),
                    }
                )
            return blocks
        case ToolMessage():
            result: JsonValue = (
                cast("list[JsonValue]", search_result_blocks(message))
                if search_tool is not None and message.name == search_tool and message.success
                else message.text
            )
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": result,
                    "is_error": not message.success,
                }
            ]
        case _:
            return []


def _anthropic_stop(finish_reason: str | None, details: Mapping[str, JsonValue], *, has_tool_calls: bool) -> StopInfo:
    """Build the stop of a Messages response from ``stop_reason`` and ``stop_details``.

    ``stop_details`` is null for every reason but ``refusal``. On a refusal its
    ``category`` and ``explanation`` can both be null, which is a normal value and is not
    treated as missing data; the category then falls back to the reason itself.
    """
    kept = {
        key: details[key]
        for key in ("fallback_credit_token", "fallback_has_prefill_claim")
        if details.get(key) is not None
    }
    category, explanation, recommended = (details.get(key) for key in ("category", "explanation", "recommended_model"))
    return classify_stop(
        finish_reason,
        has_tool_calls=has_tool_calls,
        category=str(category) if category else None,
        message=str(explanation) if explanation else None,
        recommended_model=str(recommended) if recommended else None,
        detail=kept,
    )


def _lenient_arguments(payload: str, index: int) -> JsonObject:
    """Decode a replay block's input, using an empty object when it is unusable."""
    try:
        return decode_tool_arguments(payload, index=index)
    except ToolArgumentsError:
        return {}


def _server_tool_error(output: JsonValue) -> str | None:
    """Return the structured hosted-tool error code, when present."""
    if not isinstance(output, dict):
        return None
    output_type = output.get("type")
    if not isinstance(output_type, str) or not output_type.endswith("_error"):
        return None
    code = output.get("error_code")
    return str(code) if code else output_type


def _stream_error_message(error: BaseException) -> str:
    """Return one line that says what a stream failed with."""
    body = read_object(getattr(error, "body", None))
    inner = read_object(body.get("error"))
    return read_str(inner.get("message")) or read_str(body.get("message")) or str(error)


# --- one response as it streams -----------------------------------------------------------------


@dataclass
class _Turn:
    """What has arrived of one streamed response."""

    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    stop_details: JsonObject = field(default_factory=dict)
    usage: ModelUsage = field(default_factory=ModelUsage)
    #: Tool calls the model asked for, by content block index.
    streams: dict[int, ToolCallAccumulator] = field(default_factory=dict)
    #: Calls the provider runs itself, by content block index.
    server_streams: dict[int, ToolCallAccumulator] = field(default_factory=dict)
    #: Hosted call ids already announced in this stream, including synthesized starts.
    started_server_ids: set[str] = field(default_factory=set)
    server_inputs: dict[int, JsonObject] = field(default_factory=dict)
    #: Every content block as it must be sent back, by index.
    replay: dict[int, JsonObject] = field(default_factory=dict)


def _without_unresolved_server_tools(message: AssistantMessage, *, keep_open: bool) -> AssistantMessage:
    """Drop hosted calls that cannot be replayed from one assistant message.

    A hosted ``*_tool_use`` (``server_tool_use``, ``mcp_tool_use``, ...) needs its
    ``*_tool_result`` in the same message. An open call survives only in the last
    assistant message, where it is the pause being continued; every other message
    loses unmatched starts and orphan results. A start and its result in different
    messages are treated as unpaired, which favors a history the endpoint accepts
    over keeping a split result the model would otherwise see.
    """
    blocks = message.replay_blocks
    starts = {read_str(block.get("id")) for block in blocks if read_str(block.get("type")).endswith("_tool_use")}
    results = {
        read_str(block.get("tool_use_id")) for block in blocks if read_str(block.get("type")).endswith("_tool_result")
    }
    kept = tuple(
        block
        for block in blocks
        if not (
            (
                read_str(block.get("type")).endswith("_tool_use")
                and not keep_open
                and read_str(block.get("id")) not in results
            )
            or (
                read_str(block.get("type")).endswith("_tool_result")
                and read_str(block.get("tool_use_id")) not in starts
            )
        )
    )
    return replace(message, replay_blocks=kept) if len(kept) != len(blocks) else message


# --- the provider ---------------------------------------------------------------------------------


class AnthropicCompatibleProvider(RetryingProvider):
    """Stream a service that speaks the Anthropic Messages protocol.

    This is the base class of :class:`~zett_agent.providers.anthropic.AnthropicProvider` and the
    providers for the services that bend the protocol. A provider for another one is a subclass
    that sets :attr:`provider_name` and overrides the hooks below where the service differs; it
    does not copy the stream loop.

    * :meth:`vendor_headers` -- headers sent on every request.
    * :meth:`request_fields` -- the reasoning and output-budget fields of one request.
    * :meth:`mark_cache` -- cache markers.
    * :meth:`mark_stable_cache` -- the stable tool and system prefix used by ``CacheMode.READ``.
    * :meth:`start_usage` and :meth:`finish_usage` -- token counts.
    * :meth:`message_stop` -- why the response ended.
    * :meth:`status_error` -- what an HTTP failure means.

    Claude 5.x binds a signed thinking block to the system prompt, the tools, and the messages before
    it. When one of them changed, for instance because compaction replaced older turns with a
    summary, the service refuses the block with ``400 ... bound to a different conversation``.
    The provider then drops the signed blocks of that request, sends it once more, and no longer
    replays them, which is Anthropic's documented fallback. The model loses its earlier reasoning
    from then on, so each occurrence is logged as a warning.

    Args:
        model: Model identifier the endpoint accepts.
        api_key: API credential supplied by the application.
        base_url: API root. Required unless the subclass sets :attr:`default_base_url`.
        transport: Optional HTTPX transport for isolated tests or custom routing.
        retry: Exponential backoff for transient failures before output starts.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        prompt_cache: Mark cache breakpoints, for a service that reads them. Off by default.
            A request's ``cache_mode`` can narrow this to the stable prefix or turn it off.
        max_tokens: Output budget for one response. Thinking is paid out of it.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last.
    """

    #: Name stored on messages this provider produced. Signed blocks are sent back only to the same
    #: name and model.
    provider_name: str = "anthropic"
    #: Authenticate with ``Authorization: Bearer`` as well as ``x-api-key``. The services that take
    #: only the bearer token ignore the other header.
    bearer_auth: ClassVar[bool] = False
    #: See :attr:`zett_agent.model.NativeDeferredToolsModel.native_deferred_tools`. Only Anthropic
    #: itself loads a ``defer_loading`` tool when a ``tool_reference`` names it.
    native_deferred_tools: bool = False
    #: API root used when no ``base_url`` is given. A subclass for one service sets it.
    default_base_url: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
        prompt_cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> None:
        import httpx
        import truststore

        if not api_key:
            raise ValueError("api_key is required")
        base_url = base_url or self.default_base_url
        if not base_url:
            raise ValueError("base_url is required")
        validate_retry(retry)
        validate_timeout(timeout)
        if not isinstance(prompt_cache, bool):
            raise ValueError("prompt_cache must be a boolean")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.extra_headers = validate_extra_headers(extra_headers)
        self.extra_query = validate_extra_query(extra_query)
        self.extra_body: dict[str, JsonValue] = validate_extra_body(extra_body)
        self.retry = retry
        self.timeout = timeout
        self.prompt_cache = prompt_cache
        self.model = model
        #: Signatures of thinking blocks the service refused, oldest first. See :meth:`_forget_thinking`.
        self._stale_thinking: dict[str, None] = {}
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        from anthropic import AsyncAnthropic

        headers = {**self.vendor_headers(), **self.extra_headers}
        self._client = AsyncAnthropic(
            api_key=api_key,
            auth_token=api_key if self.bearer_auth else None,
            base_url=base_url,
            http_client=self._http_client,
            max_retries=0,
            timeout=httpx_timeout(timeout),
            default_headers=headers or None,
            default_query=self.extra_query or None,
        )

    # -- hooks ------------------------------------------------------------------------------------

    def vendor_headers(self) -> dict[str, str]:
        """Return the headers this service wants on every request.

        They are sent before ``extra_headers``, so the application can still replace one.
        """
        return {}

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Return the reasoning and output-budget fields of ``request``.

        These are merged over the standard body. The default is Anthropic's: adaptive thinking, with
        the reasoning level sent as ``output_config.effort`` and no token budget, so ``max_tokens`` is
        left as configured. The model decides how much to think within that level. Thinking cannot be
        combined with a forced tool choice, and is explicitly disabled when the level is ``OFF``. A
        field given in ``extra_body`` replaces one returned here.
        """
        if request.reasoning_effort == ReasoningEffort.OFF or request.tool_choice:
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": messages_effort(request.reasoning_effort)},
        }

    def mark_cache(self, body: JsonObject, client_tools: list[JsonObject]) -> None:
        """Put cache markers into ``body`` (and the tool definitions in ``client_tools``).

        The default marks Anthropic's breakpoints on the largest stable request prefixes: the tool
        definitions, the whole system prompt, and the conversation so far. Anthropic caches only
        what a ``cache_control`` marker asks for, then reads each cached prefix at a fraction of the
        input price. A prefix shorter than the model's minimum cacheable length is ignored silently.
        Nothing is marked unless ``prompt_cache`` is on. Called only for ``CacheMode.READ_WRITE``;
        ``CacheMode.READ`` calls :meth:`mark_stable_cache` instead.
        """
        if not self.prompt_cache:
            return
        self.mark_stable_cache(body, client_tools)
        messages = body.get("messages")
        if isinstance(messages, list) and messages and isinstance(last := messages[-1], dict):
            content = last.get("content")
            if isinstance(content, list) and content and isinstance(final := content[-1], dict):
                final["cache_control"] = {"type": "ephemeral"}

    def mark_stable_cache(self, body: JsonObject, client_tools: list[JsonObject]) -> None:
        """Mark only the stable tool and system prefix, for ``CacheMode.READ``.

        Override this hook when a service needs different stable-prefix markers. The base version
        marks the last loaded tool definition and the system prompt, and nothing newer; the newest
        turn is left unmarked so a one-off request can reuse a warm prefix without writing its new
        content.
        """
        if not self.prompt_cache:
            return
        # Server tools are provider-owned definitions and must not carry the marker. A deferred tool
        # is not part of the cached prefix, so the marker goes on the last tool loaded up front.
        loaded = [tool for tool in client_tools if not tool.get("defer_loading")]
        if loaded:
            loaded[-1]["cache_control"] = {"type": "ephemeral"}
        system = body.get("system")
        if isinstance(system, str) and system:
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def start_usage(self, usage: Mapping[str, JsonValue]) -> ModelUsage:
        """Read the ``usage`` object of the ``message_start`` event.

        Cache reads and writes are reported next to ``input_tokens``, not inside it, so the total input
        is their sum.
        """
        cache_read = read_int(usage.get("cache_read_input_tokens"))
        cache_write = read_int(usage.get("cache_creation_input_tokens"))
        return ModelUsage(
            input_tokens=read_int(usage.get("input_tokens")) + cache_read + cache_write,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=read_int(usage.get("output_tokens")),
        )

    def finish_usage(self, usage: ModelUsage, update: Mapping[str, JsonValue]) -> ModelUsage:
        """Fold the ``usage`` object of the ``message_delta`` event into ``usage``.

        The default takes the final ``output_tokens``, which is the only count Anthropic updates.
        """
        output = read_int(update.get("output_tokens"))
        return replace(usage, output_tokens=output, reasoning_tokens=min(usage.reasoning_tokens, output))

    def message_stop(
        self, finish_reason: str | None, details: Mapping[str, JsonValue], *, has_tool_calls: bool
    ) -> StopInfo:
        """Say why the response ended, from ``stop_reason`` and ``stop_details``."""
        return _anthropic_stop(finish_reason, details, has_tool_calls=has_tool_calls)

    def status_error(self, error: AnthropicError) -> ProviderError:
        """Turn an SDK failure from opening the request into the typed error the runtime reports."""
        if getattr(error, "status_code", None) is None:
            # No HTTP response at all: the connection failed or timed out. The SDK's own text
            # ("Connection error.") is what says why; a timeout is recognized from the chain by the
            # retry layer, which turns it into a typed error.
            return ProviderResponseError(f"{self.provider_name} request failed: {error}")
        status, failure, description = describe_status_error(self.provider_name, error)
        if status == 401:
            return ProviderAuthError(f"Provider rejected API credential. {description}")
        return http_error(status, failure, description)

    # -- requests ---------------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the owned SDK connection pool."""
        await self._client.close()

    def _messages(self, request: ModelRequest) -> tuple[list[JsonValue], list[str]]:
        """Render the conversation as Messages turns, and collect the system prompt."""
        messages: list[JsonValue] = []
        system: list[str] = []
        search_tool = local_tool_search_name(request.tools) if self.native_deferred_tools else None
        expanded = list(expand_tool_images(request.messages))
        for index, message in enumerate(expanded):
            if isinstance(message, AssistantMessage) and message.replay_blocks:
                message = _without_unresolved_server_tools(message, keep_open=index == len(expanded) - 1)
            if isinstance(message, SystemMessage):
                system.append(message.content)
                continue
            role = "assistant" if isinstance(message, AssistantMessage) else "user"
            blocks = to_content_blocks(
                message,
                provider=self.provider_name,
                model=self.model,
                search_tool=search_tool if isinstance(message, ToolMessage) else None,
                stale=self._stale_thinking,
            )
            if blocks:
                messages.append({"role": role, "content": cast("list[JsonValue]", blocks)})
        return messages, system

    def _body(self, request: ModelRequest) -> JsonObject:
        messages, system = self._messages(request)
        client_tools = messages_tools(request.tools, request.server_tools)
        tools = [*client_tools, *messages_server_tools(request.server_tools)]
        body: JsonObject = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if system:
            body["system"] = "\n\n".join(system)
        if tools:
            body["tools"] = cast("list[JsonValue]", tools)
        if request.tool_choice:
            body["tool_choice"] = {
                "type": "tool",
                "name": request.tool_choice,
                "disable_parallel_tool_use": not request.parallel_tool_call,
            }
        elif not request.parallel_tool_call and (request.tools or request.server_tools):
            body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if request.cache_mode is CacheMode.READ:
            self.mark_stable_cache(body, client_tools)
        elif request.cache_mode is CacheMode.READ_WRITE:
            self.mark_cache(body, client_tools)
        return body

    def _forget_thinking(self, error: AnthropicError, body: JsonObject) -> bool:
        """Remember the signed thinking blocks of ``body`` when the service refused them.

        The newer Claude models bind a thinking block to everything sent before it: the system
        prompt, the tools, and the earlier messages. When any of that changed since the block was
        written, for example because older turns became a summary or a tool was added, the service
        answers ``400`` with ``The block is bound to a different conversation``. Which block failed
        is only named by position, and a block after the first failure fails too, so every signed
        block of the refused request is dropped, from this request and from later ones. Once a
        block is gone it stays gone: sending it again would invalidate the blocks written since.

        The change is recorded in the log, because the model loses its earlier reasoning from then on.

        Returns:
            True when blocks were newly forgotten, so the same request is worth sending again.
        """
        if getattr(error, "status_code", None) != 400 or not _BOUND_ELSEWHERE.search(str(error)):
            return False
        messages = body.get("messages")
        found: list[str] = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                key = _thinking_key(block) if isinstance(block, dict) else None
                if key is not None and key not in self._stale_thinking:
                    found.append(key)
        for key in found:
            self._stale_thinking[key] = None
        while len(self._stale_thinking) > _STALE_LIMIT:
            del self._stale_thinking[next(iter(self._stale_thinking))]
        if found:
            logger.warning(
                "%s refused %d signed thinking block(s) because the history before them changed; "
                "resending without them: %s",
                self.provider_name,
                len(found),
                str(error)[:300],
            )
        return bool(found)

    async def _open(self, request: ModelRequest) -> AsyncStream[RawMessageStreamEvent]:
        from anthropic import AnthropicError

        body = self._request_body(request)
        # The request body is assembled as a dict, which the SDK's many typed overloads cannot match
        # (they select on literal ``stream=True``), so the call is typed by what it returns here.
        create = cast("Callable[..., Awaitable[AsyncStream[RawMessageStreamEvent]]]", self._client.messages.create)
        try:
            return await self._first_byte(create(**body))
        except AnthropicError as error:
            if not self._forget_thinking(error, body):
                raise self.status_error(error) from error
        # The refused request generated nothing, so sending it again without the blocks is safe.
        try:
            return await self._first_byte(create(**self._request_body(request)))
        except AnthropicError as error:
            raise self.status_error(error) from error

    def _request_body(self, request: ModelRequest) -> JsonObject:
        """Build the body of one request, with the service's own fields in ``extra_body``.

        Fields of the service's own go in the JSON body through ``extra_body``: the SDK rejects a
        keyword it does not declare, and declares only what Anthropic itself reads.
        """
        body = self._body(request)
        if extra := {**self.request_fields(request), **self.extra_body}:
            body["extra_body"] = extra
        return body

    # -- the stream -------------------------------------------------------------------------------

    def _read_event(self, turn: _Turn, data: JsonObject) -> list[ModelEvent]:
        """Apply one streamed event to ``turn`` and return the model events it produces."""
        events: list[ModelEvent] = []
        index = read_int(data.get("index"))
        match data.get("type"):
            case "message_start":
                turn.usage = self.start_usage(read_object(read_object(data.get("message")).get("usage")))
            case "content_block_delta":
                delta = read_object(data.get("delta"))
                block = turn.replay.setdefault(index, {})
                match delta.get("type"):
                    case "text_delta":
                        chunk = read_str(delta.get("text"))
                        turn.text += chunk
                        block["text"] = read_str(block.get("text")) + chunk
                        events.append(ModelEvent.text(chunk))
                    case "thinking_delta":
                        chunk = read_str(delta.get("thinking"))
                        turn.reasoning += chunk
                        block["thinking"] = read_str(block.get("thinking")) + chunk
                        events.append(ModelEvent.reasoning(chunk))
                    case "signature_delta":
                        block["signature"] = read_str(block.get("signature")) + read_str(delta.get("signature"))
                    case "input_json_delta":
                        if partial := read_str(delta.get("partial_json")):
                            if (server := turn.server_streams.get(index)) is not None:
                                server.argument_buffer += partial
                                events.append(
                                    ModelEvent.server_tool_input(ServerToolInputDelta(server.call_id, partial))
                                )
                            else:
                                call = turn.streams.setdefault(index, ToolCallAccumulator(index=index))
                                call.append(ToolCallDelta(index=index, arguments_delta=partial))
                                events.append(ModelEvent.tool_call(ToolCallDelta(index=index, arguments_delta=partial)))
            case "content_block_start":
                block = read_object(data.get("content_block"))
                turn.replay[index] = dict(block)
                match block.get("type"):
                    case "tool_use":
                        call = turn.streams.setdefault(index, ToolCallAccumulator(index=index))
                        call.call_id = read_str(block.get("id")) or call.call_id
                        call.name = read_str(block.get("name")) or call.name
                        events.append(
                            ModelEvent.tool_call(
                                ToolCallDelta(index=index, id_delta=call.call_id, name_delta=call.name)
                            )
                        )
                    case "server_tool_use":
                        call_id, name = read_str(block.get("id")), read_str(block.get("name"))
                        initial = read_object(block.get("input"))
                        turn.server_streams[index] = ToolCallAccumulator(index=index, call_id=call_id, name=name)
                        turn.started_server_ids.add(call_id)
                        turn.server_inputs[index] = initial
                        events.append(ModelEvent.server_tool_started(ServerToolCall(call_id, name, initial or None)))
                    case str(block_type) if block_type.endswith("_tool_result"):
                        call_id = read_str(block.get("tool_use_id"))
                        name = block_type.removesuffix("_tool_result")
                        if call_id and call_id not in turn.started_server_ids:
                            # A paused turn resumes with the result of a call whose start was sent in
                            # the previous step. The tracker is per step, so expose the start again;
                            # the runtime suppresses it when it already carried the call.
                            events.append(ModelEvent.server_tool_started(ServerToolCall(call_id, name, None)))
                            turn.started_server_ids.add(call_id)
                        output = block.get("content")
                        error_code = _server_tool_error(output)
                        result = ServerToolResult(call_id, name, output, error_code)
                        events.append(
                            ModelEvent.server_tool_failed(result)
                            if error_code is not None
                            else ModelEvent.server_tool_completed(result)
                        )
            case "content_block_stop":
                # A malformed block is not fatal here: the replay input falls back to an empty object,
                # and the completed response reports the call as failed to the model.
                if index in turn.streams:
                    turn.replay.setdefault(index, {})["input"] = _lenient_arguments(
                        turn.streams[index].argument_buffer, index
                    )
                if (server := turn.server_streams.get(index)) is not None:
                    turn.replay.setdefault(index, {})["input"] = (
                        _lenient_arguments(server.argument_buffer, index)
                        if server.argument_buffer
                        else turn.server_inputs[index]
                    )
            case "message_delta":
                delta = read_object(data.get("delta"))
                reason = delta.get("stop_reason")
                turn.finish_reason = reason if isinstance(reason, str) else None
                if isinstance(details := delta.get("stop_details"), dict):
                    turn.stop_details = details
                turn.usage = self.finish_usage(turn.usage, read_object(data.get("usage")))
        return events

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        from anthropic import AnthropicError, APIConnectionError, APITimeoutError

        response = await self._open(request)
        await require_event_stream(response, self.provider_name)
        turn = _Turn()
        stream_error: JsonObject = {}
        events_seen = 0
        try:
            async for event in guard_idle(response, self.timeout):
                events_seen += 1
                for model_event in self._read_event(
                    turn, cast(JsonObject, event.model_dump(exclude_none=True, warnings=False))
                ):
                    yield model_event
        except AnthropicError as error:
            # An ``error`` event inside a 200 stream (an overloaded upstream, say) is raised by the
            # SDK from the loop above. Before any output it is a typed error, so a transient failure is
            # retried; after output the response ends here as an interruption and keeps what was written.
            if isinstance(error, (APIConnectionError, APITimeoutError)):
                raise  # a dropped or timed-out connection is not an error the provider reported
            if not (turn.text or turn.reasoning or turn.streams or turn.server_streams):
                raise ProviderResponseError(_stream_error_message(error)) from error
            stream_error = {"message": _stream_error_message(error)}
        finally:
            await response.close()

        if not events_seen:
            raise ProviderResponseError(empty_stream_message(self.provider_name, response))
        tool_calls = parse_tool_calls(turn.streams, turn.finish_reason)
        for index, call in turn.streams.items():
            turn.replay.setdefault(index, {})["input"] = _lenient_arguments(call.argument_buffer, index)
        if stream_error:
            stop = classify_stop(
                "error",
                has_tool_calls=bool(tool_calls),
                message=str(stream_error["message"]),
                detail={"error": stream_error},
            )
        else:
            stop = self.message_stop(turn.finish_reason, turn.stop_details, has_tool_calls=bool(tool_calls))
        blocks: tuple[Mapping[str, JsonValue], ...] = tuple(turn.replay[index] for index in sorted(turn.replay))
        reported = {
            read_str(block.get("tool_use_id"))
            for block in blocks
            if read_str(block.get("type")).endswith("_tool_result")
        }
        if stop.reason is not StopReason.PAUSED:
            closed = set(reported)
            for server in turn.server_streams.values():
                if server.call_id in closed:
                    continue
                closed.add(server.call_id)
                logger.warning(
                    "%s ended with an open server tool %s (%s); reporting it without a result",
                    self.provider_name,
                    server.name,
                    server.call_id,
                )
                yield ModelEvent.server_tool_completed(
                    ServerToolResult(server.call_id, server.name, {"status": NOT_REPORTED_STATUS})
                )
            # A server tool that started but never reported a result cannot be replayed: the API
            # rejects a ``server_tool_use`` block without its matching ``*_tool_result``. Drop the
            # open start instead of persisting a history the next turn refuses.
            blocks = tuple(
                block
                for block in blocks
                if not (read_str(block.get("type")) == "server_tool_use" and read_str(block.get("id")) not in reported)
            )
        if stop.reason in (StopReason.REFUSAL, StopReason.INTERRUPTED):
            # A refusal can arrive after a complete ``tool_use`` was streamed. The call is withheld from
            # the message, and a ``tool_use`` block replayed without a result would be rejected, so the
            # block goes as well.
            blocks = withhold_replay_blocks(blocks, lambda block: block.get("type") == "tool_use")
        yield ModelEvent.completed(
            ModelResponse(
                message=AssistantMessage(
                    content=turn.text,
                    reasoning=turn.reasoning or None,
                    tool_calls=tool_calls,
                    provider=self.provider_name,
                    model=self.model,
                    replay_blocks=blocks,
                ),
                finish_reason=turn.finish_reason,
                usage=turn.usage,
                stop=stop,
            )
        )
