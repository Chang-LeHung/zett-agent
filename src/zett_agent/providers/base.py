from __future__ import annotations

import asyncio
import base64
import json
import ssl
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from .._compat import TypeAliasType, aclosing_stream
from .._compat import timeout as async_timeout
from ..exceptions import AgentError
from ..messages import (
    AnyMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageDetail,
    ImageUrlSource,
    Message,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    ServerToolDefinition,
    TimeoutOptions,
    ToolCallDelta,
    ToolDefinition,
    validate_response,
    validate_retry,
    validate_timeout,
)

if TYPE_CHECKING:
    import httpx
    from openai.types.chat import (
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartParam,
        ChatCompletionToolParam,
    )
    from openai.types.shared_params import FunctionDefinition

#: The OpenAI SDK is imported where this adapter needs it -- client construction
#: and request failures -- because importing it alone costs a few hundred
#: milliseconds. Payload rendering therefore uses plain dictionaries that still
#: type-check against the SDK's TypedDicts.
from ..stop import StopInfo, StopReason, classify_stop
from ..sync_runtime import SyncMethodsMixin
from .http_errors import HttpFailure, failure_message, read_failure
from .responses import (
    ResponsesStreamInterrupted,
    local_tool_search_name,
    responses_input,
    responses_reasoning,
    responses_tools,
    stream_responses,
)
from .stop_reasons import refusal_from_status_error
from .tool_arguments import ToolArgumentsError, decode_tool_arguments, tool_call_from_arguments
from .tool_images import expand_tool_images


class ProviderError(AgentError):
    """Base error for provider adapter failures."""


class ProviderAuthError(ProviderError):
    """Raised when a remote provider rejects the configured credentials."""


class ProviderResponseError(ProviderError):
    """Raised when provider responses do not satisfy the expected stream protocol."""


class ProviderHTTPError(ProviderResponseError):
    """The provider answered with an HTTP failure, or with something that was not an API reply.

    The message says what was received: the status, the provider's own explanation when the
    body was a JSON error, and whether it was a web page (a proxy, gateway, firewall or login
    page in front of the API). The original SDK error stays on ``__cause__``.

    Attributes:
        status_code: The HTTP status of the response.
        reason: The provider's own explanation, or None when the body had none.
        preview: The start of the body as plain text, for a body with no explanation.
        is_html: Whether the body was a web page.
    """

    def __init__(
        self, message: str, *, status_code: int, reason: str | None = None, preview: str = "", is_html: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.preview = preview
        self.is_html = is_html


class ProviderTimeoutError(ProviderError):
    """The provider stayed silent for longer than :class:`~zett_agent.model.TimeoutOptions` allows.

    Attributes:
        phase: Which wait ran out: ``"connect"``, ``"first_byte"`` (no response headers) or
            ``"idle"`` (a started response went quiet).
        seconds: The limit that was exceeded, or None when it is not known.
    """

    def __init__(self, message: str, *, phase: str, seconds: float | None = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.seconds = seconds


class ProviderRefusalError(ProviderResponseError):
    """A request was refused before the model produced any output.

    Adapters raise it from the code that opens the request and catch it in ``stream``,
    which turns it into a response whose stop is a refusal. A host never sees it unless it
    calls a private method of an adapter, so the refusal arrives as a message and an event
    like any other.

    Attributes:
        stop: The :class:`~zett_agent.stop.StopInfo` that describes the refusal.
    """

    def __init__(self, stop: StopInfo) -> None:
        super().__init__(stop.describe())
        self.stop = stop


def describe_status_error(provider: str, error: BaseException) -> tuple[int, HttpFailure, str]:
    """Read an SDK status error into its HTTP status, what the body was, and one sentence for a person.

    The SDKs keep the response on the error, so the body is available whether it was JSON, a
    line of text, or a web page. Nothing is raised if it cannot be read: a failure to describe
    a failure must not hide it.
    """
    status = int(getattr(error, "status_code", 0) or 0)
    response = getattr(error, "response", None)
    content_type: str | None = None
    text = ""
    if response is not None:
        try:
            content_type = response.headers.get("content-type")
            text = response.text
        except Exception:  # noqa: BLE001 - a body that cannot be read is simply not described
            text = ""
    body = getattr(error, "body", None)
    parsed = body if isinstance(body, (Mapping, str)) else None
    failure = read_failure(
        content_type=content_type, text=text or (body if isinstance(body, str) else ""), parsed=parsed
    )
    return status, failure, failure_message(provider, status, failure)


def http_error(status: int, failure: HttpFailure, message: str) -> ProviderHTTPError:
    """Build the typed error for one HTTP failure that has already been described."""
    return ProviderHTTPError(
        message, status_code=status, reason=failure.reason, preview=failure.preview, is_html=failure.is_html
    )


async def require_event_stream(stream: Any, provider: str) -> None:
    """Fail clearly when a streaming request was answered with a web page or JSON.

    A reverse proxy's error page, a captive portal, or a gateway that does not support streaming
    answers ``200`` with HTML or JSON. An SDK reading that as an event stream finds no events and
    reports an empty, successful answer, which hides the problem entirely. The content type is
    read as soon as the headers are in, and an HTML or JSON body is read (a bounded amount of
    it), described, and raised.

    Args:
        stream: The SDK stream the request returned; anything without a ``response`` is accepted.
        provider: How to name the provider in the message.

    Raises:
        ProviderHTTPError: If the response is HTML or JSON instead of a stream.
    """
    http = getattr(stream, "response", None)
    if http is None:
        return
    content_type = http.headers.get("content-type", "")
    kind = content_type.lower()
    # Only a web page or JSON is certainly not a stream. Some gateways send a real stream
    # as ``text/plain``, so anything else is read as one; a body with no events in it is
    # reported by the empty-stream check instead.
    if "html" not in kind and "json" not in kind:
        return
    body = b""
    try:
        async for chunk in http.aiter_bytes():
            body += chunk
            if len(body) > 65_536:
                break
    except Exception:  # noqa: BLE001 - what could be read is enough to describe the response
        pass
    close = getattr(stream, "close", None)
    if close is not None:
        try:
            await close()
        except Exception:  # noqa: BLE001 - the response is already being reported as an error
            pass
    status = int(http.status_code)
    failure = read_failure(content_type=content_type, text=body.decode("utf-8", errors="replace"))
    if failure.is_html:
        message = failure_message(provider, status, failure)
    else:
        detail = failure.reason or failure.preview
        kind = content_type.split(";")[0].strip()
        message = f"{provider} answered HTTP {status} with {kind} instead of an event stream" + (
            f": {detail}" if detail else ""
        )
    raise http_error(status, failure, message)


def httpx_timeout(options: TimeoutOptions) -> httpx.Timeout:
    """Return the transport-level timeouts for an HTTP client.

    Only the connection is limited here. How long to wait for the response headers and for the
    next chunk are enforced by :meth:`RetryingProvider._first_byte` and :func:`guard_idle`
    instead, because httpx has a single ``read`` limit for both and they need different values.
    """
    import httpx

    return httpx.Timeout(connect=options.connect, read=None, write=options.first_byte, pool=options.connect)


async def guard_idle(
    stream: Any,
    options: TimeoutOptions,
    *,
    first: float | None = None,
) -> AsyncIterator[Any]:
    """Yield what ``stream`` yields, giving up when it goes silent for longer than allowed.

    Every wait for the next item is limited by ``options.idle``. A stream whose request is only
    sent when the first item is asked for (the Google and Ollama clients) passes ``first`` as the
    limit for that first wait, which then covers connecting, the response headers and the first
    chunk. The wrapped stream is not closed here: whoever opened it still owns it.

    Raises:
        ProviderTimeoutError: If a wait exceeds its limit.
    """
    iterator = stream.__aiter__()
    limit: float | None = first if first is not None else options.idle
    phase = "first_byte" if first is not None else "idle"
    while True:
        try:
            item = await (iterator.__anext__() if limit is None else asyncio.wait_for(iterator.__anext__(), limit))
        except StopAsyncIteration:
            return
        except (TimeoutError, asyncio.TimeoutError):
            raise timeout_error(phase, limit) from None
        yield item
        limit, phase = options.idle, "idle"


def timeout_error(phase: str, seconds: float | None) -> ProviderTimeoutError:
    """Build the typed error for one wait that ran out, in words that say what it means."""
    shown = f"{seconds:g}" if seconds is not None else "the configured number of"
    match phase:
        case "connect":
            message = f"Could not connect to the provider within {shown} seconds."
        case "first_byte":
            message = (
                f"The provider sent no response within {shown} seconds of the request. "
                "It, or a gateway in front of it, may be down or overloaded."
            )
        case _:
            message = f"The provider stopped sending data for {shown} seconds; the response is incomplete."
    return ProviderTimeoutError(message, phase=phase, seconds=seconds)


def validate_extra_headers(value: Mapping[str, str] | None) -> dict[str, str]:
    """Copy extra HTTP headers, rejecting anything that is not a string pair.

    Providers send these on every request, after their own defaults, so a
    gateway can be given routing, tenant, or beta headers and an application can
    override a default such as ``User-Agent``. A copy is stored so later changes
    to the caller's mapping do not alter an adapter that is already in use.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("extra_headers must be a mapping of header names to values")
    headers: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("extra_headers names must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"extra_headers value for {name!r} must be a string")
        headers[name] = item
    return headers


def validate_extra_query(value: Mapping[str, str] | None) -> dict[str, str]:
    """Copy extra URL query parameters, sent on every request."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("extra_query must be a mapping of parameter names to values")
    query: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("extra_query names must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"extra_query value for {name!r} must be a string")
        query[name] = item
    return query


def validate_extra_body(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy extra JSON body fields merged into every request body.

    The fields are applied last, so they can add a vendor option the adapter
    does not model and also override one it does. They must be JSON values.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("extra_body must be a mapping of field names to JSON values")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError("extra_body names must be non-empty strings")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"extra_body must contain only JSON values: {error}") from error
    return dict(value)


def _reasoning_effort_to_budget(effort: ReasoningEffort) -> int:
    """Map provider-neutral effort to a token budget used by some vendors."""
    match effort:
        case ReasoningEffort.OFF:
            return 0
        case ReasoningEffort.MINIMAL:
            return 2048
        case ReasoningEffort.LOW:
            return 4096
        case ReasoningEffort.MEDIUM:
            return 8192
        case ReasoningEffort.HIGH:
            return 16384
        # Budget-based vendors saturate here; higher levels only exist as
        # native effort names on providers that expose them.
        case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return 32768


def _to_model_data(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=True)
    if hasattr(payload, "dict"):
        return payload.dict()
    if isinstance(payload, Message):
        data = asdict(payload)
        # The provider SDK may include extra fields that are not part of the public message schema.
        data.pop("attributes", None)
        data.pop("persist", None)
        data.pop("include_in_messages", None)
        return {"role": payload.role, **data}
    normalized = json.loads(json.dumps(payload, default=str))
    if not isinstance(normalized, dict):
        raise ProviderResponseError(f"Provider SDK value did not normalize to an object: {payload!r}")
    return normalized


def _normalize_image_source(
    part: ImageUrlSource | ImageBytesSource, detail: ImageDetail = ImageDetail.AUTO
) -> ChatCompletionContentPartImageParam:
    """Normalize a typed image source into an OpenAI-compatible content block."""
    match part:
        case ImageUrlSource(url=url):
            return {"type": "image_url", "image_url": {"url": url, "detail": detail.value}}
        case ImageBytesSource(data=data, media_type=media_type):
            encoded = base64.b64encode(data).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}", "detail": detail.value},
            }
        case _:
            raise ProviderResponseError(f"Unsupported image source type: {part!r}")


#: One chat message as this adapter sends it to an OpenAI-compatible endpoint. The SDK's own
#: ``ChatCompletionMessageParam`` is the strict OpenAI schema, and what is sent here is wider: other
#: providers take extra keys (DeepSeek's ``reasoning_content``) and multimodal tool results.
ChatMessagePayload = TypeAliasType("ChatMessagePayload", dict[str, Any])


def _message_to_openai_payload(message: AnyMessage) -> ChatMessagePayload:
    role = message.role
    match role:
        case "agent":
            return {"role": "user", "content": message.content}
        case "system":
            if not isinstance(message, SystemMessage):
                raise ProviderResponseError(f"Invalid system message data type: {type(message)!r}")
            return {"role": "system", "content": message.content}
        case "user":
            if not isinstance(message, UserMessage):
                raise ProviderResponseError(f"Invalid user message data type: {type(message)!r}")
            content = message.content
            match content:
                case str() as text:
                    return {"role": "user", "content": text}
                case []:
                    return {"role": "user", "content": ""}
                case list():
                    parts: list[ChatCompletionContentPartParam] = []
                    for part in content:
                        match part:
                            case TextContent(text=text):
                                parts.append({"type": "text", "text": text})
                            case ImageContent(source=source, detail=detail):
                                parts.append(_normalize_image_source(source, detail))
                            case _:
                                raise ProviderResponseError(f"Unsupported user content part: {part!r}")
                    return {"role": "user", "content": parts}
                case _:
                    raise ProviderResponseError(f"Unsupported user content type: {type(content)!r}")
        case "assistant":
            if not isinstance(message, AssistantMessage):
                raise ProviderResponseError(f"Invalid assistant message data type: {type(message)!r}")
            payload: ChatMessagePayload = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload
        case "tool":
            if not isinstance(message, ToolMessage):
                raise ProviderResponseError(f"Invalid tool message data type: {type(message)!r}")
            return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
        case _:
            raise ProviderResponseError(f"Unsupported role in message conversion: {role}")


def _replay_reasoning_content(
    source_messages: Sequence[AnyMessage],
    payload_messages: Sequence[ChatMessagePayload],
    *,
    provider_name: str,
) -> list[ChatMessagePayload]:
    """Replay same-provider reasoning on OpenAI-compatible assistant history.

    Reasoning-aware providers, including DeepSeek thinking mode, require the
    assistant ``reasoning_content`` to be sent back on later requests. It is
    model context that cannot be reconstructed from the final answer or tool
    calls. Messages without provider metadata remain compatible with history
    created before provider attribution was stored.

    See https://api-docs.deepseek.com/zh-cn/guides/thinking_mode.
    """
    rendered = [dict(payload) for payload in payload_messages]
    for source, target in zip(source_messages, rendered, strict=True):
        if not isinstance(source, AssistantMessage) or source.reasoning is None:
            continue
        if source.provider is not None and source.provider != provider_name:
            continue
        target["reasoning_content"] = source.reasoning
    return rendered


def _tools_to_openai_payload(tools: Sequence[ToolDefinition]) -> list[ChatCompletionToolParam]:
    reject_local_tool_search("Chat Completions", tools)
    rendered: list[ChatCompletionToolParam] = []
    for tool in tools:
        function: FunctionDefinition = {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        }
        rendered.append(
            {
                "type": "function",
                "function": function,
            }
        )
    return rendered


def reject_local_tool_search(protocol: str, tools: Sequence[ToolDefinition]) -> None:
    """Fail when a protocol cannot serve tools that stay out of the request."""
    if any(tool.local_tool_search for tool in tools):
        raise ValueError(f"{protocol} cannot answer client-side tool search; use the Responses API")


def _server_tools_to_payload(tools: Sequence[ServerToolDefinition]) -> list[Mapping[str, Any]]:
    """Render opaque server tools without treating them as local functions."""
    return [{"type": tool.type, **dict(tool.configuration)} for tool in tools]


@dataclass
class _ToolCallAccumulator:
    """Temporary state for accumulating fragmented provider tool-call payloads."""

    index: int
    call_id: str = ""
    name: str = ""
    argument_buffer: str = ""

    def append(self, delta: ToolCallDelta) -> None:
        self.call_id += delta.id_delta
        self.name += delta.name_delta
        self.argument_buffer += delta.arguments_delta


def _parse_tool_calls(
    streams: Mapping[int, _ToolCallAccumulator], finish_reason: str | None = None
) -> tuple[ToolCall, ...]:
    """Finish accumulated calls; undecodable arguments become model-visible failures."""
    calls: list[ToolCall] = []
    for index in sorted(streams):
        stream = streams[index]
        if not stream.call_id:
            stream.call_id = f"call_{index}"
        if not stream.name:
            raise ProviderResponseError(f"Tool call at index {index} did not provide a name")
        calls.append(
            tool_call_from_arguments(
                stream.call_id, stream.name, stream.argument_buffer, index=index, finish_reason=finish_reason
            )
        )
    return tuple(calls)


def _parse_tool_arguments(payload: str, *, index: int) -> dict[str, Any]:
    """Decode one argument object, raising ProviderResponseError when it is unusable.

    See :func:`~zett_agent.providers.tool_arguments.decode_tool_arguments` for
    the repair rules. Adapters that can keep going after a bad call use
    :func:`~zett_agent.providers.tool_arguments.tool_call_from_arguments`.
    """
    try:
        return decode_tool_arguments(payload, index=index)
    except ToolArgumentsError as error:
        raise ProviderResponseError(str(error)) from error


#: Cache counters accepted from OpenAI-compatible payloads. Gateways regularly
#: relay one protocol's payload through another, so the same number appears
#: under several spellings; every one of them is read.
_CACHE_READ_KEYS = ("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens")
_CACHE_WRITE_KEYS = ("cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens")
_CACHE_DETAIL_KEYS = ("prompt_tokens_details", "input_tokens_details")


def _cache_counters(payload: Mapping[str, Any]) -> tuple[int, int, bool]:
    """Read cache counters from a usage mapping in any compatible spelling.

    Nested detail objects are applied after the top level so a detailed payload
    wins over a flattened copy of itself, matching the OpenAI schema where
    ``prompt_tokens_details`` is authoritative.
    """
    sources: list[Mapping[str, Any]] = [payload]
    sources.extend(details for key in _CACHE_DETAIL_KEYS if isinstance(details := payload.get(key), Mapping))
    read_tokens = 0
    write_tokens = 0
    reported = False
    for source in sources:
        for key in _CACHE_READ_KEYS:
            if key in source:
                read_tokens = int(source[key] or 0)
                reported = True
        for key in _CACHE_WRITE_KEYS:
            if key in source:
                write_tokens = int(source[key] or 0)
                reported = True
    return read_tokens, write_tokens, reported


def _usage_from_mapping(payload: Mapping[str, Any]) -> ModelUsage:
    """Normalize token usage returned by OpenAI-compatible chat APIs.

    This helper is shared by :class:`OpenAIProvider` and
    :class:`DeepSeekProvider`. Their response fields map as follows::

        ModelUsage field     Accepted API fields
        -------------------  -------------------------------------------------
        input_tokens         prompt_tokens, input_tokens
        output_tokens        completion_tokens, output_tokens
        cache_read_tokens    prompt_cache_hit_tokens, cache_read_input_tokens,
                             cached_tokens, prompt_tokens_details.cached_tokens,
                             input_tokens_details.cached_tokens
        cache_write_tokens   cache_write_tokens, cache_creation_input_tokens,
                             cache_creation_tokens,
                             prompt_tokens_details.cache_write_tokens,
                             input_tokens_details.cache_write_tokens
        reasoning_tokens     completion_tokens_details.reasoning_tokens,
                             output_tokens_details.reasoning_tokens

    The aliases exist because gateways front one protocol with another and
    return whichever usage schema their upstream speaks. Reading all of them
    keeps a relay from silently reporting zero cache reads.

    ``prompt_tokens`` includes cached input tokens. For DeepSeek it is the sum
    of ``prompt_cache_hit_tokens`` and ``prompt_cache_miss_tokens``; the miss
    count therefore has no separate destination in :class:`ModelUsage`.
    Likewise, ``completion_tokens`` includes reasoning tokens rather than being
    added to them. Both providers return ``total_tokens``, but ``ModelUsage``
    derives that total from input plus output to preserve one internal invariant.

    A payload that carries no cache counter at all leaves ``cache_reported``
    False, so :attr:`~zett_agent.model.ModelUsage.cache_hit_rate` reports
    unknown instead of a false zero. Anthropic reports ``input_tokens``,
    ``cache_read_input_tokens``, and ``cache_creation_input_tokens`` separately.
    Gemini reports ``prompt_token_count``, ``candidates_token_count``,
    ``cached_content_token_count``, and ``thoughts_token_count``. Ollama reports
    ``prompt_eval_count`` and ``eval_count``. Those provider-specific schemas
    are normalized in their own adapters rather than by this helper.
    """
    cache_read_tokens, cache_write_tokens, cache_reported = _cache_counters(payload)

    input_tokens = payload.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("completion_tokens")
    if output_tokens is None:
        output_tokens = payload.get("output_tokens")

    reasoning_tokens = 0
    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = payload.get(details_key)
        if isinstance(details, Mapping):
            reasoning_tokens = int(details.get("reasoning_tokens", reasoning_tokens) or 0)

    return ModelUsage(
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_reported=cache_reported,
        reasoning_tokens=reasoning_tokens,
    )


class RetryingProvider(SyncMethodsMixin):
    """Shared retry configuration and transient-error classification.

    retry.max_retries counts additional attempts (2 means at most 3 requests). Only transport
    failures and HTTP 408/409/429/5xx are retried. Once any model event is yielded,
    propagate failures instead of replaying already-visible text or tool deltas.
    Cancellation also propagates immediately, including during backoff.

    Concrete providers expose their normal ``stream()`` method and apply
    ``@retry_model_stream`` to it. SDK retries remain disabled by each adapter,
    preventing nested retry multiplication.
    """

    retry: RetryOptions = DEFAULT_RETRY_OPTIONS
    timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS
    #: Whether this adapter uses a Responses-style endpoint instead of its provider's traditional
    #: message or chat-completions endpoint. Only the providers read it; the runtime does not.
    response: bool = False
    #: See :attr:`zett_agent.model.NativeDeferredToolsModel.native_deferred_tools`. A provider
    #: opts in by overriding this on its class; the default keeps Responses,
    #: Chat Completions, Google, and Ollama on the withhold-and-search path.
    native_deferred_tools: bool = False

    async def _first_byte(self, opening: Any) -> Any:
        """Await ``opening`` (the call that returns once the response headers are in) with its limit.

        Raises:
            ProviderTimeoutError: If the headers do not arrive within ``timeout.first_byte``.
        """
        limit = self.timeout.first_byte
        if limit is None:
            return await opening
        try:
            async with async_timeout(limit):
                return await opening
        except (TimeoutError, asyncio.TimeoutError):
            raise timeout_error("first_byte", limit) from None

    def _normalized(self, error: BaseException) -> BaseException:
        """Return a typed timeout error for an SDK timeout that leaked out, else the error itself.

        The SDKs turn an expired connection or read into their own exception with the httpx one
        on ``__cause__``. Recognizing it here, once, gives every adapter the same typed error.
        """
        import httpx

        seen: set[int] = set()
        cause: BaseException | None = error
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, ProviderTimeoutError):
                return cause
            if isinstance(cause, (httpx.ConnectTimeout, httpx.PoolTimeout)):
                return timeout_error("connect", self.timeout.connect)
            if isinstance(cause, httpx.TimeoutException):
                return timeout_error("idle", self.timeout.idle)
            cause = cause.__cause__
        return error

    @staticmethod
    def _retryable(error: BaseException) -> bool:
        """Inspect SDK status codes and wrapped transport causes without optional imports.

        A timeout is retried only when it was the connection that timed out: that costs a few
        seconds. A silent provider is not asked again, because that waits the whole limit a second
        time and the first request may still be running, and billed, on the other side.
        """
        import httpx

        seen: set[int] = set()
        while id(error) not in seen:
            seen.add(id(error))
            if isinstance(error, ProviderTimeoutError):
                return error.phase == "connect"
            if isinstance(error, httpx.ConnectTimeout):
                return True
            if isinstance(error, httpx.TimeoutException):
                return False
            if isinstance(error, ResponsesStreamInterrupted):
                return True
            status = getattr(error, "status_code", None) or getattr(error, "code", None)
            if isinstance(status, int):
                return status in (408, 409, 429) or 500 <= status < 600
            if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError)):
                return True
            if error.__cause__ is None:
                return False
            error = error.__cause__
        return False


ProviderT = TypeVar("ProviderT", bound=RetryingProvider)


def retry_model_stream(
    stream: Callable[[ProviderT, ModelRequest], AsyncIterator[ModelEvent]],
) -> Callable[[ProviderT, ModelRequest], AsyncIterator[ModelEvent]]:
    """Apply model-owned exponential backoff to a provider stream method.

    The decorated provider method remains publicly named ``stream``. A failed
    attempt is restarted only before its first event reaches the caller.
    """

    @wraps(stream)
    async def retrying_stream(self: ProviderT, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        retry = self.retry
        validate_retry(retry)
        delay = min(retry.base_delay, retry.max_delay)
        emitted = False
        for attempt in range(retry.max_retries + 1):
            try:
                async with aclosing_stream(stream(self, request)) as events:
                    async for event in events:
                        emitted = True
                        yield event
                return
            except Exception as raised:
                error = self._normalized(raised)
                if emitted or attempt == retry.max_retries or not self._retryable(error):
                    if error is raised:
                        raise
                    raise error from raised
                await asyncio.sleep(delay)
                # Saturate before doubling to avoid overflow for large delays.
                delay = retry.max_delay if delay >= retry.max_delay / 2 else delay * 2

    return retrying_stream


def _empty_stream_message(provider: str, stream: Any) -> str:
    """Say that a response that looked like a stream contained nothing, and what it did say."""
    http = getattr(stream, "response", None)
    status = getattr(http, "status_code", None)
    content_type = (http.headers.get("content-type", "") if http is not None else "") or "no content type"
    return (
        f"{provider} ended the stream without sending any data (HTTP {status}, {content_type}). "
        "A proxy or gateway in front of the API may have cut the response off, or the endpoint does not stream."
    )


def _stream_error_body(error: BaseException) -> dict[str, Any]:
    """Return the ``code``, ``message`` and ``type`` of an error the SDK raised from a stream."""
    body = getattr(error, "body", None)
    kept = {key: body[key] for key in ("code", "message", "type") if key in body} if isinstance(body, Mapping) else {}
    if "message" not in kept and (message := getattr(error, "message", None)):
        kept["message"] = str(message)
    return kept


def _stream_error_message(error: BaseException) -> str:
    """Return one line that says what a stream failed with."""
    body = _stream_error_body(error)
    return str(body.get("message") or error)


def _chat_stop(
    finish_reason: str | None,
    *,
    has_tool_calls: bool,
    refusal_text: str,
    stream_error: Mapping[str, Any],
    native_finish_reason: str | None,
) -> StopInfo:
    """Build the stop of one Chat Completions stream from everything it said about its end.

    The ``finish_reason`` is the base. A model that wrote a refusal (OpenAI's ``refusal``
    field) finished with ``stop``, so the refusal text, not the reason, decides that it was
    declined. A chunk-level ``error`` means the stream failed even if it names no reason, so
    it makes the stop an interruption unless the reason already says something stronger.
    """
    detail: dict[str, Any] = {}
    if native_finish_reason is not None and native_finish_reason != finish_reason:
        detail["native_finish_reason"] = native_finish_reason
    if stream_error:
        detail["error"] = {key: stream_error[key] for key in ("code", "message", "type") if key in stream_error}
    raw = finish_reason
    if raw is None and stream_error:
        raw = "error"
    stop = classify_stop(raw, has_tool_calls=has_tool_calls, detail=detail)
    if refusal_text and stop.ok:
        return StopInfo(
            StopReason.REFUSAL,
            raw=raw,
            category="refusal",
            message=refusal_text,
            detail=stop.detail,
        )
    if refusal_text and stop.message is None:
        return StopInfo(stop.reason, raw=stop.raw, category=stop.category, message=refusal_text, detail=stop.detail)
    if stream_error and stop.message is None:
        message = stream_error.get("message")
        if message:
            return StopInfo(stop.reason, raw=stop.raw, category=stop.category, message=str(message), detail=stop.detail)
    return stop


class _OpenAIStyleProvider(RetryingProvider):
    """Shared implementation for providers exposing OpenAI-style streaming chunks."""

    provider_name: str = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        temperature: float | None = None,
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        send_prompt_cache_key: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
    ) -> None:
        # HTTP clients and the trust store are only needed once a provider is
        # constructed, so importing them here keeps module import cheap.
        import httpx
        import truststore

        validate_retry(retry)
        validate_timeout(timeout)
        validate_response(response)
        self.retry = retry
        self.timeout = timeout
        self.send_prompt_cache_key = send_prompt_cache_key
        self.extra_headers = validate_extra_headers(extra_headers)
        self.extra_query = validate_extra_query(extra_query)
        self.extra_body = validate_extra_body(extra_body)
        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self.temperature = temperature
        self.response = response
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client,
            max_retries=0,
            timeout=httpx_timeout(timeout),
            default_headers=self.extra_headers or None,
            default_query=self.extra_query or None,
        )

    async def _request(self, request: ModelRequest) -> Any:
        if self.response:
            return await self._request_response(request)
        source_messages = expand_tool_images(request.messages)
        messages = _replay_reasoning_content(
            source_messages,
            [_message_to_openai_payload(message) for message in source_messages],
            provider_name=self.provider_name,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [
                *_tools_to_openai_payload(request.tools),
                *_server_tools_to_payload(request.server_tools),
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload.update(self._provider_specific_request_fields(request))
        if not request.tools and not request.server_tools:
            payload.pop("tools")
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if request.tool_choice:
            payload["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}
        if self.send_prompt_cache_key and request.cache_key:
            payload["prompt_cache_key"] = request.cache_key
        extra_body = {**self._provider_specific_request_extra_fields(request), **self.extra_body}
        if extra_body:
            payload["extra_body"] = extra_body
        from openai import APIStatusError

        try:
            return await self._first_byte(self._client.chat.completions.create(**payload))
        except APIStatusError as error:
            status_code = error.status_code
            refusal = refusal_from_status_error(status_code, error.body)
            if refusal is not None:
                raise ProviderRefusalError(refusal) from error
            status, failure, message = describe_status_error(self.provider_name, error)
            if status == 401:
                raise ProviderAuthError(f"Provider rejected API credential. {message}") from error
            raise http_error(status, failure, message) from error

    async def _request_response(self, request: ModelRequest) -> Any:
        """Open a stateless Responses API stream for OpenAI or DeepSeek."""
        tools = responses_tools(request.tools, request.server_tools)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": responses_input(
                request.messages,
                provider=self.provider_name,
                model=self.model,
                local_search_tool=local_tool_search_name(request.tools),
            ),
            "stream": True,
            "store": False,
            "parallel_tool_calls": request.parallel_tool_call,
        }
        if tools:
            payload["tools"] = tools
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if request.tool_choice:
            server_types = {tool.type for tool in request.server_tools}
            payload["tool_choice"] = (
                {"type": request.tool_choice}
                if request.tool_choice in server_types
                else {"type": "function", "name": request.tool_choice}
            )
        reasoning = responses_reasoning(request.reasoning_effort)
        if reasoning is not None:
            payload["reasoning"] = reasoning
            # Stateless reasoning models only hand back the opaque reasoning
            # state that continues a chain of thought when it is requested, and
            # the replayed reasoning items are worthless without it.
            payload["include"] = ["reasoning.encrypted_content"]
        if self.send_prompt_cache_key and request.cache_key:
            payload["prompt_cache_key"] = request.cache_key
        if self.extra_body:
            payload["extra_body"] = dict(self.extra_body)
        from openai import APIStatusError

        try:
            return await self._first_byte(self._client.responses.create(**payload))
        except APIStatusError as error:
            refusal = refusal_from_status_error(error.status_code, error.body)
            if refusal is not None:
                raise ProviderRefusalError(refusal) from error
            status, failure, message = describe_status_error(self.provider_name, error)
            if status == 401:
                raise ProviderAuthError(f"Provider rejected API credential. {message}") from error
            raise http_error(status, failure, message) from error

    def _provider_specific_request_fields(self, request: ModelRequest) -> dict[str, Any]:
        return (
            {}
            if request.reasoning_effort == ReasoningEffort.OFF
            else {"reasoning_effort": request.reasoning_effort.value}
        )

    async def aclose(self) -> None:
        """Close the SDK's owned HTTP connection pool."""
        await self._client.close()

    def _provider_specific_request_extra_fields(self, request: ModelRequest) -> dict[str, Any]:
        return {}

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        local_search_tool = local_tool_search_name(request.tools)
        from openai import APIConnectionError, APITimeoutError
        from openai import APIError as OpenAIError

        try:
            response = await self._request(request)
        except ProviderRefusalError as refusal:
            # The request was refused before any output. It is answered as a refusal, not
            # raised, so the conversation goes on and the host is told why.
            yield ModelEvent.completed(
                ModelResponse(
                    AssistantMessage(content="", provider=self.provider_name, model=self.model),
                    finish_reason=refusal.stop.raw,
                    stop=refusal.stop,
                )
            )
            return
        await require_event_stream(response, self.provider_name)
        if self.response:
            try:
                async for event in stream_responses(
                    guard_idle(response, self.timeout),
                    provider=self.provider_name,
                    model=self.model,
                    local_search_tool=local_search_tool,
                ):
                    yield event
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ProviderResponseError(str(error)) from error
            finally:
                await response.close()
            return

        text = ""
        reasoning = ""
        streams: dict[int, _ToolCallAccumulator] = {}
        finish_reason = None
        refusal_text = ""
        stream_error: Mapping[str, Any] = {}
        native_finish_reason: str | None = None
        usage = ModelUsage()

        chunks = 0
        try:
            async for chunk in guard_idle(response, self.timeout):
                if chunk is None:
                    continue
                chunks += 1
                match getattr(chunk, "usage", None):
                    case None:
                        pass
                    case usage_payload:
                        raw = _to_model_data(usage_payload)
                        usage = _usage_from_mapping(raw)

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                if isinstance(native := getattr(choices[0], "native_finish_reason", None), str):
                    native_finish_reason = native
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    if getattr(choices[0], "finish_reason", None):
                        finish_reason = choices[0].finish_reason
                    continue

                refusal_fragment = getattr(delta, "refusal", None)
                if isinstance(refusal_fragment, str) and refusal_fragment:
                    # A refusal is the model's answer to the user, so it is shown like any
                    # other text, and it is also what the stop explains itself with.
                    refusal_text += refusal_fragment
                    text += refusal_fragment
                    yield ModelEvent.text(refusal_fragment)

                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    text += content
                    yield ModelEvent.text(content)

                reasoning_fragment = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_fragment, str) and reasoning_fragment:
                    reasoning += reasoning_fragment
                    yield ModelEvent.reasoning(reasoning_fragment)

                for call in getattr(delta, "tool_calls", ()) or ():
                    call_payload = ToolCallDelta(
                        index=int(getattr(call, "index", 0)),
                        id_delta=getattr(call, "id", "") or "",
                        name_delta=getattr(getattr(call, "function", None), "name", "") or "",
                        arguments_delta=getattr(getattr(call, "function", None), "arguments", "") or "",
                    )
                    stream = streams.setdefault(call_payload.index, _ToolCallAccumulator(index=call_payload.index))
                    stream.append(call_payload)
                    yield ModelEvent.tool_call(call_payload)

                match getattr(choices[0], "finish_reason", None):
                    case str(reason):
                        finish_reason = reason
                    case _:
                        pass
        except OpenAIError as error:
            # A gateway reports a failure inside a 200 stream as a chunk with a top-level
            # ``error`` (OpenRouter does, with ``finish_reason: "error"``), and the SDK turns
            # that chunk into this exception before the loop above sees it. Before any output
            # it is raised as a typed error, so a transient failure is retried; after output
            # the response ends here as an interruption and keeps what was written.
            if isinstance(error, (APIConnectionError, APITimeoutError)):
                raise  # a dropped or timed-out connection is not an error the provider reported
            if not (text or reasoning or streams):
                raise ProviderResponseError(_stream_error_message(error)) from error
            stream_error = _stream_error_body(error)
        finally:
            await response.close()

        if not chunks:
            raise ProviderResponseError(_empty_stream_message(self.provider_name, response))

        tool_calls = _parse_tool_calls(streams, finish_reason)
        response_message = AssistantMessage(
            content=text,
            reasoning=reasoning or None,
            tool_calls=tool_calls,
            provider=self.provider_name,
            model=self.model,
        )
        yield ModelEvent.completed(
            ModelResponse(
                message=response_message,
                finish_reason=finish_reason,
                usage=usage,
                stop=_chat_stop(
                    finish_reason,
                    has_tool_calls=bool(tool_calls),
                    refusal_text=refusal_text,
                    stream_error=stream_error,
                    native_finish_reason=native_finish_reason,
                ),
            )
        )
