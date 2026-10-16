from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Protocol, TypeVar

from .._compat import aclosing_stream
from .._compat import timeout as async_timeout
from ..exceptions import AgentError
from ..json_types import JsonValue
from ..messages import ToolCall
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    ModelEvent,
    ModelRequest,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    TimeoutOptions,
    ToolCallDelta,
    ToolDefinition,
    validate_retry,
)

if TYPE_CHECKING:
    import httpx

from ..stop import StopInfo
from ..sync_runtime import SyncMethodsMixin
from .http_errors import HttpFailure, failure_message, read_failure
from .responses import ResponsesStreamInterrupted
from .tool_arguments import tool_call_from_arguments

T = TypeVar("T")


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


class SDKStream(Protocol):
    """What a provider SDK's streaming reply exposes: the HTTP response it is reading, and a way to close it.

    Both the OpenAI and the Anthropic SDK return an object of this shape. A stream built by hand
    (a test double, a custom client) may have no response, which is why it can be None.
    """

    @property
    def response(self) -> httpx.Response | None:
        """The HTTP response being read, or None when the stream is not backed by one."""
        ...

    async def close(self) -> None:
        """Release the connection."""
        ...


async def require_event_stream(stream: SDKStream, provider: str) -> None:
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
    stream: AsyncIterable[T],
    options: TimeoutOptions,
    *,
    first: float | None = None,
) -> AsyncIterator[T]:
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


def validate_extra_body(value: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
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


def reasoning_effort_to_budget(effort: ReasoningEffort) -> int:
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


def reject_local_tool_search(protocol: str, tools: Sequence[ToolDefinition]) -> None:
    """Fail when a protocol cannot serve tools that stay out of the request."""
    if any(tool.local_tool_search for tool in tools):
        raise ValueError(f"{protocol} cannot answer client-side tool search; use the Responses API")


@dataclass
class ToolCallAccumulator:
    """Temporary state for accumulating fragmented provider tool-call payloads."""

    index: int
    call_id: str = ""
    name: str = ""
    argument_buffer: str = ""

    def append(self, delta: ToolCallDelta) -> None:
        self.call_id += delta.id_delta
        self.name += delta.name_delta
        self.argument_buffer += delta.arguments_delta


def parse_tool_calls(
    streams: Mapping[int, ToolCallAccumulator], finish_reason: str | None = None
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


#: Cache counters accepted from OpenAI-compatible payloads. Gateways regularly
#: relay one protocol's payload through another, so the same number appears
#: under several spellings; every one of them is read.
_CACHE_READ_KEYS = ("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens")
_CACHE_WRITE_KEYS = ("cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens")
_CACHE_DETAIL_KEYS = ("prompt_tokens_details", "input_tokens_details")


def _count(value: JsonValue | None) -> int:
    """Read a token count that a service may spell as an integer, a float, or a numeric string."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _cache_counters(payload: Mapping[str, JsonValue]) -> tuple[int, int, bool]:
    """Read cache counters from a usage mapping in any compatible spelling.

    Nested detail objects are applied after the top level so a detailed payload
    wins over a flattened copy of itself, matching the OpenAI schema where
    ``prompt_tokens_details`` is authoritative.
    """
    sources: list[Mapping[str, JsonValue]] = [payload]
    sources.extend(details for key in _CACHE_DETAIL_KEYS if isinstance(details := payload.get(key), Mapping))
    read_tokens = 0
    write_tokens = 0
    reported = False
    for source in sources:
        for key in _CACHE_READ_KEYS:
            if key in source:
                read_tokens = _count(source[key])
                reported = True
        for key in _CACHE_WRITE_KEYS:
            if key in source:
                write_tokens = _count(source[key])
                reported = True
    return read_tokens, write_tokens, reported


def usage_from_mapping(payload: Mapping[str, JsonValue]) -> ModelUsage:
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
            reasoning_tokens = _count(details.get("reasoning_tokens", reasoning_tokens))

    return ModelUsage(
        input_tokens=_count(input_tokens),
        output_tokens=_count(output_tokens),
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

    async def _first_byte(self, opening: Awaitable[T]) -> T:
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


def empty_stream_message(provider: str, stream: SDKStream) -> str:
    """Say that a response that looked like a stream contained nothing, and what it did say."""
    http = getattr(stream, "response", None)
    status = getattr(http, "status_code", None)
    content_type = (http.headers.get("content-type", "") if http is not None else "") or "no content type"
    return (
        f"{provider} ended the stream without sending any data (HTTP {status}, {content_type}). "
        "A proxy or gateway in front of the API may have cut the response off, or the endpoint does not stream."
    )
