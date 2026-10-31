"""The Chat Completions wire format, and the base class of every provider that speaks it.

Many services answer the OpenAI Chat Completions protocol, and each of them bends it a little:
the field that carries reasoning is called ``reasoning_content`` by one and ``reasoning`` by
another, a switch that turns thinking off lives in a different place, a gateway wants identifying
headers, a chunk carries an extra block that must be sent back unchanged. Handling all of that in
one class turns it into a pile of ``if provider == ...``. This module splits the work instead::

    OpenAICompatibleProvider            one request, one stream, the shared protocol
        |
        |  hooks (all optional, all public)
        |    vendor_headers()        headers the service wants on every request
        |    request_fields()        body fields for this request (thinking, reasoning, ...)
        |    responses_fields()      body fields that refine a Responses request
        |    replay_reasoning()      what to send back of an earlier assistant turn
        |    read_delta()            what one streamed delta says
        |    merge_replay()          how streamed opaque blocks are assembled
        |    read_usage()            token counts, including vendor spellings
        |    chat_stop()             why the response ended
        |    status_error()          what an HTTP failure means
        v
    OpenAIProvider  DeepSeekProvider  OpenRouterProvider  KimiProvider  GLMProvider

A new service is a subclass that overrides only the hooks where it differs. Everything it does not
override is the plain protocol, so any endpoint that has no class of its own works through
:class:`~zett_agent.providers.openai.OpenAIProvider` with a ``base_url``.

Payloads are typed with :data:`~zett_agent.json_types.JsonValue` rather than ``Any``: what goes to
and comes from the wire is JSON, and a hook that returns something else is a type error.
"""

from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from .._compat import aclosing_stream
from ..json_types import JsonValue
from ..messages import (
    AnyMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageDetail,
    ImageUrlSource,
    SystemMessage,
    TextContent,
    ToolMessage,
    UserMessage,
)
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    CacheMode,
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
from ..stop import StopInfo, StopReason, classify_stop
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderRefusalError,
    ProviderResponseError,
    RetryingProvider,
    ToolCallAccumulator,
    describe_status_error,
    empty_stream_message,
    guard_idle,
    http_error,
    httpx_timeout,
    parse_tool_calls,
    reject_local_tool_search,
    require_event_stream,
    retry_model_stream,
    usage_from_mapping,
    validate_extra_body,
    validate_extra_headers,
    validate_extra_query,
)
from .responses import local_tool_search_name, responses_request, stream_responses
from .stop_reasons import refusal_from_status_error
from .tool_images import expand_tool_images
from .wire import JsonObject, read_object, read_objects, read_str

if TYPE_CHECKING:
    import httpx
    from openai import APIStatusError, AsyncOpenAI
    from openai._streaming import AsyncStream
    from openai.types.chat import ChatCompletionChunk
    from openai.types.responses import ResponseStreamEvent

__all__ = [
    "ChatDelta",
    "ChatMessagePayload",
    "OpenAICompatibleProvider",
    "chat_image",
    "chat_message",
    "chat_server_tools",
    "chat_tools",
    "low_medium_high",
    "tool_call_deltas",
]

#: One chat message as this adapter sends it. The SDK's own ``ChatCompletionMessageParam`` is the
#: strict OpenAI schema; what is sent is wider, because compatible services take extra keys such
#: as ``reasoning_content`` or ``reasoning_details``.
ChatMessagePayload = JsonObject


# --- the wire format ---------------------------------------------------------------------------


def chat_image(part: ImageUrlSource | ImageBytesSource, detail: ImageDetail = ImageDetail.AUTO) -> JsonObject:
    """Render an image source as a Chat Completions ``image_url`` content part."""
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


def chat_message(message: AnyMessage) -> ChatMessagePayload:
    """Render one message as a Chat Completions message, without any vendor extension."""
    match message:
        case UserMessage(content=str() as text):
            return {"role": "user", "content": text}
        case UserMessage(content=[]):
            return {"role": "user", "content": ""}
        case UserMessage(content=list() as content):
            parts: list[JsonValue] = []
            for part in content:
                match part:
                    case TextContent(text=text):
                        parts.append({"type": "text", "text": text})
                    case ImageContent(source=source, detail=detail):
                        parts.append(chat_image(source, detail))
                    case _:
                        raise ProviderResponseError(f"Unsupported user content part: {part!r}")
            return {"role": "user", "content": parts}
        case UserMessage(content=content):
            raise ProviderResponseError(f"Unsupported user content type: {type(content)!r}")
        case SystemMessage(content=text):
            return {"role": "system", "content": text}
        case AssistantMessage():
            payload: ChatMessagePayload = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                # The protocol allows no text beside a tool call, as ``null``; some gateways refuse an
                # empty string there.
                if not message.content:
                    payload["content"] = None
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
        case ToolMessage():
            content = message.content if isinstance(message.content, str) else message.text
            return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}
        case _ if message.role == "agent":
            return {"role": "user", "content": message.content}
        case _:
            raise ProviderResponseError(f"Unsupported role in message conversion: {message.role}")


def chat_tools(tools: Sequence[ToolDefinition]) -> list[JsonObject]:
    """Render function tools in Chat Completions format."""
    reject_local_tool_search("Chat Completions", tools)
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": cast(JsonObject, dict(tool.parameters)),
            },
        }
        for tool in tools
    ]


def chat_server_tools(tools: Sequence[ServerToolDefinition]) -> list[JsonObject]:
    """Render opaque server tools without treating them as local functions."""
    return [{"type": tool.type, **cast(JsonObject, dict(tool.configuration))} for tool in tools]


def low_medium_high(effort: ReasoningEffort) -> str:
    """Map a provider-neutral level onto ``low``, ``medium`` and ``high``, the names many services share.

    ``OFF`` and ``MINIMAL`` share ``low``, and every level above ``HIGH`` shares ``high``. A service
    that can switch reasoning off spells that itself.
    """
    match effort:
        case ReasoningEffort.OFF | ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
            return "low"
        case ReasoningEffort.MEDIUM:
            return "medium"
        case _:
            return "high"


# --- reading a stream ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatDelta:
    """What one streamed ``delta`` says, in the runtime's terms.

    :meth:`OpenAICompatibleProvider.read_delta` returns one for every chunk, and a subclass that
    reads a vendor's extra fields returns the same shape, so the stream loop never learns what the
    vendor calls anything.
    """

    #: Answer text to show.
    text: str = ""
    #: Reasoning text to show.
    reasoning: str = ""
    #: Refusal text; shown like ``text`` and also what explains a refusal stop.
    refusal: str = ""
    #: Fragments of tool calls.
    tool_calls: tuple[ToolCallDelta, ...] = ()
    #: Opaque blocks the vendor wants sent back unchanged with the finished message.
    replay: tuple[JsonObject, ...] = ()


def tool_call_deltas(value: JsonValue | None) -> tuple[ToolCallDelta, ...]:
    """Read the ``tool_calls`` array of a streamed delta."""
    deltas: list[ToolCallDelta] = []
    for call in read_objects(value):
        function = read_object(call.get("function"))
        index = call.get("index")
        deltas.append(
            ToolCallDelta(
                index=index if isinstance(index, int) else 0,
                id_delta=read_str(call.get("id")),
                name_delta=read_str(function.get("name")),
                arguments_delta=read_str(function.get("arguments")),
            )
        )
    return tuple(deltas)


def chunk_data(chunk: ChatCompletionChunk) -> JsonObject:
    """Return a chunk as plain JSON, extra vendor fields included.

    The SDK's models keep fields they do not declare, so a vendor's ``reasoning`` or
    ``native_finish_reason`` is in the dump even though the SDK has never heard of it.
    """
    return cast(JsonObject, chunk.model_dump(exclude_none=True, warnings=False))


def _stream_error_body(error: BaseException) -> dict[str, JsonValue]:
    """Return the ``code``, ``message`` and ``type`` of an error the SDK raised from a stream."""
    body = getattr(error, "body", None)
    kept: dict[str, JsonValue] = (
        {key: body[key] for key in ("code", "message", "type") if key in body} if isinstance(body, dict) else {}
    )
    if "message" not in kept and (message := getattr(error, "message", None)):
        kept["message"] = str(message)
    return kept


def _stream_error_message(error: BaseException) -> str:
    """Return one line that says what a stream failed with."""
    return str(_stream_error_body(error).get("message") or error)


def default_chat_stop(
    finish_reason: str | None,
    *,
    has_tool_calls: bool,
    refusal_text: str,
    stream_error: Mapping[str, JsonValue],
    native_finish_reason: str | None,
) -> StopInfo:
    """Build the stop of one Chat Completions stream from everything it said about its end.

    The ``finish_reason`` is the base. A model that wrote a refusal (OpenAI's ``refusal``
    field) finished with ``stop``, so the refusal text, not the reason, decides that it was
    declined. A chunk-level ``error`` means the stream failed even if it names no reason, so
    it makes the stop an interruption unless the reason already says something stronger.
    """
    detail: dict[str, JsonValue] = {}
    if native_finish_reason is not None and native_finish_reason != finish_reason:
        detail["native_finish_reason"] = native_finish_reason
    if stream_error:
        detail["error"] = {key: stream_error[key] for key in ("code", "message", "type") if key in stream_error}
    raw = finish_reason
    if raw is None and stream_error:
        raw = "error"
    stop = classify_stop(raw, has_tool_calls=has_tool_calls, detail=detail)
    if refusal_text and stop.ok:
        return StopInfo(StopReason.REFUSAL, raw=raw, category="refusal", message=refusal_text, detail=stop.detail)
    if refusal_text and stop.message is None:
        return StopInfo(stop.reason, raw=stop.raw, category=stop.category, message=refusal_text, detail=stop.detail)
    if stream_error and stop.message is None:
        message = stream_error.get("message")
        if message:
            return StopInfo(stop.reason, raw=stop.raw, category=stop.category, message=str(message), detail=stop.detail)
    return stop


# --- the provider ---------------------------------------------------------------------------------


class OpenAICompatibleProvider(RetryingProvider):
    """Stream a service that speaks the OpenAI Chat Completions protocol.

    This is the base class of :class:`~zett_agent.providers.openai.OpenAIProvider` and the
    providers for the services that bend the protocol. A provider for another one is a subclass
    that sets :attr:`provider_name` and overrides the hooks below where the service differs; it
    does not copy the stream loop.

    The hooks are ordinary methods and never need ``super()`` to keep the protocol working: each
    default is the plain OpenAI behavior, and a subclass that wants to add to it calls
    ``super()`` itself.

    * :meth:`vendor_headers` -- headers sent on every request.
    * :meth:`request_fields` -- JSON fields of one request.
    * :meth:`responses_fields` -- JSON fields that refine a Responses request.
    * :meth:`replay_reasoning` -- the vendor extension of an earlier assistant message.
    * :meth:`read_delta` -- one streamed delta.
    * :meth:`merge_replay` -- how streamed opaque blocks are put together.
    * :meth:`read_usage` -- token counts.
    * :meth:`chat_stop` -- why the response ended.
    * :meth:`status_error` -- what an HTTP failure means.
    """

    #: Name stored on messages this provider produced and compared before reasoning is replayed.
    provider_name: str = "openai"
    #: Fields of a streamed delta that carry reasoning text, in order of preference. Services
    #: differ on the name (``reasoning_content`` and ``reasoning`` are the two in use), and reading
    #: both is harmless because only the first one that has text is used.
    reasoning_keys: ClassVar[tuple[str, ...]] = ("reasoning_content", "reasoning")
    #: API root used when no ``base_url`` is given. A subclass for one service sets it; the generic
    #: class has none, so it requires ``base_url``.
    default_base_url: ClassVar[str | None] = None
    #: Whether the run's prompt cache key is sent when ``send_prompt_cache_key`` is not given. A
    #: service that rejects fields it does not know turns it off.
    sends_prompt_cache_key: ClassVar[bool] = True
    #: Whether ``stream_options`` asks for a final usage chunk. A service that already sends one, and
    #: rejects fields it does not know, turns it off.
    sends_stream_options: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        temperature: float | None = None,
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        send_prompt_cache_key: bool | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
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
        self.send_prompt_cache_key = (
            self.sends_prompt_cache_key if send_prompt_cache_key is None else send_prompt_cache_key
        )
        self.extra_headers = validate_extra_headers(extra_headers)
        self.extra_query = validate_extra_query(extra_query)
        self.extra_body: dict[str, JsonValue] = validate_extra_body(extra_body)
        if not api_key:
            raise ValueError("api_key is required")
        base_url = base_url or self.default_base_url
        if not base_url:
            raise ValueError("base_url is required")
        self.model = model
        self.temperature = temperature
        self.response = response
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        from openai import AsyncOpenAI

        headers = {**self.vendor_headers(), **self.extra_headers}
        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
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
        """Return the JSON fields this service takes for ``request``, beyond the standard ones.

        The standard ones (``model``, ``messages``, ``tools``, ``temperature``, ``tool_choice``,
        ``prompt_cache_key``) are set by the provider. This is where the reasoning level is
        spelled the way the service wants it. The default is OpenAI's top-level
        ``reasoning_effort``. A field given in ``extra_body`` replaces one returned here.
        """
        if request.reasoning_effort == ReasoningEffort.OFF:
            return {}
        return {"reasoning_effort": request.reasoning_effort.value}

    def responses_fields(self, request: ModelRequest) -> JsonObject:
        """Return the JSON fields this service takes for a Responses request, beyond the standard ones.

        The standard ones (``model``, ``input``, ``tools``, ``temperature``, ``tool_choice``,
        ``reasoning``, ``include``) are set by the provider. They are merged into the request body
        after those fields, so a provider can replace one, such as the ``reasoning`` object. A
        field given in ``extra_body`` replaces one returned here.
        """
        return {}

    def replay_reasoning(self, message: AssistantMessage, payload: ChatMessagePayload) -> None:
        """Add what this service needs of an earlier assistant turn to its rendered ``payload``.

        The default returns the reasoning as ``reasoning_content``, which reasoning services expect
        to see again on later requests. A message from another provider is skipped, because
        reasoning from one model is not context for another.
        """
        if message.reasoning is not None and self._same_provider(message):
            payload["reasoning_content"] = message.reasoning

    def read_delta(self, delta: Mapping[str, JsonValue]) -> ChatDelta:
        """Read one streamed ``delta``.

        The default reads ``content``, ``refusal``, ``tool_calls``, and the first of
        :attr:`reasoning_keys` that has text. A service with fields of its own returns a
        :class:`ChatDelta` that also carries them, usually by extending ``super().read_delta``.
        """
        reasoning = next((text for key in self.reasoning_keys if (text := read_str(delta.get(key)))), "")
        return ChatDelta(
            text=read_str(delta.get("content")),
            reasoning=reasoning,
            refusal=read_str(delta.get("refusal")),
            tool_calls=tool_call_deltas(delta.get("tool_calls")),
        )

    def merge_replay(self, blocks: list[JsonObject], fragment: JsonObject) -> None:
        """Add one streamed opaque block (:attr:`ChatDelta.replay`) to the ones collected so far.

        The default appends. A service that streams one block in pieces overrides this to join
        them, and ``blocks`` is what finally becomes
        :attr:`~zett_agent.messages.AssistantMessage.replay_blocks`.
        """
        blocks.append(fragment)

    def read_usage(self, usage: Mapping[str, JsonValue]) -> ModelUsage:
        """Read the ``usage`` object of a chunk.

        The default accepts every spelling of the cache and reasoning counters that the
        compatible services use.
        """
        return usage_from_mapping(usage)

    def chat_stop(
        self,
        finish_reason: str | None,
        *,
        has_tool_calls: bool,
        refusal_text: str,
        stream_error: Mapping[str, JsonValue],
        native_finish_reason: str | None,
    ) -> StopInfo:
        """Say why the response ended. Override it for a service with finish reasons of its own."""
        return default_chat_stop(
            finish_reason,
            has_tool_calls=has_tool_calls,
            refusal_text=refusal_text,
            stream_error=stream_error,
            native_finish_reason=native_finish_reason,
        )

    def status_error(self, error: APIStatusError) -> ProviderError:
        """Turn an HTTP failure into the typed error the runtime reports.

        The default reports a refusal the service sent as an HTTP error, a rejected credential,
        and everything else as :class:`~zett_agent.providers.base.ProviderHTTPError`. A service
        that explains failures in a body of its own overrides this and falls back to ``super()``.
        """
        refusal = refusal_from_status_error(error.status_code, error.body)
        if refusal is not None:
            return ProviderRefusalError(refusal)
        status, failure, message = describe_status_error(self.provider_name, error)
        if status == 401:
            return ProviderAuthError(f"Provider rejected API credential. {message}")
        return http_error(status, failure, message)

    # -- requests ---------------------------------------------------------------------------------

    def _same_provider(self, message: AssistantMessage) -> bool:
        """Whether ``message`` came from this provider, or from before provenance was recorded."""
        return message.provider is None or message.provider == self.provider_name

    def _messages(self, request: ModelRequest) -> list[ChatMessagePayload]:
        rendered: list[ChatMessagePayload] = []
        for message in expand_tool_images(request.messages):
            payload = chat_message(message)
            if isinstance(message, AssistantMessage):
                self.replay_reasoning(message, payload)
            rendered.append(payload)
        return rendered

    async def _open_chat(self, request: ModelRequest) -> AsyncStream[ChatCompletionChunk]:
        """Open a Chat Completions stream."""
        from openai import APIStatusError

        tools = [*chat_tools(request.tools), *chat_server_tools(request.server_tools)]
        fields: dict[str, object] = {
            "model": self.model,
            "messages": self._messages(request),
            "stream": True,
        }
        if self.sends_stream_options:
            fields["stream_options"] = {"include_usage": True}
        if tools:
            fields["tools"] = tools
        if self.temperature is not None:
            fields["temperature"] = self.temperature
        if request.tool_choice:
            fields["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}
        if self.send_prompt_cache_key and request.cache_key and request.cache_mode is not CacheMode.OFF:
            fields["prompt_cache_key"] = request.cache_key
        if extra_body := {**self.request_fields(request), **self.extra_body}:
            fields["extra_body"] = extra_body
        create = cast(
            "Callable[..., Awaitable[AsyncStream[ChatCompletionChunk]]]", self._client.chat.completions.create
        )
        try:
            return await self._first_byte(create(**fields))
        except APIStatusError as error:
            raise self.status_error(error) from error

    async def _open_responses(self, request: ModelRequest) -> AsyncStream[ResponseStreamEvent]:
        """Open a stateless Responses API stream."""
        from openai import APIStatusError

        fields = responses_request(
            request,
            provider=self.provider_name,
            model=self.model,
            temperature=self.temperature,
            send_prompt_cache_key=self.send_prompt_cache_key,
            extra_body={**self.responses_fields(request), **self.extra_body},
        )
        create = cast("Callable[..., Awaitable[AsyncStream[ResponseStreamEvent]]]", self._client.responses.create)
        try:
            return await self._first_byte(create(**fields))
        except APIStatusError as error:
            raise self.status_error(error) from error

    async def aclose(self) -> None:
        """Close the SDK's owned HTTP connection pool."""
        await self._client.close()

    # -- the stream -------------------------------------------------------------------------------

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        handle = self._stream_responses if self.response else self._stream_chat
        async with aclosing_stream(handle(request)) as events:
            async for event in events:
                yield event

    def _refused(self, refusal: ProviderRefusalError) -> ModelEvent:
        """Answer a request that was refused before any output, so the conversation goes on."""
        return ModelEvent.completed(
            ModelResponse(
                AssistantMessage(content="", provider=self.provider_name, model=self.model),
                finish_reason=refusal.stop.raw,
                stop=refusal.stop,
            )
        )

    async def _response_events(self, response: AsyncStream[ResponseStreamEvent]) -> AsyncIterator[JsonObject]:
        """Read a Responses stream as JSON, guarded against a silent service."""
        async for event in guard_idle(response, self.timeout):
            yield cast(JsonObject, event.model_dump(exclude_none=True, warnings=False))

    async def _stream_responses(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        try:
            response = await self._open_responses(request)
        except ProviderRefusalError as refusal:
            yield self._refused(refusal)
            return
        await require_event_stream(response, self.provider_name)
        try:
            async for event in stream_responses(
                self._response_events(response),
                provider=self.provider_name,
                model=self.model,
                local_search_tool=local_tool_search_name(request.tools),
            ):
                yield event
        except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderResponseError(str(error)) from error
        finally:
            await response.close()

    async def _stream_chat(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        from openai import APIConnectionError, APITimeoutError
        from openai import APIError as OpenAIError

        try:
            response = await self._open_chat(request)
        except ProviderRefusalError as refusal:
            yield self._refused(refusal)
            return
        await require_event_stream(response, self.provider_name)

        text = ""
        reasoning = ""
        streams: dict[int, ToolCallAccumulator] = {}
        replay: list[JsonObject] = []
        finish_reason: str | None = None
        refusal_text = ""
        stream_error: Mapping[str, JsonValue] = {}
        native_finish_reason: str | None = None
        usage = ModelUsage()

        chunks = 0
        try:
            async for chunk in guard_idle(response, self.timeout):
                if chunk is None:
                    continue
                chunks += 1
                data = chunk_data(chunk)
                if isinstance(usage_payload := data.get("usage"), dict):
                    usage = self.read_usage(usage_payload)

                choice = next(iter(read_objects(data.get("choices"))), None)
                if choice is None:
                    continue
                if isinstance(native := choice.get("native_finish_reason"), str):
                    native_finish_reason = native

                delta = self.read_delta(read_object(choice.get("delta")))
                if delta.refusal:
                    # A refusal is the model's answer to the user, so it is shown like any
                    # other text, and it is also what the stop explains itself with.
                    refusal_text += delta.refusal
                    text += delta.refusal
                    yield ModelEvent.text(delta.refusal)
                if delta.text:
                    text += delta.text
                    yield ModelEvent.text(delta.text)
                if delta.reasoning:
                    reasoning += delta.reasoning
                    yield ModelEvent.reasoning(delta.reasoning)
                for call in delta.tool_calls:
                    streams.setdefault(call.index, ToolCallAccumulator(index=call.index)).append(call)
                    yield ModelEvent.tool_call(call)
                for fragment in delta.replay:
                    self.merge_replay(replay, fragment)

                if isinstance(reason := choice.get("finish_reason"), str):
                    finish_reason = reason
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
            raise ProviderResponseError(empty_stream_message(self.provider_name, response))

        tool_calls = parse_tool_calls(streams, finish_reason)
        stop = self.chat_stop(
            finish_reason,
            has_tool_calls=bool(tool_calls),
            refusal_text=refusal_text,
            stream_error=stream_error,
            native_finish_reason=native_finish_reason,
        )
        message = AssistantMessage(
            content=text,
            reasoning=reasoning or None,
            tool_calls=tool_calls,
            provider=self.provider_name,
            model=self.model,
            replay_blocks=tuple(replay),
        )
        yield ModelEvent.completed(ModelResponse(message=message, finish_reason=finish_reason, usage=usage, stop=stop))
