from __future__ import annotations

import asyncio
import base64
import json
import ssl
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import asdict, dataclass
from functools import wraps
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx
import truststore
from json_repair import repair_json

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
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    ServerToolDefinition,
    ToolCallDelta,
    ToolDefinition,
    validate_response,
    validate_retry,
)

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartParam,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
    from openai.types.shared_params import FunctionDefinition

#: The OpenAI SDK is imported where this adapter needs it -- client construction
#: and request failures -- because importing it alone costs a few hundred
#: milliseconds. Payload rendering therefore uses plain dictionaries that still
#: type-check against the SDK's TypedDicts.
from ..sync_runtime import SyncMethodsMixin
from .responses import local_tool_search_name, responses_input, responses_reasoning, responses_tools, stream_responses
from .tool_images import expand_tool_images


class ProviderError(AgentError):
    """Base error for provider adapter failures."""


class ProviderAuthError(ProviderError):
    """Raised when a remote provider rejects the configured credentials."""


class ProviderResponseError(ProviderError):
    """Raised when provider responses do not satisfy the expected stream protocol."""


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
        case ReasoningEffort.XHIGH:
            return 32768


def _to_model_data(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=True)  # type: ignore[call-arg]
    if hasattr(payload, "dict"):
        return payload.dict()  # type: ignore[operator]
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


def _message_to_openai_payload(message: AnyMessage) -> ChatCompletionMessageParam:
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
            payload: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": message.content}
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
    payload_messages: Sequence[ChatCompletionMessageParam],
    *,
    provider_name: str,
) -> list[ChatCompletionMessageParam]:
    """Replay same-provider reasoning on OpenAI-compatible assistant history.

    Reasoning-aware providers, including DeepSeek thinking mode, require the
    assistant ``reasoning_content`` to be sent back on later requests. It is
    model context that cannot be reconstructed from the final answer or tool
    calls. Messages without provider metadata remain compatible with history
    created before provider attribution was stored.

    See https://api-docs.deepseek.com/zh-cn/guides/thinking_mode.
    """
    rendered = [cast("ChatCompletionMessageParam", dict(payload)) for payload in payload_messages]
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


def _parse_tool_calls(streams: Mapping[int, _ToolCallAccumulator]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index in sorted(streams):
        stream = streams[index]
        args = _parse_tool_arguments(stream.argument_buffer, index=index)
        if not stream.call_id:
            stream.call_id = f"call_{index}"
        if not stream.name:
            raise ProviderResponseError(f"Tool call at index {index} did not provide a name")
        calls.append(ToolCall(id=stream.call_id, name=stream.name, arguments=args))
    return tuple(calls)


def _parse_tool_arguments(payload: str, *, index: int) -> dict[str, Any]:
    """Decode one accumulated tool argument object from a provider stream.

    Some OpenAI-compatible providers, including DeepSeek, occasionally expose
    decoded newlines or tabs inside the nested ``function.arguments`` string.
    ``strict=False`` accepts those control characters without changing valid
    JSON semantics. Models can also occasionally omit a comma or fail to escape
    a quote in a long argument string. For a complete top-level object, use the
    bounded ``json-repair`` fallback before rejecting the model response.

    The fallback deliberately requires both outer braces. A payload cut off by
    a token limit must fail instead of being completed heuristically and passed
    to a mutating tool with silently truncated content. Tool arguments must
    remain an object because the runtime invokes tools with named keyword
    arguments; the tool's Pydantic input model performs the final schema check.
    """
    try:
        parsed = json.loads(payload, strict=False) if payload else {}
    except JSONDecodeError as error:
        stripped = payload.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            raise ProviderResponseError(f"Invalid tool-call arguments for index {index}: {error}") from error
        try:
            parsed = repair_json(stripped, return_objects=True, skip_json_loads=True)
        except (ValueError, TypeError, RecursionError) as repair_error:
            raise ProviderResponseError(
                f"Invalid tool-call arguments for index {index}: {error}; repair failed: {repair_error}"
            ) from error
    if not isinstance(parsed, dict):
        raise ProviderResponseError(f"Tool-call arguments for index {index} must be a JSON object")
    return parsed


def _usage_from_mapping(payload: Mapping[str, Any]) -> ModelUsage:
    """Normalize token usage returned by OpenAI-compatible chat APIs.

    This helper is shared by :class:`OpenAIProvider` and
    :class:`DeepSeekProvider`. Their response fields map as follows::

        Provider   API field                                  ModelUsage field
        ---------- ------------------------------------------ ----------------------
        OpenAI     prompt_tokens                              input_tokens
        OpenAI     completion_tokens                          output_tokens
        OpenAI     prompt_tokens_details.cached_tokens        cache_read_tokens
        OpenAI     completion_tokens_details.reasoning_tokens reasoning_tokens
        DeepSeek   prompt_tokens                              input_tokens
        DeepSeek   completion_tokens                          output_tokens
        DeepSeek   prompt_cache_hit_tokens                    cache_read_tokens
        DeepSeek   completion_tokens_details.reasoning_tokens reasoning_tokens

    ``prompt_tokens`` includes cached input tokens. For DeepSeek it is the sum
    of ``prompt_cache_hit_tokens`` and ``prompt_cache_miss_tokens``; the miss
    count therefore has no separate destination in :class:`ModelUsage`.
    Likewise, ``completion_tokens`` includes reasoning tokens rather than being
    added to them. Both providers return ``total_tokens``, but ``ModelUsage``
    derives that total from input plus output to preserve one internal invariant.

    Neither schema reports cache-creation tokens, so ``cache_write_tokens`` is
    zero. Anthropic reports ``input_tokens``, ``cache_read_input_tokens``, and
    ``cache_creation_input_tokens`` separately. Gemini reports
    ``prompt_token_count``, ``candidates_token_count``,
    ``cached_content_token_count``, and ``thoughts_token_count``. Ollama reports
    ``prompt_eval_count`` and ``eval_count``. Those provider-specific schemas
    are normalized in their own adapters rather than by this helper.
    """
    cache_read_tokens = int(payload.get("prompt_cache_hit_tokens", 0) or 0)
    reasoning_tokens = 0

    prompt_details = payload.get("prompt_tokens_details")
    if isinstance(prompt_details, Mapping):
        cache_read_tokens = int(prompt_details.get("cached_tokens", cache_read_tokens) or 0)
    completion_details = payload.get("completion_tokens_details")
    if isinstance(completion_details, Mapping):
        reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)

    return ModelUsage(
        input_tokens=int(payload.get("prompt_tokens", 0) or 0),
        output_tokens=int(payload.get("completion_tokens", 0) or 0),
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=0,
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
    response: bool = False

    @staticmethod
    def _retryable(error: BaseException) -> bool:
        """Inspect SDK status codes and wrapped transport causes without optional imports."""
        seen: set[int] = set()
        while id(error) not in seen:
            seen.add(id(error))
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
                async with aclosing(stream(self, request)) as events:
                    async for event in events:
                        emitted = True
                        yield event
                return
            except Exception as error:
                if emitted or attempt == retry.max_retries or not self._retryable(error):
                    raise
                await asyncio.sleep(delay)
                # Saturate before doubling to avoid overflow for large delays.
                delay = retry.max_delay if delay >= retry.max_delay / 2 else delay * 2

    return retrying_stream


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
    ) -> None:
        validate_retry(retry)
        validate_response(response)
        self.retry = retry
        self.send_prompt_cache_key = send_prompt_cache_key
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

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=self._http_client, max_retries=0)

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
        extra_body = self._provider_specific_request_extra_fields(request)
        if extra_body:
            payload["extra_body"] = extra_body
        from openai import APIStatusError

        try:
            return await self._client.chat.completions.create(**payload)  # type: ignore[misc]
        except APIStatusError as error:
            status_code = error.status_code
            match status_code:
                case 401:
                    raise ProviderAuthError("Provider rejected API credential") from error
                case 400 | 422 | 500:
                    raise ProviderResponseError("Provider returned an invalid stream payload") from error
                case _:
                    raise ProviderResponseError("Provider stream request failed") from error

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
        if self.send_prompt_cache_key and request.cache_key:
            payload["prompt_cache_key"] = request.cache_key
        from openai import APIStatusError

        try:
            return await self._client.responses.create(**payload)  # type: ignore[misc]
        except APIStatusError as error:
            match error.status_code:
                case 401:
                    raise ProviderAuthError("Provider rejected API credential") from error
                case 400 | 422 | 500:
                    raise ProviderResponseError("Provider returned an invalid Responses stream") from error
                case _:
                    raise ProviderResponseError("Provider Responses request failed") from error

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
        if self.response:
            response = await self._request(request)
            try:
                async for event in stream_responses(
                    response,
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
        usage = ModelUsage()

        response = await self._request(request)
        try:
            async for chunk in response:
                if chunk is None:
                    continue
                match getattr(chunk, "usage", None):
                    case None:
                        pass
                    case usage_payload:
                        raw = _to_model_data(usage_payload)
                        usage = _usage_from_mapping(raw)

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    if getattr(choices[0], "finish_reason", None):
                        finish_reason = choices[0].finish_reason
                    continue

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
        finally:
            await response.close()

        tool_calls = _parse_tool_calls(streams)
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
            )
        )
