from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from typing import Any

import httpx
import truststore
from openai import APIStatusError, AsyncOpenAI

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
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    ToolCallDelta,
    ToolDefinition,
)


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


def _to_model_data(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=True)  # type: ignore[call-arg]
    if hasattr(payload, "dict"):
        return payload.dict()  # type: ignore[operator]
    if isinstance(payload, Message):
        return {"role": payload.role, **asdict(payload)}
    normalized = json.loads(json.dumps(payload, default=str))
    if not isinstance(normalized, dict):
        raise ProviderResponseError(f"Provider SDK value did not normalize to an object: {payload!r}")
    return normalized


def _normalize_image_source(
    part: ImageUrlSource | ImageBytesSource, detail: ImageDetail = ImageDetail.AUTO
) -> dict[str, Any]:
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


def _message_to_openai_payload(message: AnyMessage) -> dict[str, Any]:
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
                    parts: list[dict[str, Any]] = []
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
            payload: dict[str, Any] = {"role": "assistant", "content": message.content}
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
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "content": message.content,
            }
        case _:
            raise ProviderResponseError(f"Unsupported role in message conversion: {role}")


def _tools_to_openai_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    rendered = []
    for tool in tools:
        rendered.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
        )
    return rendered


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
        try:
            args = json.loads(stream.argument_buffer) if stream.argument_buffer else {}
        except JSONDecodeError as error:
            raise ProviderResponseError(f"Invalid tool-call arguments for index {index}: {error}") from error
        if not stream.call_id:
            stream.call_id = f"call_{index}"
        if not stream.name:
            raise ProviderResponseError(f"Tool call at index {index} did not provide a name")
        calls.append(ToolCall(id=stream.call_id, name=stream.name, arguments=args))
    return tuple(calls)


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


class _OpenAIStyleProvider:
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
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self.temperature = temperature
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=self._http_client)

    async def _request(self, request: ModelRequest) -> Any:
        messages = [_message_to_openai_payload(message) for message in request.messages]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": _tools_to_openai_payload(request.tools),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload.update(self._provider_specific_request_fields(request))
        if not request.tools:
            payload.pop("tools")
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if request.tool_choice:
            payload["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}
        if self.provider_name == "deepseek":
            for source, target in zip(request.messages, messages, strict=True):
                if isinstance(source, AssistantMessage) and source.reasoning is not None:
                    target["reasoning_content"] = source.reasoning
        extra_body = self._provider_specific_request_extra_fields(request)
        if extra_body:
            payload["extra_body"] = extra_body
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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
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
            content=text, reasoning=reasoning or None, tool_calls=tool_calls, provider=self.provider_name
        )
        yield ModelEvent.completed(
            ModelResponse(
                message=response_message,
                finish_reason=finish_reason,
                usage=usage,
            )
        )
