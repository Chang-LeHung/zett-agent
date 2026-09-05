from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from json import JSONDecodeError
from typing import Any

import httpx
import truststore
from anthropic import AnthropicError, AsyncAnthropic
from openai import APIStatusError, AsyncOpenAI

from .exceptions import AgentError
from .messages import (
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
from .model import (
    AgentModel,
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
    return json.loads(json.dumps(payload, default=str))


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


def _tools_to_anthropic_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    rendered = []
    for tool in tools:
        rendered.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.parameters),
            }
        )
    return rendered


def _tools_to_ollama_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
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


def _to_anthropic_content_blocks(message: Any) -> list[dict[str, Any]]:
    """Convert a domain message into Anthropic content blocks."""
    role = message.role
    match role:
        case "user":
            content = message.content
            if isinstance(content, str):
                return [{"type": "text", "text": content}] if content else []
            blocks: list[dict[str, Any]] = []
            for part in content:
                match part:
                    case TextContent(text=text):
                        blocks.append({"type": "text", "text": text})
                    case ImageContent(source=source):
                        blocks.append(_to_anthropic_image_block(source))
            return blocks
        case "assistant":
            if message.provider == "anthropic" and message.replay_blocks:
                return [dict(block) for block in message.replay_blocks]
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                )
            return blocks
        case "tool":
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": not message.success,
                }
            ]
        case _:
            return []


def _to_anthropic_image_block(source: Any) -> dict[str, Any]:
    """Convert a typed image source into an Anthropic-compatible content block."""
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
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded},
            }
        case _:
            raise ProviderResponseError(f"Unsupported image content source: {source!r}")


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
    cache_read = int(payload.get("prompt_cache_hit_tokens", 0) or 0)
    cache_write = 0
    reasoning = 0

    if "prompt_tokens_details" in payload and isinstance(payload["prompt_tokens_details"], dict):
        details = payload["prompt_tokens_details"]
        cache_read = int(details.get("cached_tokens", cache_read) or 0)
    if "completion_tokens_details" in payload and isinstance(payload["completion_tokens_details"], dict):
        details = payload["completion_tokens_details"]
        reasoning = int(details.get("reasoning_tokens", 0) or 0)

    return ModelUsage(
        input_tokens=int(payload.get("prompt_tokens", 0) or 0),
        output_tokens=int(payload.get("completion_tokens", 0) or 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
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
            transport=transport, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
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


class OpenAIProvider(_OpenAIStyleProvider):
    """OpenAI-compatible streaming adapter with event mapping to model-neutral objects."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            transport=transport,
            temperature=temperature,
        )
        self.provider_name = "openai"


class DeepSeekProvider(_OpenAIStyleProvider):
    """DeepSeek streaming adapter implemented through OpenAI-compatible protocol."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com/v1",
            transport=transport,
            temperature=temperature,
        )
        self.provider_name = "deepseek"

    def _provider_specific_request_fields(self, request: ModelRequest) -> dict[str, Any]:
        if request.reasoning_effort == ReasoningEffort.OFF or not self.model.startswith("deepseek-v4"):
            return {}
        match request.reasoning_effort:
            case ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
                effort = "low"
            case ReasoningEffort.XHIGH:
                effort = "max"
            case _:
                effort = "high"
        return {"reasoning_effort": effort}

    def _provider_specific_request_extra_fields(self, request: ModelRequest) -> dict[str, Any]:
        thinking = {"type": "enabled"} if request.reasoning_effort != ReasoningEffort.OFF else {"type": "disabled"}
        return {"thinking": thinking}


class AnthropicProvider(AgentModel):
    """Anthropic provider adapter with raw SDK stream-to-event mapping."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self._http_client = httpx.AsyncClient(
            transport=transport, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        )
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, http_client=self._http_client)

    async def aclose(self) -> None:
        """Close the owned SDK connection pool."""
        await self._client.close()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        messages: list[dict[str, Any]] = []
        system: list[str] = []
        for message in request.messages:
            match message.role:
                case "system":
                    system.append(message.content)
                case "user":
                    content = _to_anthropic_content_blocks(message)
                    if content:
                        messages.append({"role": "user", "content": content})
                case "assistant":
                    content = _to_anthropic_content_blocks(message)
                    if content:
                        messages.append({"role": "assistant", "content": content})
                case "tool":
                    content = _to_anthropic_content_blocks(message)
                    if content:
                        messages.append({"role": "user", "content": content})

        body = {
            "model": self.model,
            "system": "\n\n".join(system),
            "messages": messages,
            "tools": _tools_to_anthropic_payload(request.tools),
            "max_tokens": 4096,
            "stream": True,
        }
        if not system:
            body.pop("system")
        if request.tool_choice:
            body["tool_choice"] = {"type": "tool", "name": request.tool_choice}
        if request.reasoning_effort != ReasoningEffort.OFF and not request.tool_choice:
            body["max_tokens"] = _reasoning_effort_to_budget(request.reasoning_effort) + 4096
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": _reasoning_effort_to_budget(request.reasoning_effort),
            }
        else:
            body["thinking"] = {"type": "disabled"}

        text = ""
        reasoning = ""
        finish_reason = None
        streams: dict[int, _ToolCallAccumulator] = {}
        usage = ModelUsage()
        replay: dict[int, dict[str, Any]] = {}

        try:
            response = await self._client.messages.create(**body)
        except AnthropicError as error:
            status_code = getattr(error, "status_code", None)
            match status_code:
                case 401:
                    raise ProviderAuthError("Provider rejected API credential") from error
                case _:
                    raise ProviderResponseError("Provider stream request failed") from error

        try:
            async for event in response:
                data = _to_model_data(event)
                event_type = data.get("type")
                match event_type:
                    case "message_start":
                        message_usage = data.get("message", {}).get("usage", {})
                        usage = ModelUsage(
                            input_tokens=sum(
                                int(message_usage.get(key, 0) or 0)
                                for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                            ),
                            cache_read_tokens=int(message_usage.get("cache_read_input_tokens", 0) or 0),
                            cache_write_tokens=int(message_usage.get("cache_creation_input_tokens", 0) or 0),
                            output_tokens=int(message_usage.get("output_tokens", 0) or 0),
                        )
                    case "content_block_delta":
                        index = int(data.get("index", 0))
                        delta = data.get("delta", {})
                        match delta.get("type"):
                            case "text_delta":
                                chunk = delta.get("text", "")
                                text += chunk
                                replay[index]["text"] = replay[index].get("text", "") + chunk
                                yield ModelEvent.text(chunk)
                            case "thinking_delta":
                                chunk = delta.get("thinking", "")
                                reasoning += chunk
                                replay[index]["thinking"] = replay[index].get("thinking", "") + chunk
                                yield ModelEvent.reasoning(chunk)
                            case "signature_delta":
                                replay[index]["signature"] = replay[index].get("signature", "") + delta.get(
                                    "signature", ""
                                )
                            case "input_json_delta":
                                partial = delta.get("partial_json", "")
                                if not partial:
                                    continue
                                stream = streams.setdefault(index, _ToolCallAccumulator(index=index))
                                stream.append(ToolCallDelta(index=index, arguments_delta=partial))
                                yield ModelEvent.tool_call(ToolCallDelta(index=index, arguments_delta=partial))
                            case _:
                                pass
                    case "content_block_start":
                        index = int(data.get("index", 0))
                        block = data.get("content_block", {})
                        replay[index] = dict(block)
                        match block.get("type"):
                            case "tool_use":
                                stream = streams.setdefault(index, _ToolCallAccumulator(index=index))
                                stream.call_id = block.get("id", stream.call_id)
                                stream.name = block.get("name", stream.name)
                                yield ModelEvent.tool_call(
                                    ToolCallDelta(index=index, id_delta=stream.call_id, name_delta=stream.name)
                                )
                            case _:
                                pass
                    case "content_block_stop":
                        index = int(data.get("index", 0))
                        if index in streams:
                            replay[index]["input"] = json.loads(streams[index].argument_buffer or "{}")
                    case "message_delta":
                        delta = data.get("delta", {})
                        finish_reason = delta.get("stop_reason")
                        token_info = data.get("usage", {})
                        usage = replace(
                            usage,
                            output_tokens=int(token_info.get("output_tokens", 0) or 0),
                        )
                    case _:
                        pass
        finally:
            await response.close()

        tool_calls = _parse_tool_calls(streams)
        for index, stream in streams.items():
            replay[index]["input"] = json.loads(stream.argument_buffer or "{}")
        yield ModelEvent.completed(
            ModelResponse(
                message=AssistantMessage(
                    content=text,
                    reasoning=reasoning or None,
                    tool_calls=tool_calls,
                    provider="anthropic",
                    replay_blocks=tuple(replay[index] for index in sorted(replay)),
                ),
                finish_reason=finish_reason,
                usage=usage,
            )
        )


class GoogleProvider(AgentModel):
    """Map official Google GenAI SDK streams and preserve signed replay parts."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self._http_client = httpx.AsyncClient(
            transport=transport, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(httpx_async_client=self._http_client),
        )

    async def aclose(self) -> None:
        """Release both SDK clients and the injected HTTP connection pool."""
        await self._client.aio.aclose()
        self._client.close()
        await self._http_client.aclose()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        from google.genai import types

        system: list[str] = []
        contents: list[types.Content] = []
        for message in request.messages:
            match message.role:
                case "system":
                    system.append(message.content)
                    continue
                case "user":
                    parts: list[types.Part] = []
                    for part in message.parts:
                        match part:
                            case TextContent(text=text):
                                parts.append(types.Part.from_text(text=text))
                            case ImageContent(source=ImageBytesSource(data=data, media_type=media_type)):
                                parts.append(types.Part.from_bytes(data=data, mime_type=media_type))
                            case ImageContent(source=ImageUrlSource(url=url)):
                                if url.startswith("data:"):
                                    header, _, data = url.partition(",")
                                    parts.append(
                                        types.Part.from_bytes(
                                            data=base64.b64decode(data, validate=True),
                                            mime_type=header[5:].split(";")[0],
                                        )
                                    )
                                else:
                                    parts.append(types.Part.from_uri(file_uri=url))
                case "assistant":
                    if message.provider == "google" and message.replay_blocks:
                        parts = [types.Part.model_validate(dict(block)) for block in message.replay_blocks]
                    else:
                        parts = [types.Part.from_text(text=message.content)] if message.content else []
                        parts.extend(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=call.name,
                                    args=dict(call.arguments),
                                    id=call.id,
                                )
                            )
                            for call in message.tool_calls
                        )
                case "tool":
                    parts = [
                        types.Part.from_function_response(
                            name=message.name,
                            response={"content": message.content},
                        )
                    ]
            if parts:
                contents.append(types.Content(role="model" if message.role == "assistant" else "user", parts=parts))
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system) or None,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=dict(tool.parameters),
                        )
                        for tool in request.tools
                    ]
                )
            ]
            if request.tools
            else None,
            thinking_config=types.ThinkingConfig(
                thinking_budget=_reasoning_effort_to_budget(request.reasoning_effort),
                include_thoughts=True,
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[request.tool_choice],
                )
            )
            if request.tool_choice
            else None,
        )
        text, reasoning = "", ""
        calls: list[ToolCall] = []
        replay: list[dict[str, Any]] = []
        usage = ModelUsage()
        finish_reason = None
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        try:
            async for chunk in stream:
                for candidate in (chunk.candidates or [])[:1]:
                    for part in candidate.content.parts if candidate.content and candidate.content.parts else []:
                        replay.append(part.model_dump(exclude_none=True))
                        if part.text:
                            if part.thought:
                                reasoning += part.text
                                yield ModelEvent.reasoning(part.text)
                            else:
                                text += part.text
                                yield ModelEvent.text(part.text)
                        if part.function_call:
                            call = part.function_call
                            normalized = ToolCall(call.id or f"call_{len(calls)}", call.name, call.args or {})
                            calls.append(normalized)
                            yield ModelEvent.tool_call(
                                ToolCallDelta(
                                    index=len(calls) - 1,
                                    id_delta=normalized.id,
                                    name_delta=normalized.name,
                                    arguments_delta=json.dumps(dict(normalized.arguments)),
                                )
                            )
                    if candidate.finish_reason:
                        finish_reason = str(candidate.finish_reason.value)
                if chunk.usage_metadata:
                    info = chunk.usage_metadata
                    usage = ModelUsage(
                        input_tokens=info.prompt_token_count or 0,
                        output_tokens=(info.candidates_token_count or 0) + (info.thoughts_token_count or 0),
                        reasoning_tokens=info.thoughts_token_count or 0,
                        cache_read_tokens=info.cached_content_token_count or 0,
                    )
        finally:
            await stream.aclose()
        if finish_reason is None:
            raise ProviderResponseError("Google stream ended without a finish reason")
        yield ModelEvent.completed(
            ModelResponse(
                AssistantMessage(
                    content=text,
                    reasoning=reasoning or None,
                    tool_calls=tuple(calls),
                    provider="google",
                    replay_blocks=tuple(replay),
                ),
                finish_reason=finish_reason,
                usage=usage,
            )
        )


class OllamaProvider(AgentModel):
    """Stream official Ollama SDK events without buffering the entire HTTP response."""

    def __init__(
        self,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str = "http://localhost:11434",
    ) -> None:
        from ollama import AsyncClient

        self.model = model
        self._client = AsyncClient(
            host=base_url, transport=transport, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        )

    async def aclose(self) -> None:
        """Close the SDK-owned HTTP connection pool."""
        await self._client._client.aclose()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            match message.role:
                case "system":
                    messages.append({"role": "system", "content": message.content})
                case "user":
                    payload: dict[str, Any] = {"role": "user", "content": message.text}
                    images: list[str] = []
                    for part in message.parts:
                        match part:
                            case ImageContent(source=ImageBytesSource(data=data)):
                                images.append(base64.b64encode(data).decode("ascii"))
                            case ImageContent(source=ImageUrlSource(url=url)) if url.startswith("data:"):
                                images.append(url.partition(",")[2])
                            case ImageContent(source=ImageUrlSource()):
                                raise ProviderResponseError("Ollama images must be bytes or base64 data URLs")
                    if images:
                        payload["images"] = images
                    messages.append(payload)
                case "assistant":
                    payload = {"role": "assistant", "content": message.content}
                    if message.tool_calls:
                        payload["tool_calls"] = [
                            {"function": {"name": call.name, "arguments": dict(call.arguments)}}
                            for call in message.tool_calls
                        ]
                    if message.reasoning:
                        payload["thinking"] = message.reasoning
                    messages.append(payload)
                case "tool":
                    messages.append(
                        {
                            "role": "tool",
                            "content": message.content,
                            "tool_name": message.name,
                        }
                    )
        if request.tool_choice:
            messages.insert(0, {"role": "system", "content": f"Respond only by calling {request.tool_choice}."})
        stream = await self._client.chat(
            model=self.model,
            messages=messages,
            tools=_tools_to_ollama_payload(request.tools) or None,
            stream=True,
            think=request.reasoning_effort != ReasoningEffort.OFF,
        )
        text, reasoning = "", ""
        calls: list[ToolCall] = []
        usage = ModelUsage()
        finish_reason = None
        done = False
        try:
            async for chunk in stream:
                if chunk.message.thinking:
                    reasoning += chunk.message.thinking
                    yield ModelEvent.reasoning(chunk.message.thinking)
                if chunk.message.content:
                    text += chunk.message.content
                    yield ModelEvent.text(chunk.message.content)
                for call in chunk.message.tool_calls or []:
                    normalized = ToolCall(f"call_{len(calls)}", call.function.name, call.function.arguments)
                    calls.append(normalized)
                    yield ModelEvent.tool_call(
                        ToolCallDelta(
                            index=len(calls) - 1,
                            id_delta=normalized.id,
                            name_delta=normalized.name,
                            arguments_delta=json.dumps(dict(normalized.arguments)),
                        )
                    )
                if chunk.done:
                    done = True
                    finish_reason = chunk.done_reason
                    usage = ModelUsage(input_tokens=chunk.prompt_eval_count or 0, output_tokens=chunk.eval_count or 0)
        finally:
            await stream.aclose()
        if not done:
            raise ProviderResponseError("Ollama stream ended without a terminal response")
        yield ModelEvent.completed(
            ModelResponse(
                AssistantMessage(content=text, reasoning=reasoning or None, tool_calls=tuple(calls), provider="ollama"),
                finish_reason=finish_reason,
                usage=usage,
            )
        )
