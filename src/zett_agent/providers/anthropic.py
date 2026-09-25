from __future__ import annotations

import base64
import ssl
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import httpx
import truststore
from anthropic import AnthropicError, AsyncAnthropic

from ..messages import (
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
    TextContent,
)
from ..model import (
    DEFAULT_RETRY_OPTIONS,
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
    ToolCallDelta,
    ToolDefinition,
    validate_response,
    validate_retry,
)
from .base import (
    ProviderAuthError,
    ProviderResponseError,
    RetryingProvider,
    _parse_tool_arguments,
    _parse_tool_calls,
    _reasoning_effort_to_budget,
    _to_model_data,
    _ToolCallAccumulator,
    reject_local_tool_search,
    retry_model_stream,
)
from .tool_images import expand_tool_images


def _tools_to_anthropic_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    reject_local_tool_search("Anthropic messages", tools)
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


def _server_tools_to_anthropic_payload(tools: Sequence[ServerToolDefinition]) -> list[dict[str, Any]]:
    """Render Anthropic server tools using their versioned native schemas."""
    return [{"type": tool.type, **dict(tool.configuration)} for tool in tools]


def _to_anthropic_content_blocks(message: Any, *, model: str | None = None) -> list[dict[str, Any]]:
    """Convert a domain message into Anthropic content blocks."""
    role = message.role
    match role:
        case "user" | "agent":
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
            if message.provider == "anthropic" and message.model == model and message.replay_blocks:
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


class AnthropicProvider(RetryingProvider):
    """Map Anthropic SDK streams, including signed thinking and tool blocks.

    Args:
        model: Model identifier accepted by the Anthropic-compatible endpoint.
        api_key: API credential supplied by the application.
        base_url: API root, defaulting to the official Anthropic endpoint.
        transport: Optional HTTPX transport for isolated tests or custom routing.
        response: Must remain false; this adapter uses Anthropic Messages API.
        retry: Exponential backoff for transient failures before output starts.

    Note:
        Signed replay blocks belong to the originating provider and model.
        Persist them unchanged for tool round trips; do not render them as
        user-facing reasoning. Await aclose() to release client resources.

    Examples:
        Stream an Anthropic model through the provider-neutral client::

            model = AnthropicProvider(
                model=os.environ["ANTHROPIC_MODEL"],
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            try:
                client = await create_agent(model)
                reply = await client.run("Review the proposed API")
            finally:
                await model.aclose()

    .. warning::
        Signed thinking and tool replay blocks must be returned unchanged only
        to the same provider and model. They are not interchangeable with plain
        reasoning text.

    .. seealso::
        :class:`~zett_agent.AssistantMessage` documents ``replay_blocks``;
        :doc:`/concepts/context` explains replay during tool round trips.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
    ) -> None:
        validate_response(response)
        if response:
            raise ValueError("AnthropicProvider does not support the Responses API")
        if not api_key:
            raise ValueError("api_key is required")
        validate_retry(retry)
        self.retry = retry
        self.response = response
        self.model = model
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, http_client=self._http_client, max_retries=0)

    async def aclose(self) -> None:
        """Close the owned SDK connection pool."""
        await self._client.close()

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        messages: list[dict[str, Any]] = []
        system: list[str] = []
        for message in expand_tool_images(request.messages):
            match message.role:
                case "system":
                    system.append(message.content)
                case "user" | "agent":
                    content = _to_anthropic_content_blocks(message, model=self.model)
                    if content:
                        messages.append({"role": "user", "content": content})
                case "assistant":
                    content = _to_anthropic_content_blocks(message, model=self.model)
                    if content:
                        messages.append({"role": "assistant", "content": content})
                case "tool":
                    content = _to_anthropic_content_blocks(message, model=self.model)
                    if content:
                        messages.append({"role": "user", "content": content})

        body = {
            "model": self.model,
            "system": "\n\n".join(system),
            "messages": messages,
            "tools": [
                *_tools_to_anthropic_payload(request.tools),
                *_server_tools_to_anthropic_payload(request.server_tools),
            ],
            "max_tokens": 4096,
            "stream": True,
        }
        if not system:
            body.pop("system")
        if request.tool_choice:
            body["tool_choice"] = {
                "type": "tool",
                "name": request.tool_choice,
                "disable_parallel_tool_use": not request.parallel_tool_call,
            }
        elif not request.parallel_tool_call and (request.tools or request.server_tools):
            body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
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
        server_streams: dict[int, _ToolCallAccumulator] = {}
        server_initial_inputs: dict[int, dict[str, Any]] = {}
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
                                stream = server_streams.get(index)
                                if stream is not None:
                                    stream.argument_buffer += partial
                                    yield ModelEvent.server_tool_input(
                                        ServerToolInputDelta(stream.call_id, partial),
                                    )
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
                            case "server_tool_use":
                                call_id = str(block.get("id", ""))
                                name = str(block.get("name", ""))
                                initial_input = block.get("input")
                                input_data = initial_input if isinstance(initial_input, dict) else {}
                                stream = _ToolCallAccumulator(index=index, call_id=call_id, name=name)
                                server_streams[index] = stream
                                server_initial_inputs[index] = input_data
                                yield ModelEvent.server_tool_started(ServerToolCall(call_id, name, input_data or None))
                            case str(block_type) if block_type.endswith("_tool_result"):
                                call_id = str(block.get("tool_use_id", ""))
                                name = block_type.removesuffix("_tool_result")
                                output = block.get("content")
                                error_code = _anthropic_server_tool_error(output)
                                result = ServerToolResult(call_id, name, output, error_code)
                                yield (
                                    ModelEvent.server_tool_failed(result)
                                    if error_code is not None
                                    else ModelEvent.server_tool_completed(result)
                                )
                            case _:
                                pass
                    case "content_block_stop":
                        index = int(data.get("index", 0))
                        if index in streams:
                            replay[index]["input"] = _parse_tool_arguments(
                                streams[index].argument_buffer,
                                index=index,
                            )
                        if index in server_streams:
                            stream = server_streams[index]
                            replay[index]["input"] = (
                                _parse_tool_arguments(stream.argument_buffer, index=index)
                                if stream.argument_buffer
                                else server_initial_inputs[index]
                            )
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
            replay[index]["input"] = _parse_tool_arguments(stream.argument_buffer, index=index)
        yield ModelEvent.completed(
            ModelResponse(
                message=AssistantMessage(
                    content=text,
                    reasoning=reasoning or None,
                    tool_calls=tool_calls,
                    provider="anthropic",
                    model=self.model,
                    replay_blocks=tuple(replay[index] for index in sorted(replay)),
                ),
                finish_reason=finish_reason,
                usage=usage,
            )
        )


def _anthropic_server_tool_error(output: Any) -> str | None:
    """Return Anthropic's structured hosted-tool error code, when present."""
    if not isinstance(output, dict):
        return None
    output_type = output.get("type")
    if not isinstance(output_type, str) or not output_type.endswith("_error"):
        return None
    code = output.get("error_code")
    return str(code) if code else output_type
