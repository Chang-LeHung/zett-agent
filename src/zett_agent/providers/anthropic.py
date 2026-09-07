from __future__ import annotations

import base64
import json
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
    AgentModel,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    ToolCallDelta,
    ToolDefinition,
)
from .base import (
    ProviderAuthError,
    ProviderResponseError,
    _parse_tool_calls,
    _reasoning_effort_to_budget,
    _to_model_data,
    _ToolCallAccumulator,
)


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


def _to_anthropic_content_blocks(message: Any) -> list[dict[str, Any]]:
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
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
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
                case "user" | "agent":
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
