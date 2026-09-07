from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import truststore

from ..messages import AssistantMessage, ImageBytesSource, ImageContent, ImageUrlSource, ToolCall
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    ToolCallDelta,
    ToolDefinition,
    validate_retry,
)
from .base import ProviderResponseError, RetryingProvider, retry_model_stream


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


class OllamaProvider(RetryingProvider):
    """Stream official Ollama SDK events without buffering the entire HTTP response."""

    def __init__(
        self,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str = "http://localhost:11434",
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
    ) -> None:
        from ollama import AsyncClient

        validate_retry(retry)
        self.retry = retry
        self.model = model
        self._client = AsyncClient(
            host=base_url,
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )

    async def aclose(self) -> None:
        """Close the SDK-owned HTTP connection pool."""
        await self._client._client.aclose()

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            match message.role:
                case "agent":
                    messages.append({"role": "user", "content": message.content})
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
                AssistantMessage(
                    content=text,
                    reasoning=reasoning or None,
                    tool_calls=tuple(calls),
                    provider="ollama",
                    model=self.model,
                ),
                finish_reason=finish_reason,
                usage=usage,
            )
        )
