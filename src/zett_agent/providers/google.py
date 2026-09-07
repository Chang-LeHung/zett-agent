from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator
from typing import Any

import httpx
import truststore

from ..messages import (
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
    TextContent,
    ToolCall,
)
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RetryOptions,
    ToolCallDelta,
    validate_retry,
)
from .base import ProviderResponseError, RetryingProvider, _reasoning_effort_to_budget, retry_model_stream


class GoogleProvider(RetryingProvider):
    """Map official Google GenAI SDK streams and preserve signed replay parts."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
    ) -> None:
        from google import genai
        from google.genai import types

        if not api_key:
            raise ValueError("api_key is required")
        validate_retry(retry)
        self.retry = retry
        self.model = model
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                httpx_async_client=self._http_client, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    async def aclose(self) -> None:
        """Release both SDK clients and the injected HTTP connection pool."""
        await self._client.aio.aclose()
        self._client.close()
        await self._http_client.aclose()

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        from google.genai import types

        system: list[str] = []
        contents: list[types.Content] = []
        for message in request.messages:
            match message.role:
                case "agent":
                    parts = [types.Part.from_text(text=message.content)]
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
