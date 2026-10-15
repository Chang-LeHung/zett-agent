from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..messages import AssistantMessage, ImageBytesSource, ImageContent, ImageUrlSource, ToolCall
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    TimeoutOptions,
    ToolCallDelta,
    ToolDefinition,
    validate_response,
    validate_retry,
    validate_timeout,
)
from .base import (
    ProviderAuthError,
    ProviderResponseError,
    RetryingProvider,
    guard_idle,
    http_error,
    httpx_timeout,
    reject_local_tool_search,
    retry_model_stream,
    validate_extra_headers,
    validate_extra_query,
)
from .http_errors import failure_message, read_failure
from .tool_images import expand_tool_images

if TYPE_CHECKING:
    import httpx


def _tools_to_ollama_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    reject_local_tool_search("Ollama chat", tools)
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


#: A local model may have to be loaded from disk before it answers, which can take minutes for a
#: large one, so the wait for the first chunk is longer than for a hosted provider.
OLLAMA_TIMEOUT_OPTIONS = TimeoutOptions(first_byte=600.0)


class OllamaProvider(RetryingProvider):
    """Stream Ollama SDK events without buffering the entire HTTP response.

    Args:
        model: Model name installed or available on the target Ollama server.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: Ollama server root; defaults to localhost port 11434.
        response: Must remain false; this adapter uses Ollama's chat endpoint.
        retry: Backoff policy for failures before the first streamed event.
        timeout: How long the server may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`. The default allows ten minutes for the
            first chunk, because a local model may need to be loaded first. A timeout raises
            :class:`~zett_agent.providers.base.ProviderTimeoutError` and is not retried.
        extra_headers: HTTP headers sent on every request, such as the
            ``Authorization`` a reverse proxy in front of Ollama expects.
        extra_query: URL query parameters sent on every request.

    Note:
        This adapter does not install models or start the server. Reasoning,
        vision, and tool support depend on the selected model. Close with aclose().

    Examples:
        Connect to a locally running Ollama service::

            model = OllamaProvider(model="qwen3", base_url="http://127.0.0.1:11434")
            try:
                client = await create_agent(model)
                reply = await client.run("List the public classes")
            finally:
                await model.aclose()

    .. note::
        ``base_url`` selects an existing service. Constructing this adapter does
        not download a model or launch the Ollama daemon.

    .. seealso::
        :doc:`/learn/providers` covers shared adapter ownership and retries;
        :class:`~zett_agent.model.AgentModel` defines the normalized stream contract.
    """

    def __init__(
        self,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str = "http://localhost:11434",
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        timeout: TimeoutOptions = OLLAMA_TIMEOUT_OPTIONS,
    ) -> None:
        import truststore
        from ollama import AsyncClient

        validate_response(response)
        if response:
            raise ValueError("OllamaProvider does not support the Responses API")
        validate_retry(retry)
        validate_timeout(timeout)
        self.retry = retry
        self.timeout = timeout
        self.response = response
        self.model = model
        self.extra_headers = validate_extra_headers(extra_headers)
        self.extra_query = validate_extra_query(extra_query)
        self._client = AsyncClient(
            host=base_url,
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
            timeout=httpx_timeout(timeout),
            headers=self.extra_headers or None,
            params=self.extra_query or None,
        )

    async def aclose(self) -> None:
        """Close the SDK-owned HTTP connection pool."""
        await self._client._client.aclose()

    @retry_model_stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if request.server_tools:
            requested = ", ".join(tool.type for tool in request.server_tools)
            raise ProviderResponseError(f"Ollama does not support provider-hosted server tools: {requested}")
        messages: list[dict[str, Any]] = []
        for message in expand_tool_images(request.messages):
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
        from ollama import ResponseError

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
            # The client only sends its request when the first chunk is asked for, so the first
            # wait covers connecting, the response headers and the first chunk, and an HTTP
            # failure surfaces here rather than when the stream is opened.
            async for chunk in guard_idle(stream, self.timeout, first=self.timeout.first_byte):
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
                    # Ollama reports prompt evaluation counts only; it never
                    # reports prompt-cache counters, so the hit rate is unknown
                    # rather than zero.
                    usage = ModelUsage(
                        input_tokens=chunk.prompt_eval_count or 0,
                        output_tokens=chunk.eval_count or 0,
                        cache_reported=False,
                    )
        except json.JSONDecodeError as error:
            # Ollama streams one JSON object per line. Anything else is not an Ollama server: a
            # proxy's page, a login redirect, or a different service on that port.
            raise ProviderResponseError(
                "ollama answered with something that is not a stream of JSON lines. A proxy, login page or "
                "another service on that address is the usual cause; check base_url."
            ) from error
        except ResponseError as error:
            status = int(getattr(error, "status_code", 0) or 0)
            raw = getattr(error, "error", None)
            failure = read_failure(content_type=None, text=raw if isinstance(raw, str) else json.dumps(raw), parsed=raw)
            message = failure_message("ollama", status, failure)
            if status in (401, 403):
                raise ProviderAuthError(f"Provider rejected API credential. {message}") from error
            raise http_error(status, failure, message) from error
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
