from __future__ import annotations

import base64
import json
import ssl
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

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
    DEFAULT_TIMEOUT_OPTIONS,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RetryOptions,
    ServerToolCall,
    ServerToolResult,
    TimeoutOptions,
    ToolCallDelta,
    validate_response,
    validate_retry,
    validate_timeout,
)
from ..stop import StopInfo, StopReason, classify_stop
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderResponseError,
    RetryingProvider,
    _reasoning_effort_to_budget,
    guard_idle,
    http_error,
    reject_local_tool_search,
    retry_model_stream,
    validate_extra_body,
    validate_extra_headers,
)
from .http_errors import HttpFailure, failure_message, read_failure
from .stop_reasons import withhold_replay_blocks
from .tool_images import expand_tool_images

if TYPE_CHECKING:
    import httpx


class GoogleProvider(RetryingProvider):
    """Map official Google GenAI SDK streams and preserve signed replay parts.

    Args:
        model: Google model identifier used for content generation.
        api_key: API credential supplied by the application.
        transport: Optional HTTPX transport for test isolation or custom routing.
        response: Must remain false; this adapter uses Google GenerateContent.
        retry: Retry/backoff policy, applied before any model event is emitted.
        timeout: How long the provider may stay silent before the request is given up on.
            See :class:`~zett_agent.model.TimeoutOptions`. The Google client sends its request
            when the first chunk is asked for, so ``first_byte`` covers connecting, the response
            headers and the first chunk. A timeout raises
            :class:`~zett_agent.providers.base.ProviderTimeoutError` and is not retried.
        base_url: API root; None uses Google's public Generative Language endpoint.
            Set it to reach a proxy, a gateway, or a local test server.
        extra_headers: HTTP headers sent on every request, for gateways that
            route or bill on a header.
        extra_body: JSON fields merged into every request body, for options the
            adapter does not model.

    Note:
        ModelResponse retains model-scoped replay parts for subsequent calls.
        Await aclose() to release the SDK and injected HTTP client.

    Examples:
        Configure the SDK adapter without exposing the key in source::

            model = GoogleProvider(
                model=os.environ["GOOGLE_MODEL"],
                api_key=os.environ["GOOGLE_API_KEY"],
            )
            try:
                client = await create_agent(model)
                reply = await client.run("Describe the attached image")
            finally:
                await model.aclose()

    .. note::
        Provider-specific signed parts are retained in ``replay_blocks`` and
        reused only with the same Google model.

    .. seealso::
        :class:`~zett_agent.messages.AssistantMessage` describes model-scoped replay data,
        and :doc:`/learn/providers` shows the common provider workflow.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        base_url: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
    ) -> None:
        import httpx
        import truststore
        from google import genai
        from google.genai import types

        validate_response(response)
        if response:
            raise ValueError("GoogleProvider does not support the Responses API")
        if not api_key:
            raise ValueError("api_key is required")
        validate_retry(retry)
        validate_timeout(timeout)
        self.retry = retry
        self.timeout = timeout
        self.response = response
        self.model = model
        self.extra_headers = validate_extra_headers(extra_headers)
        self.extra_body = validate_extra_body(extra_body)
        self._http_client = httpx.AsyncClient(
            transport=transport,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                headers=self.extra_headers or None,
                extra_body=self.extra_body or None,
                httpx_async_client=self._http_client,
                retry_options=types.HttpRetryOptions(attempts=1),
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
        from google.genai.errors import APIError, UnknownApiResponseError

        system: list[str] = []
        contents: list[types.Content] = []
        previous_role = ""
        for message in expand_tool_images(request.messages):
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
                    if message.provider == "google" and message.model == self.model and message.replay_blocks:
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
                    # Gemini documents ``error`` as the key that marks a failed call.
                    parts = [
                        types.Part.from_function_response(
                            name=message.name,
                            response={"content": message.content} if message.success else {"error": message.content},
                        )
                    ]
                    # The service matches the responses of parallel calls as one
                    # turn: its count must equal the calls in the model turn, so
                    # adjacent results join instead of becoming separate turns.
                    if contents and previous_role == "tool":
                        contents[-1].parts.extend(parts)
                        continue
            if parts:
                contents.append(types.Content(role="model" if message.role == "assistant" else "user", parts=parts))
            previous_role = message.role
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system) or None,
            tools=self._google_tools(types, request),
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
        block_reason: str | None = None
        block_message: str | None = None
        server_tool_sequence = 0
        seen_server_tool_metadata: set[str] = set()
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
        except APIError as error:
            raise _provider_error(error) from error
        try:
            # The SDK sends the HTTP request when the first chunk is requested,
            # so a rejected request surfaces here rather than above.
            # The SDK only sends the request when the first chunk is asked for, so the first
            # wait covers connecting, the response headers and the first chunk.
            async for chunk in guard_idle(stream, self.timeout, first=self.timeout.first_byte):
                feedback = getattr(chunk, "prompt_feedback", None)
                if feedback is not None and getattr(feedback, "block_reason", None):
                    # The prompt was blocked before the model wrote anything. There is no
                    # candidate, so this is the only place the reason is reported.
                    block_reason = str(getattr(feedback.block_reason, "value", feedback.block_reason))
                    block_message = getattr(feedback, "block_reason_message", None) or None
                for candidate in (chunk.candidates or [])[:1]:
                    server_events, server_tool_sequence = _google_server_tool_events(
                        candidate,
                        sequence=server_tool_sequence,
                        seen=seen_server_tool_metadata,
                    )
                    for server_event in server_events:
                        yield server_event
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
                        # Implicit caching reports the field; a response without
                        # it says nothing about cache usage.
                        cache_reported=info.cached_content_token_count is not None,
                    )
        except APIError as error:
            raise _provider_error(error) from error
        except UnknownApiResponseError as error:
            # The SDK could not parse a 200 as JSON: a proxy's page, or a gateway that is not Gemini.
            raise _unparsable_response(error) from error
        finally:
            await stream.aclose()
        if finish_reason is None and block_reason is not None:
            # Answered as a refusal, not raised: the conversation goes on and the host is
            # told which safety system blocked the prompt.
            yield ModelEvent.completed(
                ModelResponse(
                    AssistantMessage(content="", provider="google", model=self.model),
                    finish_reason=block_reason,
                    usage=usage,
                    stop=StopInfo(
                        StopReason.REFUSAL,
                        raw=block_reason,
                        category=block_reason,
                        message=block_message,
                        detail={"blocked": "prompt"},
                    ),
                )
            )
            return
        if finish_reason is None:
            raise ProviderResponseError("Google stream ended without a finish reason")
        stop = classify_stop(finish_reason, has_tool_calls=bool(calls))
        if stop.reason in (StopReason.REFUSAL, StopReason.INTERRUPTED):
            # The calls are withheld from the message, so the parts that carry them go too.
            replay = list(withhold_replay_blocks(replay, lambda part: "function_call" in part))
        yield ModelEvent.completed(
            ModelResponse(
                AssistantMessage(
                    content=text,
                    reasoning=reasoning or None,
                    tool_calls=tuple(calls),
                    provider="google",
                    model=self.model,
                    replay_blocks=tuple(replay),
                ),
                finish_reason=finish_reason,
                usage=usage,
                stop=stop,
            )
        )

    @staticmethod
    def _google_tools(types: Any, request: ModelRequest) -> list[Any] | None:
        """Build local function declarations and native Google server tools."""
        reject_local_tool_search("Google generative AI", request.tools)
        tools: list[Any] = []
        if request.tools:
            tools.append(
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
            )
        for tool in request.server_tools:
            try:
                tools.append(types.Tool.model_validate({tool.type: dict(tool.configuration)}))
            except ValueError as error:
                raise ProviderResponseError(f"Unsupported Google server tool type: {tool.type}") from error
        return tools or None


def _provider_error(error: Any) -> ProviderError:
    """Map an SDK failure to the provider error hierarchy.

    The original error stays reachable through ``__cause__``, which is where the retry
    policy reads the HTTP status, so a 429 or 5xx is still retried. A body that is a web
    page (a proxy in front of the API) is described as one instead of being pasted in.
    """
    status = int(getattr(error, "code", 0) or 0)
    text = str(error.message) if getattr(error, "message", None) else str(error)
    failure = read_failure(content_type=None, text=text)
    message = (
        failure_message("google", status, failure) if failure.is_html else f"google returned HTTP {status}: {text}"
    )
    if status in (401, 403):
        return ProviderAuthError(f"Provider rejected API credential. {message}")
    return http_error(status, failure if failure.is_html else HttpFailure(text, False, None, text), message)


def _unparsable_response(error: BaseException) -> ProviderResponseError:
    """Describe a 200 response the SDK could not read as JSON, naming a web page when it was one."""
    text = str(error)
    _, _, raw = text.partition("Raw response:")
    failure = read_failure(content_type=None, text=(raw or text).strip())
    if failure.is_html:
        return http_error(200, failure, failure_message("google", 200, failure))
    return ProviderResponseError(
        f"google answered HTTP 200 with something that is not a Gemini response: {failure.preview}"
    )


def _google_server_tool_events(candidate: Any, *, sequence: int, seen: set[str]) -> tuple[list[ModelEvent], int]:
    """Normalize completed Google Search and URL Context metadata.

    Generate Content exposes these hosted operations only when result metadata
    reaches the candidate, rather than as a live call lifecycle. The adapter
    therefore emits adjacent start/completion events at that boundary and
    deduplicates metadata repeated by later streaming chunks.
    """
    events: list[ModelEvent] = []
    grounding = getattr(candidate, "grounding_metadata", None)
    if grounding is not None:
        payload = grounding.model_dump(exclude_none=True)
        # Grounding metadata may be repeated or enriched by later stream chunks;
        # it still represents one hosted search operation for this model step.
        fingerprint = "google_search"
        if fingerprint not in seen:
            seen.add(fingerprint)
            call = ServerToolCall(
                id=f"google-search-{sequence}",
                name="google_search",
                input={"queries": list(payload.get("web_search_queries", ()))},
            )
            sequence += 1
            events.extend(
                (
                    ModelEvent.server_tool_started(call),
                    ModelEvent.server_tool_completed(ServerToolResult(call.id, call.name, payload)),
                )
            )

    url_context = getattr(candidate, "url_context_metadata", None)
    if url_context is not None:
        metadata = url_context.model_dump(exclude_none=True)
        for item in metadata.get("url_metadata", ()):
            url = str(item.get("retrieved_url", ""))
            fingerprint = f"url_context:{url or json.dumps(item, sort_keys=True, default=str)}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            raw_status = item.get("url_retrieval_status", "URL_RETRIEVAL_STATUS_UNSPECIFIED")
            status = str(getattr(raw_status, "value", raw_status))
            call = ServerToolCall(id=f"url-context-{sequence}", name="url_context", input={"url": url})
            sequence += 1
            result = ServerToolResult(
                call.id,
                call.name,
                item,
                None if status == "URL_RETRIEVAL_STATUS_SUCCESS" else status,
            )
            events.extend(
                (
                    ModelEvent.server_tool_started(call),
                    ModelEvent.server_tool_completed(result)
                    if result.error_code is None
                    else ModelEvent.server_tool_failed(result),
                )
            )
    return events, sequence
