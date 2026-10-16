"""OpenAI Responses API request and streaming-event normalization.

Everything here works on JSON: events arrive as objects (the provider dumps the SDK's models at the
edge), and requests leave as plain dictionaries, so this module neither imports the SDK nor trusts the
shape of what a service sends.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from ..json_types import JsonValue
from ..messages import (
    AgentMessage,
    AnyMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
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
    ServerToolCall,
    ServerToolDefinition,
    ServerToolResult,
    ToolCallDelta,
    ToolDefinition,
)
from ..stop import StopInfo, StopReason, classify_stop
from .stop_reasons import refusal_from_response_error, withhold_replay_blocks
from .tool_arguments import tool_call_from_arguments
from .tool_images import expand_tool_images
from .wire import JsonObject, read_int, read_object, read_objects, read_str

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai.types.responses import FunctionToolParam, ToolParam
    from pydantic import TypeAdapter


@lru_cache(maxsize=1)
def _searched_tools() -> TypeAdapter[list[FunctionToolParam]]:
    """Validate the definitions a client-side tool search returned.

    The adapter is built on first use because its schema comes from the OpenAI
    SDK, which this module imports only when a request needs it.
    """
    from openai.types.responses import FunctionToolParam
    from pydantic import TypeAdapter

    return TypeAdapter(list[FunctionToolParam])


def responses_input(
    messages: Sequence[AnyMessage],
    *,
    provider: str,
    model: str,
    local_search_tool: str | None = None,
) -> list[JsonObject]:
    """Render complete local history as stateless Responses API input items."""
    items: list[JsonObject] = []
    for message in expand_tool_images(messages):
        match message:
            case SystemMessage() | AgentMessage():
                role = "system" if isinstance(message, SystemMessage) else "user"
                items.append({"role": role, "content": message.content})
            case UserMessage():
                items.append({"role": "user", "content": _responses_user_content(message)})
            case AssistantMessage():
                if message.provider == provider and message.model == model and message.replay_blocks:
                    items.extend(cast(JsonObject, dict(block)) for block in message.replay_blocks)
                    continue
                if message.content:
                    items.append({"role": "assistant", "content": message.content})
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    }
                    for call in message.tool_calls
                )
            case ToolMessage():
                if local_search_tool is not None and message.name == local_search_tool:
                    items.append(_tool_search_output(message))
                    continue
                items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": message.text})
    return items


def local_tool_search_name(tools: Sequence[ToolDefinition]) -> str | None:
    """Return the name of the tool that answers client-side tool search."""
    return next((tool.name for tool in tools if tool.local_tool_search), None)


def _tool_search_output(message: ToolMessage) -> JsonObject:
    """Render one local search result as the client's ``tool_search_output``."""
    if not message.success:
        raise ValueError(f"Local tool search failed: {message.content}")
    try:
        tools = json.loads(message.text)
    except json.JSONDecodeError as error:
        raise ValueError("A local tool search tool must return JSON tool definitions") from error
    from pydantic import ValidationError

    try:
        definitions = _searched_tools().validate_python(tools)
    except ValidationError as error:
        raise ValueError(f"A local tool search tool returned invalid FunctionToolParam definitions: {error}") from error
    return {
        "type": "tool_search_output",
        "call_id": message.tool_call_id,
        "execution": "client",
        "status": "completed",
        "tools": cast("list[JsonValue]", definitions),
    }


def responses_tools(tools: Sequence[ToolDefinition], server_tools: Sequence[ServerToolDefinition]) -> list[ToolParam]:
    """Render functions and hosted tools in Responses wire format.

    Functions marked ``deferred`` use OpenAI's ``defer_loading`` protocol. The
    request must also expose the hosted ``tool_search`` tool so the model can
    discover and load those definitions on demand. A tool marked
    ``local_tool_search`` answers that search instead: it is not rendered as a
    function, and the request declares ``execution: "client"``.
    """
    search_tools = tuple(tool for tool in tools if tool.local_tool_search)
    if len(search_tools) > 1:
        raise ValueError("Only one local tool search tool can be registered")
    if search_tools and any(tool.type == "tool_search" for tool in server_tools):
        raise ValueError("A local tool search tool cannot be combined with a provider tool_search tool")
    rendered: list[ToolParam] = []
    deferred = False
    for tool in tools:
        if tool.local_tool_search:
            # The model never calls the search tool as a function; it only
            # reaches it through the client-side tool_search endpoint.
            continue
        payload: FunctionToolParam = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
            "strict": None,
        }
        if tool.deferred:
            payload["defer_loading"] = True
            deferred = True
        rendered.append(payload)
    rendered.extend(cast("ToolParam", {"type": tool.type, **dict(tool.configuration)}) for tool in server_tools)
    if search_tools:
        # The declared parameters are what the model fills in for each search:
        # without them the endpoint sends an empty argument object.
        rendered.append(
            {
                "type": "tool_search",
                "execution": "client",
                "description": search_tools[0].description,
                "parameters": dict(search_tools[0].parameters),
            }
        )
    elif deferred and not any(tool.type == "tool_search" for tool in server_tools):
        rendered.append({"type": "tool_search"})
    return rendered


def responses_reasoning(effort: ReasoningEffort) -> dict[str, str] | None:
    """Map provider-neutral effort to the Responses reasoning object."""
    match effort:
        case ReasoningEffort.OFF:
            return None
        # The Responses protocol tops out at "high".
        case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return {"effort": "high", "summary": "auto"}
        case _:
            return {"effort": effort.value, "summary": "auto"}


def responses_request(
    request: ModelRequest,
    *,
    provider: str,
    model: str,
    temperature: float | None,
    send_prompt_cache_key: bool,
    extra_body: Mapping[str, JsonValue],
) -> dict[str, object]:
    """Build the fields of one stateless Responses request.

    The request is never stored by the service (``store`` is false), so the whole history is sent
    every time. Fields the SDK does not declare go in ``extra_body``.
    """
    tools = responses_tools(request.tools, request.server_tools)
    fields: dict[str, object] = {
        "model": model,
        "input": responses_input(
            request.messages,
            provider=provider,
            model=model,
            local_search_tool=local_tool_search_name(request.tools),
        ),
        "stream": True,
        "store": False,
        "parallel_tool_calls": request.parallel_tool_call,
    }
    if tools:
        fields["tools"] = tools
    if temperature is not None:
        fields["temperature"] = temperature
    if request.tool_choice:
        server_types = {tool.type for tool in request.server_tools}
        fields["tool_choice"] = (
            {"type": request.tool_choice}
            if request.tool_choice in server_types
            else {"type": "function", "name": request.tool_choice}
        )
    if (reasoning := responses_reasoning(request.reasoning_effort)) is not None:
        fields["reasoning"] = reasoning
        # Stateless reasoning models only hand back the opaque reasoning state that continues a
        # chain of thought when it is requested, and the replayed reasoning items are worthless
        # without it.
        fields["include"] = ["reasoning.encrypted_content"]
    if send_prompt_cache_key and request.cache_key:
        fields["prompt_cache_key"] = request.cache_key
    if extra_body:
        fields["extra_body"] = dict(extra_body)
    return fields


class ResponsesStreamInterrupted(RuntimeError):
    """The endpoint closed a Responses stream before its terminal event.

    The Responses protocol ends every stream with ``response.completed``,
    ``response.incomplete``, ``response.failed``, or an ``error`` event. Seeing
    none of them means the connection was cut, usually by a gateway or proxy,
    rather than that the model finished. Nothing was lost if no event had been
    emitted yet, so the provider retries that case.
    """


def _stream_error_message(event: Mapping[str, JsonValue]) -> str:
    """Render an ``error`` stream event, which carries ``code`` and ``message`` directly."""
    code = event.get("code")
    message = read_str(event.get("message")) or "Responses API stream failed"
    return f"Responses API error ({code}): {message}" if code else f"Responses API error: {message}"


async def stream_responses(
    stream: AsyncIterable[JsonObject],
    *,
    provider: str,
    model: str,
    local_search_tool: str | None = None,
) -> AsyncIterator[ModelEvent]:
    """Convert one OpenAI-compatible Responses stream, as JSON events, to normalized events."""
    text = ""
    reasoning = ""
    refusal_text = ""
    terminal: JsonObject | None = None
    server_calls: dict[str, ServerToolCall] = {}
    #: Output items by ``output_index``: opened by ``output_item.added``, closed by ``.done``.
    open_items: set[int] = set()
    done_items: dict[int, JsonObject] = {}
    last_event = ""
    #: A hosted item reported its own terminal failure, so a stream ending here is not a dropped one.
    server_ended = False

    async for event in stream:
        event_type = read_str(event.get("type"))
        last_event = event_type or last_event
        match event_type:
            case "response.output_text.delta":
                delta = read_str(event.get("delta"))
                if delta:
                    text += delta
                    yield ModelEvent.text(delta)
            case "response.reasoning_text.delta" | "response.reasoning_summary_text.delta":
                delta = read_str(event.get("delta"))
                if delta:
                    reasoning += delta
                    yield ModelEvent.reasoning(delta)
            case "response.refusal.delta":
                # A refusal is the model's answer, so it is shown like any other text, and it is
                # also what the stop explains itself with.
                delta = read_str(event.get("delta"))
                if delta:
                    text += delta
                    refusal_text += delta
                    yield ModelEvent.text(delta)
            case "response.output_item.added":
                item = read_object(event.get("item"))
                item_type = read_str(item.get("type"))
                item_index = read_int(event.get("output_index"))
                open_items.add(item_index)
                if item_type == "function_call":
                    yield ModelEvent.tool_call(_function_call_start(item, item_index))
                elif _is_client_tool_search(item) and local_search_tool is not None:
                    yield ModelEvent.tool_call(_client_tool_search_start(item, item_index, local_search_tool))
                elif _is_server_tool_item(item_type):
                    call = _server_tool_call(item)
                    server_calls[call.id] = call
                    yield ModelEvent.server_tool_started(call)
            case "response.function_call_arguments.delta":
                yield ModelEvent.tool_call(
                    ToolCallDelta(
                        index=read_int(event.get("output_index")),
                        arguments_delta=read_str(event.get("delta")),
                    )
                )
            case "response.output_item.done":
                item = read_object(event.get("item"))
                item_type = read_str(item.get("type"))
                item_index = read_int(event.get("output_index"))
                open_items.discard(item_index)
                done_items[item_index] = item
                if _is_server_tool_item(item_type) and not (
                    _is_client_tool_search(item) and local_search_tool is not None
                ):
                    call = _server_tool_call(item)
                    original = server_calls.get(call.id)
                    if original is None:
                        original = call
                        server_calls[call.id] = call
                        yield ModelEvent.server_tool_started(call)
                    result = ServerToolResult(
                        call_id=original.id,
                        name=original.name,
                        output=item,
                        error_code=_server_tool_error(item),
                    )
                    if result.error_code:
                        yield ModelEvent.server_tool_failed(result)
                    else:
                        yield ModelEvent.server_tool_completed(result)
            case "response.completed" | "response.incomplete":
                response_payload = event.get("response")
                terminal = response_payload if isinstance(response_payload, dict) else None
            case "response.failed":
                response = read_object(event.get("response"))
                error = read_object(response.get("error"))
                if (refusal := refusal_from_response_error(error)) is not None:
                    # A policy refused the request. It is answered as a refusal, with whatever
                    # was written before it, so the host is told why instead of seeing a crash.
                    yield ModelEvent.completed(
                        ModelResponse(
                            AssistantMessage(content=text, provider=provider, model=model),
                            finish_reason=refusal.raw,
                            stop=refusal,
                        )
                    )
                    return
                raise RuntimeError(read_str(error.get("message")) or "Responses API stream failed")
            case str(kind) if _is_item_failure(kind):
                # A hosted item (an MCP server that will not answer, say) reported its own failure.
                # It is closed here so the stream can end without a terminal response and still be
                # understood as the service explaining itself rather than a dropped connection.
                open_items.discard(read_int(event.get("output_index")))
                server_ended = True
                yield ModelEvent.server_tool_failed(
                    ServerToolResult(
                        call_id=read_str(event.get("item_id")),
                        name=kind.split(".")[1],
                        output={"status": "failed"},
                        error_code="failed",
                    )
                )
            case "error":
                # Mid-stream failures (rate limits, overload, context overflow)
                # arrive as a bare ``error`` event, after which the stream just
                # ends. Without this the real cause was reported as a missing
                # terminal response.
                raise RuntimeError(_stream_error_message(event))

    if terminal is not None:
        payload = terminal
    elif server_ended and not open_items:
        # The service named a failure and stopped. Nothing was cut off, so the answer that was built
        # so far is returned with the failure in it instead of being thrown away as a broken stream.
        logger.warning("Responses stream ended after a failed hosted item (%s); no terminal response", last_event)
        finished: list[JsonValue] = [done_items[index] for index in sorted(done_items)]
        payload = {"status": "incomplete", "output": finished}
    else:
        payload = _recover_unterminated(done_items, open_items, last_event)
    output = read_objects(payload.get("output"))
    if not text:
        text = _response_text(output)
    if not reasoning:
        reasoning = _response_reasoning(output)
    reason = _finish_reason(payload)
    calls = tuple(
        local
        for index, item in enumerate(output)
        if (local := _local_or_function_call(item, local_search_tool, index=index, finish_reason=reason)) is not None
    )
    refusal_text = refusal_text or _response_refusal(output)
    stop = classify_stop(reason, has_tool_calls=bool(calls))
    if refusal_text and stop.ok:
        # A refusal content block arrives in a response whose status is ``completed``.
        stop = StopInfo(StopReason.REFUSAL, raw=reason, category="refusal", message=refusal_text)
    replay: tuple[Mapping[str, JsonValue], ...] = tuple(output)
    if stop.reason in (StopReason.REFUSAL, StopReason.INTERRUPTED):
        # The calls are withheld from the message, so the provider's record of them goes too.
        replay = withhold_replay_blocks(
            replay,
            lambda item: _local_or_function_call(item, local_search_tool) is not None,
        )
    yield ModelEvent.completed(
        ModelResponse(
            message=AssistantMessage(
                content=text,
                reasoning=reasoning or None,
                tool_calls=calls,
                provider=provider,
                model=model,
                replay_blocks=replay,
            ),
            finish_reason=reason,
            usage=_responses_usage(read_object(payload.get("usage"))),
            stop=stop,
        )
    )


def _finish_reason(payload: Mapping[str, JsonValue]) -> str | None:
    """Prefer why a response is incomplete over the bare ``incomplete`` status."""
    details = read_object(payload.get("incomplete_details"))
    reason = details.get("reason")
    if reason:
        return str(reason)
    return str(payload["status"]) if payload.get("status") is not None else None


def _recover_unterminated(done_items: Mapping[int, JsonObject], open_items: set[int], last_event: str) -> JsonObject:
    """Rebuild a response from finished output items when the terminal event is missing.

    Some gateways drop the final ``response.completed`` after forwarding every
    item. If at least one item finished and none is still open, the model's
    output is complete as far as this stream shows, so it is used instead of
    discarding a good answer. Usage is unknown and the status is marked
    ``incomplete`` so callers can tell. An item that never finished means the
    stream was cut mid-output; guessing would hand a truncated answer or tool
    call to the agent, so that case raises and the provider may retry it.
    """
    if done_items and not open_items:
        logger.warning("Responses stream ended without a terminal event; using %d finished item(s)", len(done_items))
        output: list[JsonValue] = [done_items[index] for index in sorted(done_items)]
        return {"status": "incomplete", "output": output}
    detail = f"last event: {last_event or 'none'}"
    if open_items:
        detail += f"; {len(open_items)} output item(s) never finished"
    raise ResponsesStreamInterrupted(
        f"Responses API stream ended without a terminal response ({detail}); "
        "the endpoint or a proxy closed the connection early"
    )


def _responses_user_content(message: UserMessage) -> str | list[JsonValue]:
    if isinstance(message.content, str):
        return message.content
    content: list[JsonValue] = []
    for part in message.content:
        match part:
            case TextContent(text=text):
                content.append({"type": "input_text", "text": text})
            case ImageContent(source=ImageUrlSource(url=url), detail=detail):
                content.append({"type": "input_image", "image_url": url, "detail": detail.value})
            case ImageContent(source=ImageBytesSource(data=data, media_type=media_type), detail=detail):
                encoded = base64.b64encode(data).decode("ascii")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{encoded}",
                        "detail": detail.value,
                    }
                )
    return content


def _function_call_start(item: Mapping[str, JsonValue], index: int) -> ToolCallDelta:
    return ToolCallDelta(
        index=index,
        id_delta=str(item.get("call_id") or item.get("id") or f"call_{index}"),
        name_delta=str(item.get("name") or ""),
    )


def _response_tool_call(
    item: Mapping[str, JsonValue], *, index: int = 0, finish_reason: str | None = "completed"
) -> ToolCall:
    """Decode one function call; undecodable arguments become a model-visible failure."""
    arguments = item.get("arguments") or "{}"
    call_id, name = str(item.get("call_id") or item.get("id")), str(item.get("name"))
    if isinstance(arguments, str):
        return tool_call_from_arguments(call_id, name, arguments, index=index, finish_reason=finish_reason)
    if not isinstance(arguments, Mapping):
        raise TypeError("Responses function-call arguments must be an object")
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _response_text(output: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(
        str(part.get("text") or part.get("refusal") or "")
        for item in output
        if item.get("type") == "message"
        for part in read_objects(item.get("content"))
        if part.get("type") in {"output_text", "refusal"}
    )


def _response_refusal(output: Sequence[Mapping[str, JsonValue]]) -> str:
    """Return the text of the refusal content blocks of a response, or an empty string."""
    return "".join(
        str(part.get("refusal") or "")
        for item in output
        if item.get("type") == "message"
        for part in read_objects(item.get("content"))
        if part.get("type") == "refusal"
    )


def _response_reasoning(output: Sequence[Mapping[str, JsonValue]]) -> str:
    rendered: list[str] = []
    for item in output:
        if item.get("type") != "reasoning":
            continue
        rendered.extend(str(part.get("text", "")) for part in read_objects(item.get("content") or item.get("summary")))
    return "".join(rendered)


def _responses_usage(payload: Mapping[str, JsonValue]) -> ModelUsage:
    input_details = read_object(payload.get("input_tokens_details"))
    output_details = read_object(payload.get("output_tokens_details"))
    cache_reported = any(key in input_details for key in ("cached_tokens", "cache_write_tokens"))
    return ModelUsage(
        input_tokens=read_int(payload.get("input_tokens")),
        output_tokens=read_int(payload.get("output_tokens")),
        cache_read_tokens=read_int(input_details.get("cached_tokens")),
        cache_write_tokens=read_int(input_details.get("cache_write_tokens")),
        cache_reported=cache_reported,
        reasoning_tokens=read_int(output_details.get("reasoning_tokens")),
    )


def _is_server_tool_item(item_type: str) -> bool:
    """Whether an output item is a hosted tool activity rather than a local function call.

    Most hosted calls are named ``*_call``; an MCP server names its tool listing ``mcp_list_tools``.
    """
    if item_type in {"function_call", "custom_tool_call"}:
        return False
    return item_type.endswith("_call") or item_type in {"mcp_list_tools", "mcp_approval_request"}


def _is_item_failure(kind: str) -> bool:
    """Whether a stream event is one hosted item reporting that it failed.

    A service may end such an item with ``response.<item>.failed`` and then close the stream without
    a terminal response. That is the service reporting a failure, not a connection that dropped.
    """
    return kind.startswith("response.") and kind.count(".") == 2 and kind.rsplit(".", 1)[1] in {"failed", "incomplete"}


def _is_client_tool_search(item: Mapping[str, JsonValue]) -> bool:
    """Whether one provider item asks this client to run a tool search."""
    return item.get("type") == "tool_search_call" and item.get("execution") == "client"


def _client_tool_search_start(item: Mapping[str, JsonValue], index: int, tool_name: str) -> ToolCallDelta:
    """Expose one client search request as an ordinary local tool call."""
    return ToolCallDelta(
        index=index,
        id_delta=str(item.get("call_id") or item.get("id") or ""),
        name_delta=tool_name,
        arguments_delta=json.dumps(item.get("arguments") or {}, ensure_ascii=False),
    )


def _client_tool_search_call(item: Mapping[str, JsonValue], tool_name: str) -> ToolCall:
    """Complete one client search request as an ordinary local tool call."""
    arguments = item.get("arguments")
    return ToolCall(
        id=str(item.get("call_id") or item.get("id") or ""),
        name=tool_name,
        arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
    )


def _local_or_function_call(
    item: Mapping[str, JsonValue], local_search_tool: str | None, *, index: int = 0, finish_reason: str | None = None
) -> ToolCall | None:
    """Normalize one response output item into a local tool call when it is one."""
    if item.get("type") == "function_call":
        return _response_tool_call(item, index=index, finish_reason=finish_reason)
    if local_search_tool is not None and _is_client_tool_search(item):
        return _client_tool_search_call(item, local_search_tool)
    return None


def _server_tool_call(item: Mapping[str, JsonValue]) -> ServerToolCall:
    item_type = str(item.get("type", "server_tool_call"))
    name = str(item.get("name") or item_type.removesuffix("_call"))
    input_value = item.get("action", item.get("input"))
    input_mapping = dict(input_value) if isinstance(input_value, Mapping) else None
    return ServerToolCall(id=str(item.get("id") or item.get("call_id")), name=name, input=input_mapping)


def _server_tool_error(item: Mapping[str, JsonValue]) -> str | None:
    status = str(item.get("status", "completed"))
    if status not in {"failed", "incomplete"}:
        return None
    error = item.get("error")
    if isinstance(error, Mapping):
        return str(error.get("code") or status)
    return status
