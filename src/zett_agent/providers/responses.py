"""OpenAI Responses API request and streaming-event normalization."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter, ValidationError

from ..messages import (
    AnyMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ..model import (
    ModelEvent,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    ServerToolCall,
    ServerToolDefinition,
    ServerToolResult,
    ToolCallDelta,
    ToolDefinition,
)
from .tool_images import expand_tool_images

if TYPE_CHECKING:
    from openai.types.responses import FunctionToolParam, ToolParam


@lru_cache(maxsize=1)
def _searched_tools() -> TypeAdapter[list[FunctionToolParam]]:
    """Validate the definitions a client-side tool search returned.

    The adapter is built on first use because its schema comes from the OpenAI
    SDK, which this module imports only when a request needs it.
    """
    from openai.types.responses import FunctionToolParam

    return TypeAdapter(list[FunctionToolParam])


def responses_input(
    messages: Sequence[AnyMessage],
    *,
    provider: str,
    model: str,
    local_search_tool: str | None = None,
) -> list[dict[str, Any]]:
    """Render complete local history as stateless Responses API input items."""
    items: list[dict[str, Any]] = []
    for message in expand_tool_images(messages):
        match message.role:
            case "system":
                items.append({"role": "system", "content": message.content})
            case "agent":
                items.append({"role": "user", "content": message.content})
            case "user":
                if not isinstance(message, UserMessage):
                    raise TypeError(f"Invalid user message: {type(message)!r}")
                items.append({"role": "user", "content": _responses_user_content(message)})
            case "assistant":
                if not isinstance(message, AssistantMessage):
                    raise TypeError(f"Invalid assistant message: {type(message)!r}")
                if message.provider == provider and message.model == model and message.replay_blocks:
                    items.extend(dict(block) for block in message.replay_blocks)
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
            case "tool":
                if not isinstance(message, ToolMessage):
                    raise TypeError(f"Invalid tool message: {type(message)!r}")
                if local_search_tool is not None and message.name == local_search_tool:
                    items.append(_tool_search_output(message))
                    continue
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
    return items


def local_tool_search_name(tools: Sequence[ToolDefinition]) -> str | None:
    """Return the name of the tool that answers client-side tool search."""
    return next((tool.name for tool in tools if tool.local_tool_search), None)


def _tool_search_output(message: ToolMessage) -> dict[str, Any]:
    """Render one local search result as the client's ``tool_search_output``."""
    if not message.success:
        raise ValueError(f"Local tool search failed: {message.content}")
    try:
        tools = json.loads(message.content)
    except json.JSONDecodeError as error:
        raise ValueError("A local tool search tool must return JSON tool definitions") from error
    try:
        definitions = _searched_tools().validate_python(tools)
    except ValidationError as error:
        raise ValueError(f"A local tool search tool returned invalid FunctionToolParam definitions: {error}") from error
    return {
        "type": "tool_search_output",
        "call_id": message.tool_call_id,
        "execution": "client",
        "status": "completed",
        "tools": definitions,
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
        case ReasoningEffort.XHIGH:
            return {"effort": "high", "summary": "auto"}
        case _:
            return {"effort": effort.value, "summary": "auto"}


async def stream_responses(
    stream: Any,
    *,
    provider: str,
    model: str,
    local_search_tool: str | None = None,
) -> AsyncIterator[ModelEvent]:
    """Convert one OpenAI-compatible Responses stream to normalized events."""
    text = ""
    reasoning = ""
    terminal: Any = None
    server_calls: dict[str, ServerToolCall] = {}

    async for event in stream:
        event_type = str(getattr(event, "type", ""))
        match event_type:
            case "response.output_text.delta":
                delta = str(getattr(event, "delta", ""))
                if delta:
                    text += delta
                    yield ModelEvent.text(delta)
            case "response.reasoning_text.delta" | "response.reasoning_summary_text.delta":
                delta = str(getattr(event, "delta", ""))
                if delta:
                    reasoning += delta
                    yield ModelEvent.reasoning(delta)
            case "response.output_item.added":
                item = _object_mapping(getattr(event, "item", None))
                item_type = str(item.get("type", ""))
                if item_type == "function_call":
                    yield ModelEvent.tool_call(_function_call_start(item, int(getattr(event, "output_index", 0))))
                elif _is_client_tool_search(item) and local_search_tool is not None:
                    yield ModelEvent.tool_call(
                        _client_tool_search_start(item, int(getattr(event, "output_index", 0)), local_search_tool)
                    )
                elif _is_server_tool_item(item_type):
                    call = _server_tool_call(item)
                    server_calls[call.id] = call
                    yield ModelEvent.server_tool_started(call)
            case "response.function_call_arguments.delta":
                yield ModelEvent.tool_call(
                    ToolCallDelta(
                        index=int(getattr(event, "output_index", 0)),
                        arguments_delta=str(getattr(event, "delta", "")),
                    )
                )
            case "response.output_item.done":
                item = _object_mapping(getattr(event, "item", None))
                item_type = str(item.get("type", ""))
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
                terminal = getattr(event, "response", None)
            case "response.failed":
                response = _object_mapping(getattr(event, "response", None))
                error = _object_mapping(response.get("error"))
                message = error.get("message") or "Responses API stream failed"
                raise RuntimeError(str(message))

    if terminal is None:
        raise RuntimeError("Responses API stream ended without a terminal response")
    payload = _object_mapping(terminal)
    output = [_object_mapping(item) for item in payload.get("output", ())]
    if not text:
        text = _response_text(output)
    if not reasoning:
        reasoning = _response_reasoning(output)
    calls = tuple(
        call for call in (_local_or_function_call(item, local_search_tool) for item in output) if call is not None
    )
    yield ModelEvent.completed(
        ModelResponse(
            message=AssistantMessage(
                content=text,
                reasoning=reasoning or None,
                tool_calls=calls,
                provider=provider,
                model=model,
                replay_blocks=tuple(output),
            ),
            finish_reason=str(payload.get("status")) if payload.get("status") is not None else None,
            usage=_responses_usage(_object_mapping(payload.get("usage"))),
        )
    )


def _responses_user_content(message: UserMessage) -> str | list[dict[str, Any]]:
    if isinstance(message.content, str):
        return message.content
    content: list[dict[str, Any]] = []
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


def _object_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)  # type: ignore[no-any-return]
    raise TypeError(f"Responses API object did not normalize to a mapping: {value!r}")


def _function_call_start(item: Mapping[str, Any], index: int) -> ToolCallDelta:
    return ToolCallDelta(
        index=index,
        id_delta=str(item.get("call_id") or item.get("id") or f"call_{index}"),
        name_delta=str(item.get("name") or ""),
    )


def _response_tool_call(item: Mapping[str, Any]) -> ToolCall:
    arguments = item.get("arguments") or "{}"
    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(parsed, dict):
        raise TypeError("Responses function-call arguments must be an object")
    return ToolCall(
        id=str(item.get("call_id") or item.get("id")),
        name=str(item.get("name")),
        arguments=parsed,
    )


def _response_text(output: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        str(part.get("text", ""))
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", ())
        if isinstance(part, Mapping) and part.get("type") == "output_text"
    )


def _response_reasoning(output: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for item in output:
        if item.get("type") != "reasoning":
            continue
        parts = item.get("content") or item.get("summary") or ()
        rendered.extend(str(part.get("text", "")) for part in parts if isinstance(part, Mapping))
    return "".join(rendered)


def _responses_usage(payload: Mapping[str, Any]) -> ModelUsage:
    input_details = _object_mapping(payload.get("input_tokens_details"))
    output_details = _object_mapping(payload.get("output_tokens_details"))
    cache_reported = any(key in input_details for key in ("cached_tokens", "cache_write_tokens"))
    return ModelUsage(
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cache_read_tokens=int(input_details.get("cached_tokens", 0) or 0),
        cache_write_tokens=int(input_details.get("cache_write_tokens", 0) or 0),
        cache_reported=cache_reported,
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
    )


def _is_server_tool_item(item_type: str) -> bool:
    return item_type.endswith("_call") and item_type not in {"function_call", "custom_tool_call"}


def _is_client_tool_search(item: Mapping[str, Any]) -> bool:
    """Whether one provider item asks this client to run a tool search."""
    return item.get("type") == "tool_search_call" and item.get("execution") == "client"


def _client_tool_search_start(item: Mapping[str, Any], index: int, tool_name: str) -> ToolCallDelta:
    """Expose one client search request as an ordinary local tool call."""
    return ToolCallDelta(
        index=index,
        id_delta=str(item.get("call_id") or item.get("id") or ""),
        name_delta=tool_name,
        arguments_delta=json.dumps(item.get("arguments") or {}, ensure_ascii=False),
    )


def _client_tool_search_call(item: Mapping[str, Any], tool_name: str) -> ToolCall:
    """Complete one client search request as an ordinary local tool call."""
    arguments = item.get("arguments")
    return ToolCall(
        id=str(item.get("call_id") or item.get("id") or ""),
        name=tool_name,
        arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
    )


def _local_or_function_call(item: Mapping[str, Any], local_search_tool: str | None) -> ToolCall | None:
    """Normalize one response output item into a local tool call when it is one."""
    if item.get("type") == "function_call":
        return _response_tool_call(item)
    if local_search_tool is not None and _is_client_tool_search(item):
        return _client_tool_search_call(item, local_search_tool)
    return None


def _server_tool_call(item: Mapping[str, Any]) -> ServerToolCall:
    item_type = str(item.get("type", "server_tool_call"))
    name = str(item.get("name") or item_type.removesuffix("_call"))
    input_value = item.get("action", item.get("input"))
    input_mapping = dict(input_value) if isinstance(input_value, Mapping) else None
    return ServerToolCall(id=str(item.get("id") or item.get("call_id")), name=name, input=input_mapping)


def _server_tool_error(item: Mapping[str, Any]) -> str | None:
    status = str(item.get("status", "completed"))
    if status not in {"failed", "incomplete"}:
        return None
    error = item.get("error")
    if isinstance(error, Mapping):
        return str(error.get("code") or status)
    return status
