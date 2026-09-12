"""OpenAI Responses API request and streaming-event normalization."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

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


def responses_input(messages: Sequence[AnyMessage], *, provider: str, model: str) -> list[dict[str, Any]]:
    """Render complete local history as stateless Responses API input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
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
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
    return items


def responses_tools(
    tools: Sequence[ToolDefinition], server_tools: Sequence[ServerToolDefinition]
) -> list[dict[str, Any]]:
    """Render local functions and opaque hosted tools in Responses wire format."""
    return [
        *(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for tool in tools
        ),
        *({"type": tool.type, **dict(tool.configuration)} for tool in server_tools),
    ]


def responses_reasoning(effort: ReasoningEffort) -> dict[str, str] | None:
    """Map provider-neutral effort to the Responses reasoning object."""
    match effort:
        case ReasoningEffort.OFF:
            return None
        case ReasoningEffort.XHIGH:
            return {"effort": "high", "summary": "auto"}
        case _:
            return {"effort": effort.value, "summary": "auto"}


async def stream_responses(stream: Any, *, provider: str, model: str) -> AsyncIterator[ModelEvent]:
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
                if _is_server_tool_item(item_type):
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
    calls = tuple(_response_tool_call(item) for item in output if item.get("type") == "function_call")
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
    return ModelUsage(
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cache_read_tokens=int(input_details.get("cached_tokens", 0) or 0),
        cache_write_tokens=int(input_details.get("cache_write_tokens", 0) or 0),
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
    )


def _is_server_tool_item(item_type: str) -> bool:
    return item_type.endswith("_call") and item_type not in {"function_call", "custom_tool_call"}


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
