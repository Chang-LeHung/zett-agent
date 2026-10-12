"""Decode streamed tool-call arguments and explain why a call is unusable."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from ..messages import ToolCall

#: Finish reasons that mean the model ran into an output limit.
_OUTPUT_LIMIT_REASONS = frozenset({"max_tokens", "length", "max_output_tokens", "model_context_window_exceeded"})


class ToolArgumentsError(ValueError):
    """Raised when accumulated tool arguments are not a complete JSON object."""


def decode_tool_arguments(payload: str, *, index: int) -> dict[str, Any]:
    """Decode one accumulated tool argument object from a provider stream.

    Some OpenAI-compatible providers, including DeepSeek, occasionally expose
    decoded newlines or tabs inside the nested ``function.arguments`` string.
    ``strict=False`` accepts those control characters without changing valid
    JSON semantics. Models can also occasionally omit a comma or fail to escape
    a quote in a long argument string. For a complete top-level object, use the
    bounded ``json-repair`` fallback before rejecting the model response.

    The fallback deliberately requires both outer braces. A payload cut off by
    a token limit must fail instead of being completed heuristically and passed
    to a mutating tool with silently truncated content. Tool arguments must
    remain an object because the runtime invokes tools with named keyword
    arguments; the tool's Pydantic input model performs the final schema check.
    """
    try:
        parsed = json.loads(payload, strict=False) if payload else {}
    except JSONDecodeError as error:
        stripped = payload.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            raise ToolArgumentsError(f"Invalid tool-call arguments for index {index}: {error}") from error
        from json_repair import repair_json

        try:
            parsed = repair_json(stripped, return_objects=True, skip_json_loads=True)
        except (ValueError, TypeError, RecursionError) as repair_error:
            raise ToolArgumentsError(
                f"Invalid tool-call arguments for index {index}: {error}; repair failed: {repair_error}"
            ) from error
    if not isinstance(parsed, dict):
        raise ToolArgumentsError(f"Tool-call arguments for index {index} must be a JSON object")
    return parsed


def invalid_arguments_message(name: str, detail: str, finish_reason: str | None) -> str:
    """Tell the model why its call was not run and how to recover.

    The text is returned as a failed tool result, so it speaks to the model:
    what happened, that nothing ran, and the cheapest way to try again.
    """
    reason = (finish_reason or "").lower()
    if reason in _OUTPUT_LIMIT_REASONS:
        cause = (
            "The response hit the model's output limit, so the arguments were cut off. "
            "Retry with a much smaller call, for example several short edits instead of one large write."
        )
    elif not reason or reason == "incomplete":
        cause = (
            "The response stream ended before the model finished, so the arguments are incomplete. "
            "Retry the same call, keeping the arguments short."
        )
    else:
        cause = "Resend the call with a complete, valid JSON object of arguments."
    return f"Invalid arguments for tool {name!r}: {detail}. The tool was not run. {cause}"


def tool_call_from_arguments(
    call_id: str,
    name: str,
    payload: str,
    *,
    index: int,
    finish_reason: str | None,
) -> ToolCall:
    """Build a call whose broken arguments become a model-visible failure.

    Failing the whole request for one malformed call discards a long answer and
    every other tool call in the same response. The call is instead flagged with
    ``error`` so the Agent reports it as a failed tool result and the model gets
    a chance to resend it. The tool itself never runs.
    """
    try:
        return ToolCall(id=call_id, name=name, arguments=decode_tool_arguments(payload, index=index))
    except ToolArgumentsError as error:
        return ToolCall(id=call_id, name=name, error=invalid_arguments_message(name, str(error), finish_reason))
