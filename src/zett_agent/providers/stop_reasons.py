"""How provider-specific refusals and stops become a :class:`~zett_agent.stop.StopInfo`.

A stop reason reaches an adapter through four channels, and each one needs its own
reading before it can be normalized:

* a ``finish_reason`` or ``stop_reason`` value, which :func:`~zett_agent.stop.classify_stop`
  maps for every provider;
* a refusal carried in the content (OpenAI's ``refusal`` field and content part);
* a request that was refused before any output, which arrives as an HTTP error or a
  failed response rather than as a stop (Kimi's ``400 content_filter``, an OpenRouter
  moderation ``403``, Google's blocked prompt, an OpenAI policy error);
* an error inside a ``200`` stream (an OpenRouter ``finish_reason: "error"`` chunk).

This module reads the last three. What it recognizes is only what each provider's
documentation says, and anything else is left as the error it was.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ..json_types import JsonValue
from ..stop import StopInfo, StopReason

__all__ = [
    "RESPONSES_POLICY_ERROR_CODES",
    "refusal_from_response_error",
    "refusal_from_status_error",
    "withhold_replay_blocks",
]

#: ``error.code`` values of an OpenAI Responses ``response.failed`` that mean a policy
#: refused the request. The other codes the SDK lists (``server_error``,
#: ``rate_limit_exceeded``, image decoding errors, and so on) are failures, not refusals.
RESPONSES_POLICY_ERROR_CODES = frozenset({"bio_policy", "image_content_policy_violation"})


def refusal_from_status_error(status_code: int, body: object) -> StopInfo | None:
    """Return the refusal an HTTP error response describes, or None when it is something else.

    Recognized, and only these:

    * ``400`` with ``type`` or ``code`` ``content_filter`` (Kimi: "considered high risk").
    * ``403`` with ``metadata.reasons`` (an OpenRouter moderation flag) or
      ``metadata.patterns`` (an OpenRouter guardrail). A budget or permission ``403`` has
      other metadata and stays an error.

    Args:
        status_code: The HTTP status.
        body: The ``error`` object of the response body, as the SDK exposes it.
    """
    if not isinstance(body, Mapping):
        return None
    message = body.get("message")
    text = str(message) if message else None
    if status_code == 400 and "content_filter" in (body.get("type"), body.get("code")):
        return StopInfo(
            StopReason.REFUSAL,
            raw="content_filter",
            category="content_filter",
            message=text,
            detail={"status": status_code},
        )
    metadata = body.get("metadata")
    if status_code == 403 and isinstance(metadata, Mapping):
        if "reasons" in metadata:
            kept = {
                key: metadata[key]
                for key in ("reasons", "flagged_input", "provider_name", "model_slug")
                if key in metadata
            }
            return StopInfo(
                StopReason.REFUSAL,
                raw=str(status_code),
                category="moderation",
                message=text,
                detail={"status": status_code, **kept},
            )
        if "patterns" in metadata:
            return StopInfo(
                StopReason.REFUSAL,
                raw=str(status_code),
                category="guardrail",
                message=text,
                detail={"status": status_code, "patterns": metadata["patterns"]},
            )
    return None


def refusal_from_response_error(error: Mapping[str, JsonValue]) -> StopInfo | None:
    """Return the refusal a failed OpenAI Responses stream describes, or None for a plain failure."""
    code = error.get("code")
    if code not in RESPONSES_POLICY_ERROR_CODES:
        return None
    message = error.get("message")
    return StopInfo(
        StopReason.REFUSAL,
        raw=str(code),
        category=str(code),
        message=str(message) if message else None,
        detail={"status": "failed"},
    )


def withhold_replay_blocks(
    blocks: Sequence[Mapping[str, JsonValue]], is_tool_call: Callable[[Mapping[str, JsonValue]], bool]
) -> tuple[Mapping[str, JsonValue], ...]:
    """Drop the replay blocks that are client tool calls.

    When a step is refused or interrupted its tool calls are withheld from the message, so
    the provider's own record of them must go too: a ``tool_use`` block sent back without a
    ``tool_result`` is rejected by the API.
    """
    return tuple(block for block in blocks if not is_tool_call(block))
