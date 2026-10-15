"""Why a model stopped, in one vocabulary that every provider maps onto.

Providers report this differently: a ``finish_reason`` string, a ``stop_reason``
with a side object, a refusal block inside the content, an HTTP error, or a
``200`` that carries an error chunk. A host that only checks for tool calls sees
all of them as "the model answered". :class:`StopInfo` is what the runtime
attaches to every assistant message so the difference is never lost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._compat import StrEnum

__all__ = ["StopInfo", "StopReason", "classify_stop"]


class StopReason(StrEnum):
    """Provider-neutral reason a model step ended.

    Only :attr:`END_TURN` and :attr:`TOOL_USE` are normal. Everything else means the
    output may be incomplete, absent, or not what the model would have said.
    """

    #: The model finished its answer, or hit a stop sequence you set.
    END_TURN = "end_turn"
    #: The model asked for tools to run.
    TOOL_USE = "tool_use"
    #: The output limit was reached; the text, or a tool call's arguments, is cut off.
    MAX_OUTPUT = "max_output"
    #: The context window was filled; the output is cut off.
    CONTEXT_FULL = "context_full"
    #: The model, or a safety system in front of it, declined. The output is partial
    #: or empty, and a tool call that came with it is withheld.
    REFUSAL = "refusal"
    #: The provider handed control back and wants the turn continued, as Anthropic
    #: does when a server-tool loop reaches its limit.
    PAUSED = "paused"
    #: Infrastructure ended the response: the provider failed, dropped the stream, or
    #: ran out of capacity. It is not a decision of the model, so the output cannot be
    #: trusted to be complete and a tool call that came with it is withheld.
    INTERRUPTED = "interrupted"
    #: A value this library does not know. :attr:`StopInfo.raw` holds it unchanged.
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class StopInfo:
    """Why one model step stopped, normalized, with what the provider said kept.

    Attributes:
        reason: The normalized reason. Branch on this.
        raw: The provider's own value, unchanged, such as ``"content_filter"`` or
            ``"model_context_window_exceeded"``; None when it gave none.
        category: What kind of refusal or interruption, when the provider says. It
            is the provider's policy category where there is one (Anthropic's
            ``"cyber"``, Google's ``"SAFETY"``) and otherwise the raw value.
        message: A sentence for a person, such as Anthropic's ``explanation``, Google's
            ``finish_message``, or the refusal text an OpenAI model wrote. Show it;
            do not parse it, because providers change the wording.
        recommended_model: A model the provider suggests retrying on, when it does.
        detail: Provider-specific facts that have no normalized home, such as an
            OpenRouter ``native_finish_reason`` or an error code. JSON-compatible.

    Examples:
        React to a refusal without caring which provider produced it::

            message = await agent.run("...")
            if message.stop is not None and message.stop.reason is StopReason.REFUSAL:
                print("Declined:", message.stop.category, message.stop.message)

    .. note::
        The runtime writes a ``StopInfo`` on the assistant message only when the step did
        not end normally. A message that finished or asked for tools has ``stop`` None, as
        does every message stored before this existed, so test ``stop is None or stop.ok``.
    """

    reason: StopReason
    raw: str | None = None
    category: str | None = None
    message: str | None = None
    recommended_model: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the step ended normally: it finished, or it asked for tools."""
        return self.reason in (StopReason.END_TURN, StopReason.TOOL_USE)

    @property
    def refused(self) -> bool:
        """Whether the model or a safety system declined."""
        return self.reason is StopReason.REFUSAL

    @property
    def truncated(self) -> bool:
        """Whether the output ran into a limit and is cut off."""
        return self.reason in (StopReason.MAX_OUTPUT, StopReason.CONTEXT_FULL)

    def describe(self) -> str:
        """Return one line for a log or a notice, such as ``refusal (cyber): ...``."""
        head = self.reason.value
        label = self.category or (self.raw if self.raw not in (None, head) else None)
        if label:
            head = f"{head} ({label})"
        return f"{head}: {self.message}" if self.message else head


#: Every raw value the supported providers are known to send, lower-cased, and what each
#: means. A provider sends one of these in its ``finish_reason`` or ``stop_reason``; the
#: values for each provider are listed in the documentation. An unknown value is
#: :attr:`StopReason.OTHER` rather than an error, so a new provider value never breaks a run.
_REASONS: dict[str, StopReason] = {
    # Finished.
    "stop": StopReason.END_TURN,
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "completed": StopReason.END_TURN,
    # Asked for tools.
    "tool_calls": StopReason.TOOL_USE,
    "tool_use": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    # Limits.
    "length": StopReason.MAX_OUTPUT,
    "max_tokens": StopReason.MAX_OUTPUT,
    "max_output_tokens": StopReason.MAX_OUTPUT,
    "model_context_window_exceeded": StopReason.CONTEXT_FULL,
    "model_length": StopReason.CONTEXT_FULL,
    "context_length_exceeded": StopReason.CONTEXT_FULL,
    # Declined or blocked.
    "refusal": StopReason.REFUSAL,
    "content_filter": StopReason.REFUSAL,
    "sensitive": StopReason.REFUSAL,
    "safety": StopReason.REFUSAL,
    "recitation": StopReason.REFUSAL,
    "blocklist": StopReason.REFUSAL,
    "prohibited_content": StopReason.REFUSAL,
    "spii": StopReason.REFUSAL,
    "image_safety": StopReason.REFUSAL,
    "image_prohibited_content": StopReason.REFUSAL,
    "image_recitation": StopReason.REFUSAL,
    "jailbreak": StopReason.REFUSAL,
    "model_armor": StopReason.REFUSAL,
    # Handed back.
    "pause_turn": StopReason.PAUSED,
    # Ended by infrastructure.
    "error": StopReason.INTERRUPTED,
    "network_error": StopReason.INTERRUPTED,
    "insufficient_system_resource": StopReason.INTERRUPTED,
    "aborted": StopReason.INTERRUPTED,
}


def classify_stop(
    raw: str | None,
    *,
    has_tool_calls: bool = False,
    category: str | None = None,
    message: str | None = None,
    recommended_model: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> StopInfo:
    """Turn a provider's raw stop value into a :class:`StopInfo`.

    The raw value is looked up without regard to case. A step that finished
    (``stop``, ``end_turn``) but carries tool calls is :attr:`StopReason.TOOL_USE`,
    because that is what Gemini, which reports ``STOP`` either way, means. A step
    with no value at all is treated as finished, or as asking for tools, so an
    adapter that reports nothing does not look like a failure.

    A refusal or an interruption with no ``category`` takes the raw value as its
    category, so "what kind" is never empty when the provider said anything.

    Args:
        raw: The provider's value, or None when it gave none.
        has_tool_calls: Whether the step produced tool calls.
        category: A more specific category than the raw value, when the provider has one.
        message: A sentence for a person.
        recommended_model: A model the provider suggests retrying on.
        detail: Provider-specific facts to keep.
    """
    if raw is None:
        reason = StopReason.TOOL_USE if has_tool_calls else StopReason.END_TURN
    else:
        reason = _REASONS.get(raw.strip().lower(), StopReason.OTHER)
        if reason is StopReason.END_TURN and has_tool_calls:
            reason = StopReason.TOOL_USE
    if category is None and raw is not None and reason in (StopReason.REFUSAL, StopReason.INTERRUPTED):
        category = raw
    return StopInfo(
        reason=reason,
        raw=raw,
        category=category,
        message=message,
        recommended_model=recommended_model,
        detail=dict(detail) if detail else {},
    )
