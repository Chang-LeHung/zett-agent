"""Where an interrupted request can be picked up again.

A request can end without finishing: the connection drops, the user presses stop, the
process is killed. Nothing records that. What is left is the history, and its last
message says what was happening when it stopped, so that is what is read. A finished
request ends in an assistant answer that asked for nothing and stopped normally; every
other ending is an interruption, and what to do about it depends on which one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._compat import StrEnum
from .exceptions import AgentResumeError
from .messages import AnyMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage
from .stop import StopInfo, StopReason

__all__ = ["ResumeKind", "ResumePoint", "classify_resume"]


class ResumeKind(StrEnum):
    """What resuming does, which depends on how the request was interrupted."""

    #: The history is ready for the next model call: it ends in a user or internal message,
    #: in tool results that are all there, or in a step the provider paused. Nothing is added.
    CALL_MODEL = "call_model"
    #: An assistant message asked for tools and not every call has a result, because the
    #: process stopped while they ran. Each missing result is added, saying that the tool was
    #: interrupted and may or may not have run, and the model decides what to do. A tool is
    #: never run again on its own, because nothing says whether its effect already happened.
    RECOVER_TOOLS = "recover_tools"
    #: A response was cut off by a failure, not by the model. It stays in the history, and a
    #: message is added telling the model it was interrupted and to start that response again.
    RETRY_RESPONSE = "retry_response"
    #: A response reached the output limit. It stays in the history, and a message is added
    #: telling the model to continue where it stopped.
    CONTINUE_RESPONSE = "continue_response"


@dataclass(frozen=True, slots=True)
class ResumePoint:
    """Where a session was interrupted and what resuming it will do.

    Attributes:
        kind: How it was interrupted, and so what resuming does about it.
        description: One sentence for a person, such as for a "Resume" button.
        unanswered_tool_calls: The calls that have no result, for
            :attr:`ResumeKind.RECOVER_TOOLS`; empty otherwise.
        stop: How the last assistant message stopped, when there is one and it did not
            finish normally.
    """

    kind: ResumeKind
    description: str
    unanswered_tool_calls: tuple[ToolCall, ...] = ()
    stop: StopInfo | None = None


def classify_resume(messages: Sequence[AnyMessage]) -> ResumePoint:
    """Read where a history was interrupted from its last message.

    Args:
        messages: The session's context messages, oldest first. System messages are ignored.

    Raises:
        AgentResumeError: If there is nothing to resume, or what the history ends in cannot be
            picked up again. Its ``reason`` says which.
    """
    dialogue = [message for message in messages if not isinstance(message, SystemMessage)]
    if not dialogue:
        raise AgentResumeError("empty", "The session has no messages, so there is nothing to resume.")
    last = dialogue[-1]
    if isinstance(last, ToolMessage):
        return _after_tools(dialogue)
    if isinstance(last, AssistantMessage):
        return _after_assistant(last)
    return ResumePoint(
        ResumeKind.CALL_MODEL,
        "The model was never answered: the request stopped before it replied.",
    )


def _after_tools(dialogue: Sequence[AnyMessage]) -> ResumePoint:
    """Classify a history that ends in tool results."""
    index = len(dialogue) - 1
    answered: set[str] = set()
    while index >= 0 and isinstance(dialogue[index], ToolMessage):
        answered.add(dialogue[index].tool_call_id)  # type: ignore[union-attr]
        index -= 1
    batch = dialogue[index] if index >= 0 else None
    if not isinstance(batch, AssistantMessage) or not batch.tool_calls:
        raise AgentResumeError("malformed", "The history ends in a tool result that answers no tool call.")
    missing = tuple(call for call in batch.tool_calls if call.id not in answered)
    if missing:
        return _unanswered(missing)
    return ResumePoint(
        ResumeKind.CALL_MODEL,
        "The tools finished, and the request stopped before the model saw their results.",
    )


def _unanswered(calls: tuple[ToolCall, ...]) -> ResumePoint:
    names = ", ".join(sorted({call.name for call in calls}))
    return ResumePoint(
        ResumeKind.RECOVER_TOOLS,
        f"The request stopped while tools were running ({names}). They will be reported to the "
        "model as interrupted, and it will decide what to do.",
        unanswered_tool_calls=calls,
    )


def _after_assistant(message: AssistantMessage) -> ResumePoint:
    """Classify a history that ends in an assistant message."""
    if message.tool_calls:
        return _unanswered(tuple(message.tool_calls))
    stop = message.stop
    if stop is None or stop.ok:
        raise AgentResumeError("completed", "The last request finished, so there is nothing to resume.")
    match stop.reason:
        case StopReason.INTERRUPTED:
            return ResumePoint(
                ResumeKind.RETRY_RESPONSE,
                f"The last response was cut off ({stop.describe()}). The model will be told and asked to start it again.",
                stop=stop,
            )
        case StopReason.MAX_OUTPUT:
            return ResumePoint(
                ResumeKind.CONTINUE_RESPONSE,
                "The last response reached the output limit. The model will be asked to continue it.",
                stop=stop,
            )
        case StopReason.PAUSED:
            return ResumePoint(
                ResumeKind.CALL_MODEL,
                "The provider paused the last response and asked for the turn to be continued.",
                stop=stop,
            )
        case StopReason.REFUSAL:
            raise AgentResumeError(
                "refused",
                f"The model declined the last request ({stop.describe()}); the same input would be declined again.",
            )
        case StopReason.CONTEXT_FULL:
            raise AgentResumeError(
                "context_full",
                "The last response stopped because the context window was full, which resuming does not fix.",
            )
        case _:
            raise AgentResumeError(
                "unknown_stop", f"The last response stopped for a reason that cannot be resumed ({stop.describe()})."
            )
