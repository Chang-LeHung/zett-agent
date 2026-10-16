"""Picking an interrupted session up: reading where it stopped, and closing what it left open."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import AgentResumeError
from ..messages import AgentMessage, AssistantMessage, ToolCall, ToolMessage
from ..resume import ResumeKind, ResumePoint, classify_resume
from .contract import AgentContract, mixin_api

if TYPE_CHECKING:
    from ..agent import AgentRunContext

#: What a tool call is answered with when the request that made it was cancelled while it ran.
CANCELLED_TOOL_RESULT = "Request cancelled before tool completion"
#: What a tool call is answered with when the process stopped while it ran and left no trace. It
#: says the effect is unknown, because a tool that writes a file may have done so before the stop.
INTERRUPTED_TOOL_RESULT = (
    "The process stopped while this tool was running, so it may or may not have completed, and whatever "
    "it changes may or may not have happened. Check the current state before running it again."
)
#: What the model is told when its response was cut off by a failure and the session is resumed.
RETRY_RESPONSE_NOTICE = (
    "Your previous response was cut off by an interruption (a lost connection or a provider failure) "
    "before it finished. Start that response again from the beginning."
)
#: What the model is told when its response reached the output limit and the session is resumed.
CONTINUE_RESPONSE_NOTICE = (
    "Your previous response was cut off at the output limit. Continue exactly where it stopped, "
    "without repeating what you already wrote."
)

#: The message a resume adds for the kinds that tell the model something, keyed by kind.
_RESUME_NOTICES = {
    ResumeKind.RETRY_RESPONSE: RETRY_RESPONSE_NOTICE,
    ResumeKind.CONTINUE_RESPONSE: CONTINUE_RESPONSE_NOTICE,
}


class RecoveryMixin(AgentContract):
    """Read the tail of a session's history and make it ready to carry on from."""

    @mixin_api
    async def _close_unanswered_tool_calls(self, context: AgentRunContext, content: str) -> None:
        """Append a failed result, saying ``content``, for each call of the latest batch that has none."""
        from ..extensions.events import MessageTiming

        pending: dict[str, ToolCall] = {}
        for message in context.state.messages:
            match message:
                case AssistantMessage(tool_calls=calls) if calls:
                    pending = {call.id: call for call in calls}
                case ToolMessage(tool_call_id=tool_call_id) if pending:
                    pending.pop(tool_call_id, None)
                case _:
                    # A valid tool batch is contiguous. Another message means
                    # this is older malformed history, not the operation that
                    # the current cancellation interrupted.
                    if pending:
                        pending.clear()

        for call in pending.values():
            await context.append_message(
                ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=content,
                    success=False,
                ),
                MessageTiming.instant(),
            )

    @mixin_api
    async def _open_resume(self, context: AgentRunContext) -> None:
        """Open inboxes, restore history, and prepare the interrupted request to carry on."""
        self._internal_message_extension.open(context)
        self._steering_extension.open(context)
        await self._restore_context(context)
        await self._settle_context(context)
        await self._prepare_resume(context, classify_resume(context.state.messages))
        await self._broadcast("before_run", context)

    async def _prepare_resume(self, context: AgentRunContext, point: ResumePoint) -> None:
        """Make the history ready for the next model call, telling the model what it missed."""
        from ..extensions.events import MessageTiming

        if point.kind is ResumeKind.RECOVER_TOOLS:
            await self._close_unanswered_tool_calls(context, INTERRUPTED_TOOL_RESULT)
        elif (notice := _RESUME_NOTICES.get(point.kind)) is not None:
            await context.append_message(
                AgentMessage(content=notice, attributes={"resume": point.kind.value}),
                MessageTiming.instant(),
            )

    @mixin_api
    async def _inspect_session(self, context: AgentRunContext) -> ResumePoint | None:
        """Restore a session as a request would and report where it was interrupted, changing nothing."""
        await self._restore_context(context)
        await self._settle_context(context)
        try:
            point: ResumePoint | None = classify_resume(context.state.messages)
        except AgentResumeError:
            point = None
        await self._complete_request(context)
        return point
