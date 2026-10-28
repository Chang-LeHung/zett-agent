"""The loop that alternates model calls and tool runs until an answer ends the request."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..events import AgentEventType
from ..exceptions import AgentIterationLimitError
from ..messages import AgentMessage, AssistantMessage, ToolCall, ToolMessage, UserMessage
from ..model import ReasoningEffort
from ..stop import continues_pause
from .contract import AgentContract, mixin_api

if TYPE_CHECKING:
    from ..agent import AgentRunContext

#: The event that reports what became of an input that was being worked on, by whether the input
#: was steering from the user (first) or an internal message from an extension (second).
_INPUT_EVENTS = {
    "completed": (AgentEventType.STEERING_COMPLETED, AgentEventType.INTERNAL_MESSAGE_COMPLETED),
    "interrupted": (AgentEventType.STEERING_INTERRUPTED, AgentEventType.INTERNAL_MESSAGE_INTERRUPTED),
}


@dataclass(slots=True)
class _LoopState:
    """What the loop carries from one model call to the next."""

    #: The steering or internal message being worked on, or None while working on the user's input.
    active_input: AgentMessage | UserMessage | None = None
    #: Model calls spent on the current input; the budget is ``max_iterations``.
    message_iterations: int = 0
    #: Internal messages taken up so far in this request; the budget is ``max_internal_messages``.
    processed_internal_messages: int = 0
    #: Paused steps continued in a row; the budget is ``max_pause_continuations``.
    pauses: int = 0
    #: Whether the step about to run continues a paused one, so its turn is not opened again.
    continuing: bool = False


class TurnLoopMixin(AgentContract):
    """Drive a request: one model call, then its tools or its answer, until the request ends."""

    @mixin_api
    async def _run_loop(
        self,
        context: AgentRunContext,
        reasoning_effort: ReasoningEffort,
        parallel_tool_call: bool,
    ) -> None:
        """Alternate model and tool steps while emitting into the run queue."""
        loop = _LoopState()
        while True:
            if loop.message_iterations >= self.max_iterations:
                raise AgentIterationLimitError(f"Agent exceeded {self.max_iterations} model iterations")
            loop.message_iterations += 1
            # A paused step is continued inside the same turn, so the turn is not reopened.
            if not loop.continuing:
                await self._broadcast("before_turn", context)
            loop.continuing = False
            may_pause = loop.pauses < self.max_pause_continuations
            response = await self._run_model_step(context, reasoning_effort, parallel_tool_call, may_pause=may_pause)
            stop = response.stop
            if continues_pause(stop, may_pause=may_pause, has_tool_calls=bool(response.message.tool_calls)):
                loop.pauses += 1
                loop.continuing = True
                continue
            loop.pauses = 0
            await self._announce_stop(context, response)
            if response.message.tool_calls:
                await self._run_tool_turn(context, response.message, parallel_tool_call, loop)
            elif await self._finish_answer(context, response.message, loop):
                return

    async def _finish_answer(self, context: AgentRunContext, answer: AssistantMessage, loop: _LoopState) -> bool:
        """Close the turn an answer ended, then start more work or complete the request.

        More work is steering from the user first, then an internal message from an extension.
        Returns True when there was none and the request is complete.

        Raises:
            AgentIterationLimitError: If another internal message would exceed the request's budget.
        """
        from ..extensions.events import MessageTiming

        if loop.active_input is not None:
            await self._announce_input(context, "completed", loop.active_input, message=answer)
            loop.active_input = None

        # The answer ends this turn: close the turn before the runtime
        # decides whether the request continues with new input.
        await self._broadcast("after_turn", context, answer)
        steering, next_input = self._select_pending_input(context)
        if steering is not None:
            await self._start_steering(context, steering)
            loop.active_input = steering
            loop.message_iterations = 0
            return False
        if next_input is not None:
            if loop.processed_internal_messages >= self.max_internal_messages:
                raise AgentIterationLimitError(
                    f"Agent exceeded {self.max_internal_messages} internal messages in one request"
                )
            loop.processed_internal_messages += 1
            loop.message_iterations = 0
            await context._emit_event(AgentEventType.INTERNAL_MESSAGE_STARTED, internal_message=next_input)
            await context.append_message(next_input, MessageTiming.instant())
            loop.active_input = next_input
            return False

        await self._broadcast("after_run", context, answer)
        await self._broadcast("on_success", context, answer)
        await self._complete_request(context)
        await context._emit_event(AgentEventType.RUN_COMPLETED, message=answer)
        return True

    async def _run_tool_turn(
        self,
        context: AgentRunContext,
        message: AssistantMessage,
        parallel_tool_call: bool,
        loop: _LoopState,
    ) -> None:
        """Run the tools a model asked for, or skip them when steering arrived first, and close the turn."""
        steering = self._steering_extension.take(context)
        if steering is not None:
            await self._start_steering(context, steering, message.tool_calls, loop.active_input)
            # Steering skipped the requested tools; their skipped results are
            # appended, so the turn closes before the next turn begins.
            await self._broadcast("after_turn", context, message)
            loop.active_input = steering
            loop.message_iterations = 0
            return

        steering = await self._execute_tools(context, message.tool_calls, parallel_tool_call, loop.active_input)
        await self._broadcast("after_turn", context, message)
        if steering is not None:
            loop.active_input = steering
            loop.message_iterations = 0

    def _select_pending_input(self, context: AgentRunContext) -> tuple[UserMessage | None, AgentMessage | None]:
        """Reserve steering first, or internal input, without a queue-closing race."""
        internal: AgentMessage | None = None

        def reserve_internal_input() -> bool:
            nonlocal internal
            internal = self._internal_message_extension.next_message(context)
            return internal is not None

        steering = self._steering_extension.select_next(context, reserve_internal_input)
        return steering, internal

    async def _announce_input(
        self,
        context: AgentRunContext,
        outcome: str,
        active_input: AgentMessage | UserMessage,
        **fields: Any,
    ) -> None:
        """Report that the steering or internal message being worked on ``"completed"`` or was ``"interrupted"``."""
        steering_event, internal_event = _INPUT_EVENTS[outcome]
        internal = isinstance(active_input, AgentMessage)
        await context._emit_event(
            internal_event if internal else steering_event,
            internal_message=active_input if internal else None,
            steering_message=active_input if isinstance(active_input, UserMessage) else None,
            **fields,
        )

    @mixin_api
    async def _start_steering(
        self,
        context: AgentRunContext,
        message: UserMessage,
        pending_calls: Sequence[ToolCall] = (),
        active_input: AgentMessage | UserMessage | None = None,
    ) -> None:
        """Finish pending tool protocol entries before appending urgent user input.

        Skipped calls are never invoked and do not trigger execution hooks. Their
        explicit failed ToolMessages preserve the provider's call/result pairing
        and Raw Log history. Interrupted inputs never emit a false completion.
        """
        from ..extensions.events import MessageTiming

        if active_input is not None:
            await self._announce_input(context, "interrupted", active_input)
        for call in pending_calls:
            result = ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content="Superseded by a steering message",
                success=False,
            )
            await context.append_message(result, MessageTiming.instant())
            await context._emit_event(AgentEventType.TOOL_SKIPPED, tool_calls=[call], message=result)
        await context.append_message(message, MessageTiming.instant())
        await context._emit_event(AgentEventType.STEERING_STARTED, steering_message=message)
