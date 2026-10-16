"""One request from start to finish: its context, how its events reach the caller, and how it ends.

Every way of running the agent, whether a request, a resume, a compaction pass or a read-only
look at a session, claims a session, builds a context, runs inside a guard that releases it however
it ends, and hands its events to a caller. They share those steps here instead of each carrying a
copy.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .._compat import aclosing_stream, add_note
from ..events import AgentEvent, AgentEventType
from ..exceptions import AgentProtocolError
from ..ids import new_uuid7
from ..json_types import JsonValue, json_object
from ..messages import AssistantMessage, SystemMessage, UserMessage
from ..model import AgentModel, ReasoningEffort
from .contract import AgentContract, mixin_api
from .recovery import CANCELLED_TOOL_RESULT, INTERRUPTED_TOOL_RESULT

if TYPE_CHECKING:
    from ..agent import AgentRunConfig, AgentRunContext


async def final_answer(events: AsyncIterator[AgentEvent]) -> AssistantMessage:
    """Drain a run's events, closing them, and return the answer that ended it.

    Raises:
        AgentProtocolError: If the events end without a ``RUN_COMPLETED`` carrying an answer.
    """
    result: AssistantMessage | None = None
    async with aclosing_stream(events) as stream:
        async for event in stream:
            if event.type == AgentEventType.RUN_COMPLETED and isinstance(event.message, AssistantMessage):
                result = event.message
    if result is None:
        raise AgentProtocolError("The agent did not produce a final answer")
    return result


class RequestLifecycleMixin(AgentContract):
    """Create, run, and release the request-scoped context of one pass over a session.

    The hooks, recovery and turn methods this calls belong to the sibling mixins it is composed with;
    :class:`~zett_agent._agent.contract.AgentContract` declares them for the type checker.
    """

    # ------------------------------------------------------------------ the context

    @mixin_api
    def _resolve_run_options(
        self, reasoning_effort: ReasoningEffort | None, parallel_tool_call: bool | None
    ) -> tuple[ReasoningEffort, bool]:
        """Apply the agent's defaults to a request's overrides, rejecting a non-boolean parallel flag."""
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        allow_parallel = self.parallel_tool_call if parallel_tool_call is None else parallel_tool_call
        if not isinstance(allow_parallel, bool):
            raise ValueError("parallel_tool_call must be a boolean")
        return effort, allow_parallel

    @staticmethod
    def _resolve_config(config: AgentRunConfig) -> AgentRunConfig:
        """Fill in the session and request identity a request needs, generating what is missing."""
        session_id = config.session_id or new_uuid7()
        if config.parent_session_id == session_id:
            raise ValueError("parent_session_id must differ from session_id")
        config = replace(config, session_id=session_id)
        if config.request_id is None:
            config = replace(config, request_id=new_uuid7())
        return config

    @mixin_api
    def _prepare_request_context(
        self,
        config: AgentRunConfig | None,
        model: AgentModel | None,
        metadata: Mapping[str, JsonValue] | None,
        tags: Mapping[str, JsonValue] | None,
        input_message: UserMessage | None,
    ) -> AgentRunContext:
        """Validate input, create isolated state/tools, and claim the session.

        ``input_message`` is ``None`` for a pass that takes no new input, such as a compaction
        pass, a resume, or a read-only look at the session.
        """
        initialized = self._initialized_config
        if initialized is None:
            raise AgentProtocolError(
                "Agent is not initialized; await agent.initialize(config=...) or Agent.create(...)"
            )
        config = self._resolve_config(config or initialized)
        resolved_model = model if model is not None else self.model
        if resolved_model is None:
            raise AgentProtocolError("This request requires a model because the Agent has no default model")
        context = self._build_context(config, resolved_model, metadata, tags, input_message)
        self._claim_session(context)
        for extension in self.extensions:
            extension._request_started(context)
        return context

    def _build_context(
        self,
        config: AgentRunConfig,
        model: AgentModel,
        metadata: Mapping[str, JsonValue] | None,
        tags: Mapping[str, JsonValue] | None,
        input_message: UserMessage | None,
    ) -> AgentRunContext:
        """Create the request-scoped state, tools and context, with the system prompt in place."""
        from ..agent import AgentRunContext, AgentState

        # State and dynamic tools are request-scoped. Constructor tools are copied
        # so extension registrations cannot leak into later requests or sessions.
        context = AgentRunContext(
            config=config,
            state=AgentState(parent_session_id=config.parent_session_id),
            tools={},
            extensions=self.extensions,
            model=model,
            metadata=json_object(metadata, field_name="Context metadata"),
            tags=json_object(tags, field_name="Context tags", nonempty_keys=True),
            input_message=input_message,
        )
        if self.system_prompt:
            context.add_message(SystemMessage(content=self.system_prompt))
        for registered in self.tools.values():
            context.register_tool(registered)
        return context

    def _claim_session(self, context: AgentRunContext) -> None:
        """Make this request the one that owns its session, or refuse if another does."""
        session_id, state = context.session_id, context.state
        with self._sessions_lock:
            previous = self._states.get(session_id)
            if self.is_conflict(session_id):
                phase = previous.phase.value if previous is not None else "active"
                raise AgentProtocolError(f"Agent cannot start a new request while its current phase is {phase!r}")
            self._active_configs[session_id] = context.config
            self._states[session_id] = state
            self.state = state

    def _release_request(self, context: AgentRunContext) -> None:
        """Clear only this request's inboxes and release its session for reuse."""
        self._steering_extension.close(context)
        self._internal_message_extension.close(context)
        for extension in self.extensions:
            extension._request_finished(context)
        with self._sessions_lock:
            self._active_configs.pop(context.session_id, None)

    # ------------------------------------------------------------------ loading history

    @mixin_api
    async def _restore_context(self, context: AgentRunContext) -> None:
        """Begin loading: let the extensions register tools, then restore the session's history."""
        await self._start_context_loading(context)
        await self._broadcast("on_tool", context)
        await self._broadcast("on_state", context)

    @mixin_api
    async def _settle_context(self, context: AgentRunContext) -> None:
        """Finish loading: publish what the setup hooks added, in the order a request writes it."""
        await context.flush_message_events()
        await self._finish_context_loading(context)

    async def _open_request(self, context: AgentRunContext) -> None:
        """Open inboxes, restore history through hooks, then append the new input."""
        from ..extensions.events import MessageTiming

        self._internal_message_extension.open(context)
        self._steering_extension.open(context)
        await self._restore_context(context)
        await self._broadcast("on_message", context)
        if not isinstance(context.input_message, UserMessage):
            raise AgentProtocolError("Request setup must produce a UserMessage")
        # Persist setup-generated instructions before the user input so Raw Log
        # order matches the final model context.
        await self._settle_context(context)
        # A request that was killed could not close its own tool calls. The history is
        # restored with the results that did arrive, so any call still without one is
        # answered here, before the new input, or the provider would reject the history.
        await self._close_unanswered_tool_calls(context, INTERRUPTED_TOOL_RESULT)
        await context.append_message(context.input_message, MessageTiming.instant())
        await self._broadcast("before_run", context)

    # ------------------------------------------------------------------ running and ending

    @mixin_api
    @asynccontextmanager
    async def _request_scope(self, context: AgentRunContext, *, via_queue: bool = False) -> AsyncGenerator[None]:
        """Run one pass, and however it ends, report it and release its session.

        A cancellation closes interrupted tool calls and publishes the cancellation, then propagates.
        A failure runs the error hooks. With ``via_queue`` the failure is handed to the event queue,
        which raises it in the caller that is iterating the events, and is not raised here; without
        it the failure propagates to the caller of the scope. The session is released and the
        queue closed either way.
        """
        try:
            yield
        except asyncio.CancelledError as cancellation:
            await self._handle_request_cancellation(context, cancellation)
            if via_queue:
                await context.event_queue.fail(cancellation)
            raise
        except Exception as error:
            await self._handle_request_failure(context, error)
            if not via_queue:
                raise
            await context.event_queue.fail(error)
        finally:
            self._release_request(context)
            await context.event_queue.close()

    @mixin_api
    async def _run_request(
        self,
        context: AgentRunContext,
        reasoning_effort: ReasoningEffort,
        parallel_tool_call: bool,
        *,
        resume: bool = False,
    ) -> None:
        """Produce one run into its queue and own every terminal cleanup path."""
        async with self._request_scope(context, via_queue=True):
            if resume:
                await self._open_resume(context)
            else:
                await self._open_request(context)
            await self._run_loop(context, reasoning_effort, parallel_tool_call)

    @mixin_api
    async def _run_compaction(self, context: AgentRunContext) -> None:
        """Restore one session's history, then let every extension compact it."""
        async with self._request_scope(context, via_queue=True):
            await self._restore_context(context)
            # Setup-generated instructions are persisted before the pass, the
            # order a request writes them in.
            await self._settle_context(context)
            for extension in self.extensions:
                await extension.on_compact(context)
            await self._complete_request(context)

    async def _handle_request_cancellation(self, context: AgentRunContext, cancellation: BaseException) -> None:
        """Close interrupted tool calls, then publish the request cancellation.

        A completed assistant tool-call message is persisted before execution
        starts. If cancellation interrupts a tool, leaving that message without
        one ToolMessage per call makes the next provider request invalid. Write
        explicit failed observations while persistence bookkeeping is still
        active so the restored conversation remains protocol-complete.
        """
        try:
            await self._close_unanswered_tool_calls(context, CANCELLED_TOOL_RESULT)
            await self._cancel_request(context)
        except Exception as notification_error:
            add_note(cancellation, f"Cancellation notification failed: {notification_error!r}")

    async def _handle_request_failure(self, context: AgentRunContext, error: Exception) -> None:
        """Report failure without masking the original error or terminal phase."""
        # A terminal transition is already committed before subscribers run.
        # Never replace the original error with an illegal terminal transition
        # or a secondary failure in an error-reporting hook.
        if context.state.phase in self._ACTIVE_PHASES:
            try:
                await self._fail_request(context)
            except Exception as notification_error:
                add_note(error, f"Failure notification failed: {notification_error!r}")
        try:
            await self._broadcast("on_error", context, error)
        except Exception as notification_error:
            add_note(error, f"Error hook failed: {notification_error!r}")

    # ------------------------------------------------------------------ handing events out

    @mixin_api
    async def _pump_events(
        self,
        context: AgentRunContext,
        work: Coroutine[Any, Any, None],
        *,
        ends_silently: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Run ``work`` in a producer task and yield the events it queues, one at a time.

        A request, a resume and a compaction pass all end here, so each is cancelled, acknowledged
        and cleaned up the same way. A pass that can end without emitting anything, as compaction
        does when there is nothing to compact, passes ``ends_silently`` so that the end of the
        producer, not a terminal event, ends the stream.
        """
        context.event_queue.enable_acknowledgements()
        producer = asyncio.create_task(work)
        try:
            while True:
                if ends_silently:
                    event = await self._next_silent_event(context, producer)
                    if event is None:
                        return
                else:
                    event = await context.event_queue.get()
                yield event
                # Acknowledge only after the caller asks for the next event: the
                # producer then stays parked between events for as long as the
                # caller handles the current one. A caller that stops early (a UI
                # stop, a dropped client) therefore cancels the producer before it
                # starts the next model call or tool, instead of inside one.
                # Releasing earlier would let it finish work nobody asked for.
                # AgentEventQueue documents why this replaces a bounded queue.
                context.event_queue.acknowledge()
                if event.type is AgentEventType.RUN_COMPLETED:
                    await producer
                    return
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _next_silent_event(
        self,
        context: AgentRunContext,
        producer: asyncio.Task[None],
    ) -> AgentEvent | None:
        """Return the next event, or None once a pass that may emit nothing has ended.

        A pass with nothing to report emits nothing, so the queue alone cannot
        say that it is over: acknowledgement keeps the producer parked between
        events, which makes "the producer finished" and "no more events" the
        same moment. A queue closed while empty is that moment; anything else
        the queue reports is a real error and is raised.
        """
        waiting = asyncio.ensure_future(context.event_queue.get())
        await asyncio.wait({waiting, producer}, return_when=asyncio.FIRST_COMPLETED)
        if waiting.done():
            error = waiting.exception()
            if error is None:
                return waiting.result()
            if isinstance(error, AgentProtocolError) and context.event_queue.closed and context.event_queue.empty:
                await producer
                return None
            raise error
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await producer
        return None
