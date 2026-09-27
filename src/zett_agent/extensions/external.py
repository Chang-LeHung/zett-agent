"""External event envelopes and the shared request-response extension base."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Collection
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..tools.base import current_tool_call_id
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent

if TYPE_CHECKING:
    from ..agent import AgentRunConfig, AgentRunContext


@dataclass(frozen=True, slots=True)
class ExternalEvent:
    """Provider-neutral envelope whose payload protocol belongs to its receiver."""

    name: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("External event name cannot be empty")
        if not isinstance(self.payload, dict) or any(not isinstance(key, str) for key in self.payload):
            raise TypeError("External event payload must be a dictionary with string keys")


@dataclass(slots=True)
class _PendingExternalEvent:
    """One context-bound Future protected against duplicate cross-thread delivery."""

    context: AgentRunContext
    future: asyncio.Future[ExternalEvent]
    response_event_names: frozenset[str]
    accepted: bool = False


class ExternalEventExtension(AgentExtension):
    """Base class for extensions that suspend until an external response arrives.

    Use this base for tools that need a UI decision before execution can finish.
    The subclass owns the tool, outbound UI event, and business validation. The
    base owns pending Futures, routing, duplicate rejection, thread-safe wake-up,
    and cleanup. It does not send HTTP responses or render UI components.

    Args:
        response_event_name: Accepted inbound name, or a collection of names.
        correlation_field: Payload key identifying the operation, usually
            tool_call_id. Its value must equal the ID passed to the wait helper.

    The configuration passed separately to accept(config, event) selects the
    waiting session and optional request ID. Agent does not inject routing
    identifiers into the payload.

    Lifecycle::

        +---------------------------+     +---------------------------+
        | before_tool()             |     | UI / external caller      |
        +---------------------------+     +---------------------------+
        | enter async with:         |     |                           |
        | register pending Future   |     |                           |
        | context.emit(CUSTOM)      |---->| display question          |
        | leave body: await Future  |     | emit_external_event(      |
        |                           |<----|     event, config=config) |
        | accept(): claim response, |     +---------------------------+
        | schedule Future completion|
        | stage response by context |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | tool executes             |
        | _take_external_event()    |
        | validate and return result|
        +---------------------------+

    Entering the async context registers the wait BEFORE the UI event is emitted,
    so even an immediate response is safe. The await happens when leaving the
    async-with body, not on entry. It suspends this task without blocking other
    sessions. The tool executes only after the response has been staged.

    Example::

        from zett_agent.agent import AgentRunContext
        from zett_agent.events import (
            AgentEvent,
            AgentEventType,
        )
        from zett_agent.extensions.external import ExternalEventExtension
        from zett_agent.messages import ToolCall
        from zett_agent.tools.base import tool

        class ConfirmationExtension(ExternalEventExtension):
            def __init__(self) -> None:
                super().__init__(
                    response_event_name="confirmation_response",
                    correlation_field="tool_call_id",
                )

            async def on_tool(self, context: AgentRunContext) -> None:
                @tool
                async def confirm() -> bool:
                    '''Ask the user whether to proceed.

                    Snippet:
                        confirm()

                    Guidelines:
                        - Use when proceeding requires an explicit user decision.
                    '''
                    response = self._take_external_event(context)
                    approved = response.payload.get("approved")
                    if not isinstance(approved, bool):
                        raise ValueError("approved must be a boolean")
                    return approved

                context.register_tool(confirm)

            async def before_tool(self, context: AgentRunContext, call: ToolCall) -> None:
                if call.name != "confirm":
                    return
                async with self._wait_for_external_event(context, call.id):
                    await context.emit(AgentEvent(
                        AgentEventType.CUSTOM,
                        session_id=context.config.session_id,
                        name="confirmation_requested",
                        payload={"tool_call_id": call.id, "question": "Proceed?"},
                    ))

    Usage::

        from zett_agent.agent import (
            Agent,
            AgentRunConfig,
        )
        from zett_agent.extensions.external import ExternalEvent

        config = AgentRunConfig(session_id="session-42", request_id="request-1")
        agent = await Agent.create(model, config=config, extensions=[ConfirmationExtension()])

        # Consume agent.stream(...) and forward confirmation_requested to the UI.
        # A separate UI callback returns the exact tool_call_id from that event.
        accepted_by = agent.emit_external_event(
            ExternalEvent(
                name="confirmation_response",
                payload={"tool_call_id": "call-7", "approved": True},
            ),
            config=config,
        )
        # ["ConfirmationExtension"] if accepted; [] for a stale/duplicate response.

    Acceptance means delivery was claimed, not that tool execution succeeded.
    Payload business validation belongs to the subclass. A missing config,
    mismatched request, unknown correlation ID, or duplicate response is rejected.
    Different sessions may reuse the same tool_call_id without sharing responses.

    Cancellation clears staged responses and cancels pending Futures; on_error
    wakes pending waits with the error. Overrides of on_event/on_error must call
    super() to preserve cleanup. Future mutation is scheduled on its owning loop
    when accept is called from another thread. Keep that loop alive until request
    cleanup has finished. This base supports sequential tools: one staged response
    per AgentRunContext, not concurrent external-response tools in the same request.
    """

    def __init__(self, *, response_event_name: str | Collection[str], correlation_field: str) -> None:
        names = (response_event_name,) if isinstance(response_event_name, str) else tuple(response_event_name)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("response_event_name cannot be empty")
        if not correlation_field.strip():
            raise ValueError("correlation_field cannot be empty")
        self._response_event_names = frozenset(names)
        self._correlation_field = correlation_field
        self._pending: dict[tuple[str, str], _PendingExternalEvent] = {}
        self._accepted: dict[tuple[AgentRunContext, str], ExternalEvent] = {}
        self._pending_lock = Lock()

    @asynccontextmanager
    async def _wait_for_external_event(
        self,
        context: AgentRunContext,
        correlation_id: str,
        *,
        response_event_name: str | None = None,
    ) -> AsyncGenerator[None]:
        """Register on entry; await and stage the response when the body exits.

        Yield the outbound AgentEvent inside this async-with body. Optionally
        restrict response_event_name to one of the configured inbound names.
        The pending entry is removed on every exit, including cancellation.
        """
        if not correlation_id.strip():
            raise ValueError("correlation_id cannot be empty")
        response_names = self._response_event_names
        if response_event_name is not None:
            if response_event_name not in response_names:
                raise ValueError(f"Unsupported response event name: {response_event_name!r}")
            response_names = frozenset((response_event_name,))
        key = (context.config.session_id, correlation_id)
        pending = _PendingExternalEvent(context, asyncio.get_running_loop().create_future(), response_names)
        with self._pending_lock:
            if key in self._pending:
                raise RuntimeError(f"External event is already pending: {correlation_id}")
            self._pending[key] = pending
        try:
            yield
            response = await pending.future
            self._accepted[(context, correlation_id)] = response
        finally:
            with self._pending_lock:
                self._pending.pop(key, None)
            if not pending.future.done():
                pending.future.cancel()

    def _take_external_event(self, context: AgentRunContext) -> ExternalEvent:
        """Pop this invocation's staged response exactly once inside its tool.

        Call only after _wait_for_external_event has finished successfully.
        Raises RuntimeError if no response is staged or it was already consumed.
        """
        correlation_id = current_tool_call_id()
        response = self._accepted.pop((context, correlation_id), None) if correlation_id is not None else None
        if response is None:
            raise RuntimeError("Tool executed without an accepted external response")
        return response

    def accept(self, config: AgentRunConfig | None, event: ExternalEvent) -> bool:
        """Claim a matching response and wake its waiter on the Future's loop.

        Usually called by Agent.emit_external_event, not directly by the UI.
        True means this extension accepted delivery; False leaves other
        extensions free to recognize the event. Neither argument is modified.
        """
        if event.name not in self._response_event_names:
            return False
        if config is None:
            return False
        session_id = config.session_id
        correlation_id = event.payload.get(self._correlation_field)
        if not isinstance(session_id, str) or not session_id.strip():
            return False
        if not isinstance(correlation_id, str) or not correlation_id.strip():
            return False
        with self._pending_lock:
            pending = self._pending.get((session_id, correlation_id))
            if (
                pending is None
                or event.name not in pending.response_event_names
                or pending.accepted
                or pending.future.done()
            ):
                return False
            request_id = config.request_id
            if request_id is not None and request_id != pending.context.config.request_id:
                return False
            pending.accepted = True

        def deliver() -> None:
            if not pending.future.done():
                pending.future.set_result(event)

        self._run_on_future_loop(pending.future, deliver)
        return True

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Cancel active waits when their request enters CANCELLED."""
        if isinstance(event, RunCancelledEvent):
            self._discard_staged(context)
            self._terminate_waits(context)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Complete active waits with the request's original error."""
        self._discard_staged(context)
        self._terminate_waits(context, error=error)

    def _terminate_waits(self, context: AgentRunContext, *, error: Exception | None = None) -> None:
        """Remove and wake every pending operation owned by one request."""
        with self._pending_lock:
            keys = [key for key, pending in self._pending.items() if pending.context is context]
            pending_events = [self._pending.pop(key) for key in keys]
            for pending in pending_events:
                pending.accepted = True

        for pending in pending_events:

            def terminate(current: _PendingExternalEvent = pending) -> None:
                if current.future.done():
                    return
                if error is None:
                    current.future.cancel()
                else:
                    current.future.set_exception(error)

            self._run_on_future_loop(pending.future, terminate)

    def _discard_staged(self, context: AgentRunContext) -> None:
        """Remove responses that can no longer be consumed by a tool."""
        stale = [key for key in self._accepted if key[0] is context]
        for key in stale:
            self._accepted.pop(key, None)

    @staticmethod
    def _run_on_future_loop(future: asyncio.Future[Any], callback: Callable[[], None]) -> None:
        """Mutate a Future directly or schedule the mutation on its owner loop."""
        loop = future.get_loop()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            callback()
        else:
            loop.call_soon_threadsafe(callback)
