"""External event envelopes and the shared request-response extension base."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Collection
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from .extension_events import ExtensionEvent, RunCancelledEvent
from .extension_hooks import AgentExtension

if TYPE_CHECKING:
    from .agent import AgentContext


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

    context: AgentContext
    future: asyncio.Future[ExternalEvent]
    response_event_names: frozenset[str]
    accepted: bool = False


class ExternalEventExtension(AgentExtension):
    """Base class for extensions that suspend until an external response arrives.

    Subclasses define one response event name and one payload field used to
    correlate that response with a pending operation. This base owns routing,
    thread synchronization, Future completion, cancellation, error propagation,
    and temporary delivery storage::

        subclass                         ExternalEventExtension
           |                                      |
           | await _wait_for_external_event()     | create and await Future
           |------------------------------------->|
           |                                      |<-- accept() from any thread
           |                                      |    set_result on owner loop
           |<-------------------------------------|
           |
           | _take_external_event() -> response

        RunCancelledEvent -> future.cancel()
        on_error(error)   -> future.set_exception(error)

    The Agent broadcasts external events to accept(). Concrete extensions do
    not override accept(), on_event(), or on_error() unless they need additional
    protocol behavior.
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
        self._accepted: dict[AgentContext, ExternalEvent] = {}
        self._pending_lock = Lock()

    @asynccontextmanager
    async def _wait_for_external_event(
        self,
        context: AgentContext,
        correlation_id: str,
        *,
        response_event_name: str | None = None,
    ) -> AsyncGenerator[None]:
        """Register before yielding UI output, then await and stage its response."""
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
            self._accepted[context] = response
        finally:
            with self._pending_lock:
                self._pending.pop(key, None)
            if not pending.future.done():
                pending.future.cancel()

    def _take_external_event(self, context: AgentContext) -> ExternalEvent:
        """Consume the oldest response staged for the current request."""
        response = self._accepted.pop(context, None)
        if response is None:
            raise RuntimeError("Tool executed without an accepted external response")
        return response

    def accept(self, event: ExternalEvent) -> bool:
        """Route one matching external response to its waiting Future."""
        if event.name not in self._response_event_names:
            return False
        session_id = event.payload.get("session_id")
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
            pending.accepted = True

        def deliver() -> None:
            if not pending.future.done():
                pending.future.set_result(event)

        self._run_on_future_loop(pending.future, deliver)
        return True

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Cancel active waits when their request enters CANCELLED."""
        if isinstance(event, RunCancelledEvent):
            self._discard_staged(context)
            self._terminate_waits(context)

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Complete active waits with the request's original error."""
        self._discard_staged(context)
        self._terminate_waits(context, error=error)

    def _terminate_waits(self, context: AgentContext, *, error: Exception | None = None) -> None:
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

    def _discard_staged(self, context: AgentContext) -> None:
        """Remove responses that can no longer be consumed by a tool."""
        self._accepted.pop(context, None)

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
