"""Single-producer event channel used by one Agent run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .events import AgentEvent, AgentEventType
from .exceptions import AgentProtocolError


@dataclass(frozen=True, slots=True)
class _QueueFailure:
    error: BaseException


@dataclass(frozen=True, slots=True)
class _QueueClosed:
    pass


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    event: AgentEvent
    acknowledged: asyncio.Future[None] | None = None


type _QueueItem = _QueuedEvent | _QueueFailure | _QueueClosed


class AgentEventQueue:
    """Carry one run's ordered events from the runtime task to ``Agent.stream``.

    Exactly one producer owns the queue. ``Agent.stream`` enables acknowledgement
    before starting that producer, which preserves stream backpressure: producing
    the next event waits until the caller advances beyond the current yield. This
    also ensures closing a stream at an event boundary still cancels unfinished
    work instead of allowing a detached producer to run ahead in the background.

    Success is represented by ``RUN_COMPLETED``; failures retain and re-raise the
    original exception in the consumer task.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._closed = False
        self._completed = False
        self._acknowledgements = False
        self._pending_acknowledgement: asyncio.Future[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def empty(self) -> bool:
        """Return whether no produced item is waiting for consumption."""
        return self._queue.empty()

    def enable_acknowledgements(self) -> None:
        """Make future writes wait until the stream consumes each event."""
        if not self._queue.empty():
            raise AgentProtocolError("Cannot enable event acknowledgement after production starts")
        self._acknowledgements = True

    async def put(self, event: AgentEvent) -> None:
        """Append one event, rejecting writes after a terminal signal."""
        if self._closed:
            raise AgentProtocolError("Cannot emit an event after the run event queue closed")
        if self._completed:
            raise AgentProtocolError("Cannot emit an event after RUN_COMPLETED")
        self._completed = event.type is AgentEventType.RUN_COMPLETED
        acknowledged = asyncio.get_running_loop().create_future() if self._acknowledgements else None
        await self._queue.put(_QueuedEvent(event, acknowledged))
        if acknowledged is not None:
            await acknowledged

    def acknowledge(self) -> None:
        """Release the producer after the consumer advances past one yield."""
        acknowledged = self._pending_acknowledgement
        self._pending_acknowledgement = None
        if acknowledged is not None and not acknowledged.done():
            acknowledged.set_result(None)

    async def fail(self, error: BaseException) -> None:
        """Wake the consumer with the original producer failure."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_QueueFailure(error))

    async def close(self) -> None:
        """Wake the consumer when a producer exits without another event."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_QueueClosed())

    async def get(self) -> AgentEvent:
        """Return the next event or propagate the producer's terminal state."""
        item = await self._queue.get()
        match item:
            case _QueuedEvent(event=event, acknowledged=acknowledged):
                if self._pending_acknowledgement is not None:
                    raise AgentProtocolError("Previous run event was not acknowledged")
                self._pending_acknowledgement = acknowledged
                return event
            case _QueueFailure(error=error):
                raise error
            case _QueueClosed():
                raise AgentProtocolError("Agent run ended without RUN_COMPLETED")


__all__ = ["AgentEventQueue"]
