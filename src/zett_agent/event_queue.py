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

    Why acknowledgement instead of a bounded queue
    ----------------------------------------------

    ``asyncio.Queue(maxsize=n)`` bounds buffered items, but it does not bound how
    much work the producer has already finished, and that difference decides
    where cancellation lands. Two properties of a bounded queue open the gap:

    1. The producer suspends inside ``put``, which runs after the work for that
       event completed. A full queue blocks handing over the result, not doing
       the work, so the producer can always finish one more unit that nobody has
       consumed yet.
    2. A blocked ``put`` is released by ``get``, which runs when the consumer
       takes the item, before it has handled that item. Every dequeue therefore
       grants one more unit of production work while the consumer is still busy
       with the previous event.

    Together they let a ``maxsize=1`` queue carry up to two finished events past
    the consumer's position. With 30 ms of work and 150 ms of handling per
    event, the producer finishes events 1 and 2 while the consumer is still
    handling event 0, and only then parks on ``put``::

         31.5 ms  producer queued 0       (queue is full)
         31.5 ms  producer START work 1   (not suspended)
         31.5 ms  consumer took 0         (slot freed again)
         93.1 ms  producer FINISH work 2  (event 2 already executed)
        182.6 ms  consumer DONE handling 0

    Acknowledgement moves the suspension point between two events instead of
    after the work of one: ``put`` returns once the event is queued and then
    waits for ``acknowledge``, which ``Agent.stream`` calls only after the
    caller asks for the next event. ``_pending_acknowledgement`` enforces that
    single in-flight contract, so a consumer that pulls twice without advancing
    raises ``AgentProtocolError`` instead of silently losing backpressure.

    The guarantee matters because a run has side effects: tool handlers execute
    shell commands and write files, and persistence appends records, all before
    the caller has seen the corresponding event. Applications stop a run by
    closing the stream, and acknowledgement makes that cancellation land between
    events, where no model call or tool execution is in flight. A bounded queue
    would instead cancel inside the extra work it allowed the producer to start.

    Terminal signals stay non-blocking by design: ``fail`` and ``close`` publish
    through the unbounded queue, so only data events ever wait for the consumer.
    Acknowledgement is opt-in because the queue also serves callers that treat
    it as a plain FIFO channel.

    Trade-off: the producer pulls from the model provider no faster than the
    consumer drains events. A slow consumer stalls the run instead of buffering
    it, which is the intended meaning of backpressure for a stream that performs
    side effects.
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
        """Make future writes wait until the stream consumes each event.

        Optional: without it the queue is a plain FIFO channel whose ``put``
        never waits for the consumer. Enabled before production starts, which
        keeps the first event from racing ahead of the acknowledgement setup.
        """
        if not self._queue.empty():
            raise AgentProtocolError("Cannot enable event acknowledgement after production starts")
        self._acknowledgements = True

    async def put(self, event: AgentEvent) -> None:
        """Append one event, rejecting writes after a terminal signal.

        With acknowledgement enabled this returns only after ``acknowledge``
        releases the event, so the caller resumes between events rather than
        inside the work that produced one. Closing the stream while the caller
        waits here cancels it without interrupting a model call or tool.
        """
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
        """Release the producer after the consumer advances past one yield.

        ``Agent.stream`` calls this after the caller asks for the next event, not
        when the event was handed over, so the producer stays parked for the
        whole time the caller is processing the current event.
        """
        acknowledged = self._pending_acknowledgement
        self._pending_acknowledgement = None
        if acknowledged is not None and not acknowledged.done():
            acknowledged.set_result(None)

    async def fail(self, error: BaseException) -> None:
        """Wake the consumer with the original producer failure.

        Never waits for the consumer: the unbounded queue keeps terminal signals
        deliverable even when nobody is draining events.
        """
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
        """Return the next event or propagate the producer's terminal state.

        The returned event stays unacknowledged until ``acknowledge`` runs; a
        second ``get`` beforehand is a protocol error because it would let the
        consumer pull past the producer's suspension point.
        """
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
