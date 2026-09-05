from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field

from .agent import AgentContext, AgentExtension
from .compaction import CompactedMessage
from .extension_events import CompactionEvent, ExtensionEvent, MessageAppendedEvent
from .ids import new_uuid7
from .messages import AnyMessage, AssistantMessage, SystemMessage


@dataclass(frozen=True, slots=True)
class RawMessageRecord:
    """One immutable original message returned by session storage."""

    id: str
    session_id: str
    request_id: str
    sequence: int
    message: AnyMessage
    created_at: datetime
    updated_at: datetime


class SessionSummary(BaseModel):
    """Small read model used to list persisted conversation sessions."""

    session_id: str = Field(description="Stable conversation identifier")
    message_count: int = Field(description="Number of immutable Raw Log messages")
    created_at: datetime = Field(description="UTC time of the first Raw Log message")
    updated_at: datetime = Field(description="UTC time of the latest Raw Log message")


class ContextSnapshot(BaseModel):
    """One immutable checkpoint representing a compacted Raw Log prefix."""

    id: str = Field(description="Stable snapshot identifier")
    session_id: str = Field(description="Session whose context was compacted")
    version: int = Field(description="Monotonically increasing snapshot version")
    compacted_message: CompactedMessage = Field(description="Summary replacing the compacted Raw Log prefix")
    compacted_through_sequence: int = Field(description="Last Raw Log sequence represented by the summary")
    created_at: datetime = Field(description="UTC snapshot creation time")
    updated_at: datetime = Field(description="UTC modification time; equal to creation time while immutable")


class SessionView(BaseModel):
    """Latest checkpoint and the subsequent immutable Raw Log tail.

    Persistence terminology::

        +-----------------+------------------------------------------------------+
        | Raw Log         | Complete, immutable original conversation history.   |
        +-----------------+------------------------------------------------------+
        | Snapshot        | One checkpoint replacing a compacted Raw Log prefix. |
        +-----------------+------------------------------------------------------+
        | Session Context | Latest Snapshot plus the Raw Log tail after it.       |
        +-----------------+------------------------------------------------------+

    A Snapshot stores only one CompactedMessage. Recent messages remain solely
    in the Raw Log and are selected using the Snapshot boundary::

        Raw Log       [1] [2] [3] [4] [5] [6] [7] | [8] [9] [10]
                       \\________ compacted _______/   \\___ tail ___/

        Snapshot      [CompactedMessage through sequence 7]

        Session       [CompactedMessage] [8] [9] [10]
        Context

    No separate "last sequence" belongs to Session Context. Sequence numbers
    identify Raw Log records and the Snapshot boundary only.
    """

    snapshot: ContextSnapshot | None = Field(default=None, description="Latest compacted checkpoint, if any")
    raw_tail: list[RawMessageRecord] = Field(
        default_factory=list,
        description="Ordered Raw Log messages after the checkpoint boundary",
    )

    @property
    def messages(self) -> list[AnyMessage]:
        """Build active model context as checkpoint followed by its Raw Log tail."""
        checkpoint = [self.snapshot.compacted_message] if self.snapshot is not None else []
        return [*checkpoint, *(record.message for record in self.raw_tail)]


class SessionStorage(Protocol):
    """Persistence boundary; implementations own transactions and message encoding."""

    async def load(self, session_id: str) -> SessionView:
        """Load the latest checkpoint and all Raw Log messages after its boundary."""
        ...

    async def append(self, session_id: str, request_id: str, message: AnyMessage) -> int:
        """Append one immutable Raw Log message and return its allocated sequence."""
        ...

    async def snapshot(
        self,
        session_id: str,
        compacted_message: CompactedMessage,
        compacted_through_sequence: int,
        expected_version: int,
    ) -> ContextSnapshot:
        """Create one immutable checkpoint, rejecting a stale snapshot version."""
        ...


@dataclass
class _Request:
    """Persistence bookkeeping for one request-scoped context."""

    request_id: str = field(default_factory=new_uuid7)
    snapshot_version: int = 0
    # One Raw Log sequence for each non-system context position. A checkpoint
    # position maps to the last original message represented by that checkpoint.
    context_sequences: list[int] = field(default_factory=list)


class SessionPersistenceExtension(AgentExtension):
    """Restore context before every request and append original messages.

    Register this instead of InMemoryMessageAccumulator, before prompt extensions.
    Raw messages remain immutable. A compaction event stores only its new
    CompactedMessage; recent context continues to live in the Raw Log.

    Example:
        agent = await Agent.create(model, config=config, extensions=[
            SessionPersistenceExtension(storage),
            ToolGuidelinesExtension(),
            CompactionExtension(model),
        ])
    """

    def __init__(self, storage: SessionStorage) -> None:
        self.storage = storage
        self._requests: WeakKeyDictionary[AgentContext, _Request] = WeakKeyDictionary()

    async def _restore(self, context: AgentContext) -> SessionView:
        view = await self.storage.load(context.config.session_id)
        # Instructions are supplied by the current application configuration;
        # persisted context contributes dialogue and checkpoints only.
        instructions = [message for message in context.state.messages if isinstance(message, SystemMessage)]
        context.state.messages[:] = [*instructions, *view.messages]
        return view

    async def on_message(self, context: AgentContext) -> None:
        """Restore context and map every dialogue position to its Raw Log boundary."""
        view = await self._restore(context)
        sequences = []
        if view.snapshot is not None:
            sequences.append(view.snapshot.compacted_through_sequence)
        sequences.extend(record.sequence for record in view.raw_tail)
        self._requests[context] = _Request(
            request_id=context.config.request_id or new_uuid7(),
            snapshot_version=view.snapshot.version if view.snapshot is not None else 0,
            context_sequences=sequences,
        )

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        request = self._requests.get(context)
        if request is None:
            return
        match event:
            case MessageAppendedEvent(message=message):
                sequence = await self.storage.append(context.config.session_id, request.request_id, message)
                request.context_sequences.append(sequence)
            case CompactionEvent() as compaction:
                await self._snapshot(context, request, compaction)

    async def _snapshot(self, context: AgentContext, request: _Request, event: CompactionEvent) -> None:
        """Store the summary and remap its context position to the compacted prefix."""
        if event.compressed_from != 1 or event.compressed_to > len(request.context_sequences):
            raise ValueError("Compaction range does not match the restored Raw Log context")
        compacted_sequences = request.context_sequences[: event.compressed_to]
        if not compacted_sequences:
            raise ValueError("Compaction must represent at least one Raw Log message")
        compacted_through_sequence = compacted_sequences[-1]
        snapshot = await self.storage.snapshot(
            context.config.session_id,
            CompactedMessage(content=event.summary),
            compacted_through_sequence,
            request.snapshot_version,
        )
        request.snapshot_version = snapshot.version
        request.context_sequences[:] = [
            compacted_through_sequence,
            *request.context_sequences[event.compressed_to :],
        ]

    async def on_success(self, context: AgentContext, result: AssistantMessage) -> None:
        """Release bookkeeping; original messages were appended as they arrived."""
        self._requests.pop(context, None)

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Release bookkeeping; append-only Raw Log history remains available."""
        self._requests.pop(context, None)
