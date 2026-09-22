from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field

from ..agent import AgentRunContext
from ..ids import new_uuid7
from ..json_types import JsonValue
from ..messages import AnyMessage, AssistantMessage, SystemMessage, ToolMessage
from ..model import ModelUsage
from .base import AgentExtension
from .compaction import CompactedMessage
from .events import CompactionEvent, ExtensionEvent, MessageAppendedEvent, MessageTiming, RunCancelledEvent


@dataclass(frozen=True, slots=True)
class RawMessageRecord:
    """One immutable original message returned by session storage."""

    id: str
    session_id: str
    request_id: str
    sequence: int
    message: AnyMessage
    #: Request-scoped application data captured independently of model messages.
    metadata: dict[str, JsonValue]
    #: Request-scoped classifications captured independently of model messages.
    tags: dict[str, JsonValue]
    #: UTC operation start; GENERATING for assistants, RUNNING_TOOL for tools,
    #: and append time for user-authored or directly imported messages.
    started_at: datetime
    #: UTC operation completion; equal to started_at for instantaneous messages.
    completed_at: datetime
    #: Total operation duration in nanoseconds, measured by a monotonic clock.
    duration_ns: int
    #: UTC arrival time of the first streamed reasoning delta, when present.
    reasoning_started_at: datetime | None
    #: UTC boundary where reasoning ended before content, tools, or completion.
    reasoning_completed_at: datetime | None
    #: Monotonic elapsed reasoning time in nanoseconds; None if not observed.
    reasoning_duration_ns: int | None
    #: UTC arrival time of the first streamed answer-content delta, when present.
    content_started_at: datetime | None
    #: UTC final-response boundary for a streamed answer-content segment.
    content_completed_at: datetime | None
    #: Monotonic elapsed content-streaming time in nanoseconds; None if absent.
    content_duration_ns: int | None
    #: Provider-reported prompt tokens for an assistant model response.
    input_tokens: int | None
    #: Provider-reported completion tokens, including reasoning when applicable.
    output_tokens: int | None
    #: Input tokens served from a provider cache.
    cache_read_tokens: int | None
    #: Input tokens written into a provider cache when separately reported.
    cache_write_tokens: int | None
    #: Reasoning tokens included within output_tokens when separately reported.
    reasoning_tokens: int | None
    created_at: datetime
    updated_at: datetime

    @property
    def total_tokens(self) -> int | None:
        """Return input plus output tokens when provider usage was recorded."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        """Return a zero-to-one input cache ratio, or None without input usage."""
        if not self.input_tokens or self.cache_read_tokens is None:
            return None
        return self.cache_read_tokens / self.input_tokens


class SessionSummary(BaseModel):
    """Small read model used to list persisted conversation sessions."""

    session_id: str = Field(description="Stable conversation identifier")
    parent_session_id: str | None = Field(
        default=None,
        description="Parent conversation for a delegated subagent; null for a root session",
    )
    session_type: int = Field(
        description="Storage-owned integer code describing the session origin",
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Human-readable session title; applications may update it",
    )
    agent_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Provider-neutral agent profile owning the session",
    )
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
        | Session Context | Latest Snapshot plus the Raw Log tail after it.      |
        +-----------------+------------------------------------------------------+

    A Snapshot stores only one CompactedMessage. Recent messages remain solely
    in the Raw Log and are selected using the Snapshot boundary::

        Raw Log       [1] [2] [3] [4] [5] [6] [7] | [8] [9] [10]
                       \\______ compacted ______/   \\__ tail__/

        Snapshot      [CompactedMessage through sequence 7]

        Session       [CompactedMessage] [8] [9] [10]
        Context

    No separate "last sequence" belongs to Session Context. Sequence numbers
    identify Raw Log records and the Snapshot boundary only.
    """

    parent_session_id: str | None = Field(
        default=None,
        description="Parent conversation for a delegated subagent; null for a root session",
    )
    session_type: int = Field(
        description="Storage-owned integer code describing the session origin",
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Human-readable session title; applications may update it",
    )
    agent_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Provider-neutral agent profile owning the session",
    )
    snapshot: ContextSnapshot | None = Field(default=None, description="Latest compacted checkpoint, if any")
    raw_tail: list[RawMessageRecord] = Field(
        default_factory=list,
        description="Ordered Raw Log messages after the checkpoint boundary",
    )

    @property
    def messages(self) -> list[AnyMessage]:
        """Build active model context as checkpoint followed by its Raw Log tail."""
        checkpoint = [self.snapshot.compacted_message] if self.snapshot is not None else []
        return [
            message
            for message in [*checkpoint, *(record.message for record in self.raw_tail)]
            if message.include_in_messages
        ]


class SessionStorage(Protocol):
    """Persistence boundary; implementations own transactions and message encoding."""

    async def load(self, session_id: str) -> SessionView:
        """Load the latest checkpoint and all Raw Log messages after its boundary."""
        ...

    async def append(
        self,
        session_id: str,
        request_id: str,
        message: AnyMessage,
        timing: MessageTiming | None = None,
        parent_session_id: str | None = None,
        title: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
        usage: ModelUsage | None = None,
    ) -> int:
        """Append one immutable Raw Log message and return its allocated sequence.

        A direct storage caller may omit timing for an instantaneous imported
        message. The Agent runtime always supplies measured timing. The optional
        Session attributes initialize the session record on its first append and
        are not copied into each Raw Log row. Usage belongs only to assistant
        model responses. The parent ID is immutable.
        """
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
    #: One Raw Log sequence for each replayable non-system context position.
    #: None represents a context-only message omitted by ``persist=False``.
    context_sequences: list[int | None] = field(default_factory=list)


class BaseSessionPersistenceExtension[StorageT: SessionStorage](AgentExtension):
    """Reusable lifecycle adapter for one typed session storage implementation.

    This base owns all framework-specific behavior: request bookkeeping,
    context restoration, immutable Raw Log appends, compaction snapshots, and
    cleanup. A concrete persistence extension only needs to construct a storage
    implementation and pass it to ``super().__init__``::

        class CustomSessionExtension(BaseSessionPersistenceExtension[CustomStorage]):
            def __init__(self, client: CustomClient) -> None:
                super().__init__(CustomStorage(client))

    Register this instead of InMemoryMessageAccumulator, before prompt extensions.
    Raw messages remain immutable. A compaction event stores only its new
    CompactedMessage; recent context continues to live in the Raw Log.

    Examples:
        Usage::

            agent = await Agent.create(model, config=config, extensions=[
                CustomSessionExtension(client),
                ToolGuidelinesExtension(),
                CompactionExtension(model),
            ])
    """

    def __init__(self, storage: StorageT) -> None:
        self.storage = storage
        self._requests: WeakKeyDictionary[AgentRunContext, _Request] = WeakKeyDictionary()

    @staticmethod
    def _provider_safe_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        """Exclude interrupted tool batches from restored model context.

        Raw Log records remain immutable and available to a UI. Older processes
        may nevertheless have stopped after persisting an AssistantMessage with
        tool calls and before persisting every matching ToolMessage. Providers
        reject that sequence, so this projection omits only the malformed
        assistant/tool batch while retaining surrounding completed dialogue.
        """
        restored: list[AnyMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if isinstance(message, ToolMessage):
                # An orphan result has no meaningful provider context.
                index += 1
                continue
            if not isinstance(message, AssistantMessage) or not message.tool_calls:
                restored.append(message)
                index += 1
                continue

            tool_messages: list[ToolMessage] = []
            cursor = index + 1
            while cursor < len(messages) and isinstance(messages[cursor], ToolMessage):
                tool_messages.append(messages[cursor])
                cursor += 1
            expected = Counter(call.id for call in message.tool_calls)
            observed = Counter(result.tool_call_id for result in tool_messages)
            if expected == observed:
                restored.extend((message, *tool_messages))
            index = cursor
        return restored

    async def _restore(self, context: AgentRunContext) -> SessionView:
        view = await self.storage.load(context.config.session_id)
        configured_parent = context.state.parent_session_id
        session_exists = view.snapshot is not None or bool(view.raw_tail)
        if configured_parent is not None and session_exists and view.parent_session_id != configured_parent:
            raise ValueError("Stored session belongs to a different parent session")
        context.state.parent_session_id = view.parent_session_id or configured_parent
        # Instructions are supplied by the current application configuration;
        # persisted context contributes dialogue and checkpoints only.
        instructions = [message for message in context.state.messages if isinstance(message, SystemMessage)]
        context.replace_messages(
            [*instructions, *self._provider_safe_messages(view.messages)],
            emit_new=False,
        )
        return view

    async def on_state(self, context: AgentRunContext) -> None:
        """Restore context and map every dialogue position to its Raw Log boundary."""
        view = await self._restore(context)
        sequences = []
        if view.snapshot is not None:
            sequences.append(view.snapshot.compacted_through_sequence)
        sequences.extend(
            record.sequence
            for record in view.raw_tail
            if record.message.include_in_messages and not isinstance(record.message, SystemMessage)
        )
        self._requests[context] = _Request(
            request_id=context.config.request_id or new_uuid7(),
            snapshot_version=view.snapshot.version if view.snapshot is not None else 0,
            context_sequences=sequences,
        )

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        if isinstance(event, RunCancelledEvent):
            self._requests.pop(context, None)
            return
        request = self._requests.get(context)
        if request is None:
            return
        match event:
            case MessageAppendedEvent(message=message, timing=timing, usage=usage):
                sequence = None
                if message.persist:
                    sequence = await self.storage.append(
                        context.config.session_id,
                        request.request_id,
                        message,
                        timing,
                        parent_session_id=context.state.parent_session_id,
                        metadata=context.metadata,
                        tags=context.tags,
                        usage=usage,
                    )
                if message.include_in_messages and not isinstance(message, SystemMessage):
                    request.context_sequences.append(sequence)
            case CompactionEvent() as compaction:
                await self._snapshot(context, request, compaction)

    async def _snapshot(self, context: AgentRunContext, request: _Request, event: CompactionEvent) -> None:
        """Store the summary and remap its context position to the compacted prefix."""
        if event.compressed_from != 1 or event.compressed_to > len(request.context_sequences):
            raise ValueError("Compaction range does not match the restored Raw Log context")
        compacted_sequences = [
            sequence for sequence in request.context_sequences[: event.compressed_to] if sequence is not None
        ]
        if not compacted_sequences:
            raise ValueError("Compaction must represent at least one persisted Raw Log message")
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

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Release bookkeeping; original messages were appended as they arrived."""
        self._requests.pop(context, None)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Release bookkeeping; append-only Raw Log history remains available."""
        self._requests.pop(context, None)
