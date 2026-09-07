import base64
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path

from pydantic import TypeAdapter
from sqlalchemy import Index, Integer, String, Text, create_engine, delete, func, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .extensions.compaction import CompactedMessage
from .extensions.events import MessageTiming
from .extensions.persistence import ContextSnapshot, RawMessageRecord, SessionSummary, SessionView
from .ids import new_uuid7
from .json_types import JsonValue, json_object
from .messages import AgentMessage, AnyMessage, AssistantMessage, SystemMessage, ToolMessage, UserMessage
from .model import ModelUsage


class Base(DeclarativeBase):
    """Schema owned exclusively by zett-agent."""


class AgentSessionModel(Base):
    """One root or delegated conversation and its provider-neutral identity."""

    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_session_id: Mapped[str | None] = mapped_column(String(36), index=True)
    title: Mapped[str | None] = mapped_column(String(200))
    agent_name: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()


class RawLogMessageModel(Base):
    """Append-only original messages; role is a numeric MessageKind."""

    __tablename__ = "raw_messages"
    __table_args__ = (Index("ux_raw_session_sequence", "session_id", "sequence", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(String)
    message_json: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    tags_json: Mapped[str] = mapped_column(Text, default="{}")
    # UTC start of message processing, not the later database insertion time.
    started_at: Mapped[datetime] = mapped_column()
    # UTC completion of message processing.
    completed_at: Mapped[datetime] = mapped_column()
    # Total monotonic elapsed time in nanoseconds.
    duration_ns: Mapped[int] = mapped_column(Integer)
    # UTC first-reasoning-delta boundary; null when reasoning was not streamed.
    reasoning_started_at: Mapped[datetime | None] = mapped_column()
    # UTC reasoning completion boundary; null when reasoning was not streamed.
    reasoning_completed_at: Mapped[datetime | None] = mapped_column()
    # Monotonic reasoning duration in nanoseconds; null when not observed.
    reasoning_duration_ns: Mapped[int | None] = mapped_column(Integer)
    # UTC first-content-delta boundary; null when content was not streamed.
    content_started_at: Mapped[datetime | None] = mapped_column()
    # UTC final-response boundary for streamed content; null when not observed.
    content_completed_at: Mapped[datetime | None] = mapped_column()
    # Monotonic content-streaming duration in nanoseconds; null when absent.
    content_duration_ns: Mapped[int | None] = mapped_column(Integer)
    # Provider token counters exist only for assistant model responses.
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    # UTC timestamps; immutable records have identical creation/modification times.
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()


class ContextSnapshotModel(Base):
    """One immutable checkpoint and the Raw Log prefix represented by it."""

    __tablename__ = "session_snapshots"
    __table_args__ = (Index("ux_snapshot_session_version", "session_id", "version", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int] = mapped_column(Integer)
    compacted_through_sequence: Mapped[int] = mapped_column(Integer)
    message_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()


class MessageKind(IntEnum):
    """Numeric types for lossless runtime messages and checkpoints."""

    SYSTEM = 1
    USER = 2
    ASSISTANT = 3
    TOOL = 4
    CHECKPOINT = 5
    AGENT = 6


MESSAGE_TYPES = {
    MessageKind.AGENT: AgentMessage,
    MessageKind.SYSTEM: SystemMessage,
    MessageKind.USER: UserMessage,
    MessageKind.ASSISTANT: AssistantMessage,
    MessageKind.TOOL: ToolMessage,
    MessageKind.CHECKPOINT: CompactedMessage,
}


def encode_messages(messages: Sequence[AnyMessage]) -> str:
    """Encode complete typed messages, including binary images and signed replay blocks."""

    def encode_bytes(value: object) -> object:
        if isinstance(value, bytes):
            return {"__kcs_bytes__": base64.b64encode(value).decode("ascii")}
        raise TypeError(f"Unsupported message value: {type(value).__name__}")

    return json.dumps(
        [
            {
                "kind": int(next(kind for kind, cls in MESSAGE_TYPES.items() if type(message) is cls)),
                "data": asdict(message),
            }
            for message in messages
        ],
        default=encode_bytes,
        ensure_ascii=False,
        allow_nan=False,
    )


def decode_messages(payload: str) -> list[AnyMessage]:
    """Validate stored message envelopes and reconstruct their concrete types."""

    def decode_bytes(value: dict) -> object:
        if set(value) == {"__kcs_bytes__"}:
            return base64.b64decode(value["__kcs_bytes__"], validate=True)
        return value

    return [
        TypeAdapter(MESSAGE_TYPES[MessageKind(item["kind"])]).validate_python(item["data"])
        for item in json.loads(payload, object_hook=decode_bytes)
    ]


def _encode_context_data(
    value: Mapping[str, JsonValue] | None,
    *,
    field_name: str,
    nonempty_keys: bool = False,
) -> str:
    """Validate and encode request context stored beside one Raw Log message."""
    validated = json_object(value, field_name=field_name, nonempty_keys=nonempty_keys)
    return json.dumps(validated, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _decode_context_data(payload: str, *, field_name: str, nonempty_keys: bool = False) -> dict[str, JsonValue]:
    """Decode one Raw Log context object and reject corrupt storage values."""
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Stored {field_name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Stored {field_name} must be a JSON object")
    return json_object(value, field_name=f"Stored {field_name}", nonempty_keys=nonempty_keys)


class SQLiteSessionStorage:
    """Standalone SQLite session storage shipped with zett-agent.

    The first append creates session identity and history. Root sessions store
    no parent; delegated sessions store a plain parent ID without a foreign key.
    Message metadata and tags live in each Raw Log message JSON envelope. Only
    compaction creates snapshots.

    Example:
        storage = SQLiteSessionStorage(Path.home() / ".zett-agent" / "sessions.sqlite3")
        # Register SessionPersistenceExtension(storage), then close when finished.
        storage.close()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path.home() / ".zett-agent" / "sessions.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(URL.create("sqlite", database=str(self.path)))
        Base.metadata.create_all(self.engine)

    @contextmanager
    def _session_scope(self):
        with Session(self.engine) as session, session.begin():
            yield session

    def close(self) -> None:
        """Release the connection pool without deleting stored data."""
        self.engine.dispose()

    def create_session(
        self,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        title: str | None = None,
        agent_name: str | None = None,
    ) -> SessionSummary:
        """Create an empty session before its first Raw Log message is appended."""
        resolved_id = session_id or new_uuid7()
        if not resolved_id.strip():
            raise ValueError("session_id cannot be empty")
        if parent_session_id is not None and not parent_session_id.strip():
            raise ValueError("parent_session_id cannot be empty")
        if parent_session_id == resolved_id:
            raise ValueError("parent_session_id must differ from session_id")
        normalized_title = title.strip() if title is not None else None
        normalized_agent_name = agent_name.strip() if agent_name is not None else None
        if normalized_title is not None and (not normalized_title or len(normalized_title) > 200):
            raise ValueError("Session title must contain between 1 and 200 characters")
        if normalized_agent_name is not None and (not normalized_agent_name or len(normalized_agent_name) > 64):
            raise ValueError("Agent name must contain between 1 and 64 characters")
        now = datetime.now(UTC)
        with self._session_scope() as session:
            if session.get(AgentSessionModel, resolved_id) is not None:
                raise ValueError(f"Session already exists: {resolved_id}")
            row = AgentSessionModel(
                id=resolved_id,
                parent_session_id=parent_session_id,
                title=normalized_title,
                agent_name=normalized_agent_name,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return self._summary(session, row)

    def list_raw_messages(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
        limit: int = 10_000,
        offset: int = 0,
    ) -> list[RawMessageRecord]:
        """Return one chronological page from a session's immutable Raw Log.

        This query deliberately ignores snapshots: snapshots are compact model
        context, while a conversation UI must display the original messages.
        ``after_sequence`` and ``through_sequence`` optionally restrict the Raw
        Log sequence range; ``offset`` and ``limit`` paginate that filtered range.
        """
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        if through_sequence is not None and through_sequence < 1:
            raise ValueError("through_sequence must be positive")
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        with self._session_scope() as session:
            statement = select(RawLogMessageModel).where(
                RawLogMessageModel.session_id == session_id,
                RawLogMessageModel.sequence > after_sequence,
            )
            if through_sequence is not None:
                statement = statement.where(RawLogMessageModel.sequence <= through_sequence)
            rows = session.scalars(statement.order_by(RawLogMessageModel.sequence).offset(offset).limit(limit))
            return [self._record(row) for row in rows]

    def list_sessions(self, *, limit: int = 100, offset: int = 0) -> list[SessionSummary]:
        """Return one page of root and subagent sessions by latest activity."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        with self._session_scope() as session:
            rows = session.execute(
                select(
                    AgentSessionModel.id.label("session_id"),
                    AgentSessionModel.parent_session_id,
                    AgentSessionModel.title,
                    AgentSessionModel.agent_name,
                    func.count(RawLogMessageModel.id).label("message_count"),
                    AgentSessionModel.created_at,
                    AgentSessionModel.updated_at,
                )
                .outerjoin(RawLogMessageModel, RawLogMessageModel.session_id == AgentSessionModel.id)
                .group_by(
                    AgentSessionModel.id,
                    AgentSessionModel.parent_session_id,
                    AgentSessionModel.title,
                    AgentSessionModel.agent_name,
                    AgentSessionModel.created_at,
                    AgentSessionModel.updated_at,
                )
                .order_by(AgentSessionModel.updated_at.desc(), AgentSessionModel.id.desc())
                .offset(offset)
                .limit(limit)
            )
            return [
                SessionSummary(
                    session_id=row.session_id,
                    parent_session_id=row.parent_session_id,
                    title=row.title,
                    agent_name=row.agent_name,
                    message_count=row.message_count,
                    created_at=row.created_at.replace(tzinfo=UTC),
                    updated_at=row.updated_at.replace(tzinfo=UTC),
                )
                for row in rows
            ]

    def count_messages(self, session_id: str) -> int:
        """Count immutable raw messages without reconstructing their payloads."""
        with self._session_scope() as session:
            return int(
                session.scalar(
                    select(func.count(RawLogMessageModel.id)).where(RawLogMessageModel.session_id == session_id)
                )
                or 0
            )

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Return one session's storage metadata without loading its messages."""
        with self._session_scope() as session:
            row = session.get(AgentSessionModel, session_id)
            return self._summary(session, row) if row is not None else None

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        agent_name: str | None = None,
    ) -> SessionSummary | None:
        """Update mutable session fields and return the refreshed typed summary.

        Session identity, parent linkage, and Raw Log messages are immutable.
        Display values are normalized before persistence. ``None`` leaves that
        field unchanged; at least one field must be supplied.
        """
        if title is None and agent_name is None:
            raise ValueError("At least one session field must be supplied")
        normalized_title = title.strip() if title is not None else None
        normalized_agent_name = agent_name.strip() if agent_name is not None else None
        if normalized_title is not None and (not normalized_title or len(normalized_title) > 200):
            raise ValueError("Session title must contain between 1 and 200 characters")
        if normalized_agent_name is not None and (not normalized_agent_name or len(normalized_agent_name) > 64):
            raise ValueError("Agent name must contain between 1 and 64 characters")
        with self._session_scope() as session:
            row = session.get(AgentSessionModel, session_id)
            if row is None:
                return None
            if normalized_title is not None:
                row.title = normalized_title
            if normalized_agent_name is not None:
                row.agent_name = normalized_agent_name
            row.updated_at = datetime.now(UTC)
            session.flush()
            return self._summary(session, row)

    @staticmethod
    def _summary(session: Session, row: AgentSessionModel) -> SessionSummary:
        """Hydrate one session ORM row and its message count into a read model."""
        message_count = int(
            session.scalar(select(func.count(RawLogMessageModel.id)).where(RawLogMessageModel.session_id == row.id))
            or 0
        )
        return SessionSummary(
            session_id=row.id,
            parent_session_id=row.parent_session_id,
            title=row.title,
            agent_name=row.agent_name,
            message_count=message_count,
            created_at=row.created_at.replace(tzinfo=UTC),
            updated_at=row.updated_at.replace(tzinfo=UTC),
        )

    def delete_session(self, session_id: str) -> bool:
        """Explicitly delete raw messages and snapshots owned by one session."""
        with self._session_scope() as session:
            message_count = session.execute(
                delete(RawLogMessageModel).where(RawLogMessageModel.session_id == session_id)
            ).rowcount
            snapshot_count = session.execute(
                delete(ContextSnapshotModel).where(ContextSnapshotModel.session_id == session_id)
            ).rowcount
            session_count = session.execute(
                delete(AgentSessionModel).where(AgentSessionModel.id == session_id)
            ).rowcount
            return bool(message_count or snapshot_count or session_count)

    @staticmethod
    def _record(row: RawLogMessageModel) -> RawMessageRecord:
        """Hydrate one ORM row into the package's typed immutable read model."""
        message = decode_messages(row.message_json)[0]
        return RawMessageRecord(
            id=row.id,
            session_id=row.session_id,
            request_id=row.request_id,
            sequence=row.sequence,
            message=message,
            metadata=_decode_context_data(row.metadata_json, field_name="message metadata"),
            tags=_decode_context_data(row.tags_json, field_name="message tags", nonempty_keys=True),
            started_at=row.started_at.replace(tzinfo=UTC),
            completed_at=row.completed_at.replace(tzinfo=UTC),
            duration_ns=row.duration_ns,
            reasoning_started_at=(
                row.reasoning_started_at.replace(tzinfo=UTC) if row.reasoning_started_at is not None else None
            ),
            reasoning_completed_at=(
                row.reasoning_completed_at.replace(tzinfo=UTC) if row.reasoning_completed_at is not None else None
            ),
            reasoning_duration_ns=row.reasoning_duration_ns,
            content_started_at=(
                row.content_started_at.replace(tzinfo=UTC) if row.content_started_at is not None else None
            ),
            content_completed_at=(
                row.content_completed_at.replace(tzinfo=UTC) if row.content_completed_at is not None else None
            ),
            content_duration_ns=row.content_duration_ns,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cache_read_tokens=row.cache_read_tokens,
            cache_write_tokens=row.cache_write_tokens,
            reasoning_tokens=row.reasoning_tokens,
            created_at=row.created_at.replace(tzinfo=UTC),
            updated_at=row.updated_at.replace(tzinfo=UTC),
        )

    @staticmethod
    def _sequence(session: Session, session_id: str) -> int:
        return (
            session.scalar(
                select(func.max(RawLogMessageModel.sequence)).where(RawLogMessageModel.session_id == session_id)
            )
            or 0
        )

    @staticmethod
    def _latest(session: Session, session_id: str) -> ContextSnapshotModel | None:
        return session.scalar(
            select(ContextSnapshotModel)
            .where(ContextSnapshotModel.session_id == session_id)
            .order_by(ContextSnapshotModel.version.desc())
            .limit(1)
        )

    @staticmethod
    def _snapshot_record(row: ContextSnapshotModel) -> ContextSnapshot:
        """Hydrate and validate one checkpoint containing exactly one summary."""
        messages = decode_messages(row.message_json)
        if len(messages) != 1 or not isinstance(messages[0], CompactedMessage):
            raise ValueError("Context snapshot must contain exactly one CompactedMessage")
        return ContextSnapshot(
            id=row.id,
            session_id=row.session_id,
            version=row.version,
            compacted_message=messages[0],
            compacted_through_sequence=row.compacted_through_sequence,
            created_at=row.created_at.replace(tzinfo=UTC),
            updated_at=row.updated_at.replace(tzinfo=UTC),
        )

    async def load(self, session_id: str) -> SessionView:
        with self._session_scope() as session:
            session_row = session.get(AgentSessionModel, session_id)
            snapshot_row = self._latest(session, session_id)
            snapshot = self._snapshot_record(snapshot_row) if snapshot_row is not None else None
            compacted_through_sequence = snapshot.compacted_through_sequence if snapshot is not None else 0
            rows = session.scalars(
                select(RawLogMessageModel)
                .where(
                    RawLogMessageModel.session_id == session_id,
                    RawLogMessageModel.sequence > compacted_through_sequence,
                )
                .order_by(RawLogMessageModel.sequence)
            )
            return SessionView(
                parent_session_id=session_row.parent_session_id if session_row is not None else None,
                title=session_row.title if session_row is not None else None,
                agent_name=session_row.agent_name if session_row is not None else None,
                snapshot=snapshot,
                raw_tail=[self._record(row) for row in rows],
            )

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
        if title is not None and (not title.strip() or len(title) > 200):
            raise ValueError("Session title must contain between 1 and 200 characters")
        if agent_name is not None and (not agent_name.strip() or len(agent_name) > 64):
            raise ValueError("Agent name must contain between 1 and 64 characters")
        if usage is not None and not isinstance(message, AssistantMessage):
            raise ValueError("Model usage belongs only to assistant messages")
        encoded_metadata = _encode_context_data(metadata, field_name="Message metadata")
        encoded_tags = _encode_context_data(tags, field_name="Message tags", nonempty_keys=True)
        with self._session_scope() as session:
            sequence = self._sequence(session, session_id)
            now = datetime.now(UTC)
            session_row = session.get(AgentSessionModel, session_id)
            if session_row is None:
                session_row = AgentSessionModel(
                    id=session_id,
                    parent_session_id=parent_session_id,
                    title=title,
                    agent_name=agent_name,
                    created_at=now,
                    updated_at=now,
                )
                session.add(session_row)
            elif session_row.parent_session_id != parent_session_id:
                raise ValueError("Session parent cannot change after creation")
            else:
                if title is not None:
                    session_row.title = title
                if agent_name is not None:
                    session_row.agent_name = agent_name
                session_row.updated_at = now
            timing = timing or MessageTiming.instant()
            kind = next(kind for kind, cls in MESSAGE_TYPES.items() if type(message) is cls)
            if kind == MessageKind.CHECKPOINT:
                raise ValueError("Checkpoints belong in snapshots, not raw logs")
            session.add(
                RawLogMessageModel(
                    id=new_uuid7(),
                    session_id=session_id,
                    request_id=request_id,
                    sequence=sequence + 1,
                    role=int(kind),
                    content=message.text if isinstance(message, UserMessage) else message.content,
                    tool_name=message.name if isinstance(message, ToolMessage) else None,
                    message_json=encode_messages([message]),
                    metadata_json=encoded_metadata,
                    tags_json=encoded_tags,
                    started_at=timing.started_at,
                    completed_at=timing.completed_at,
                    duration_ns=timing.duration_ns,
                    reasoning_started_at=timing.reasoning_started_at,
                    reasoning_completed_at=timing.reasoning_completed_at,
                    reasoning_duration_ns=timing.reasoning_duration_ns,
                    content_started_at=timing.content_started_at,
                    content_completed_at=timing.content_completed_at,
                    content_duration_ns=timing.content_duration_ns,
                    input_tokens=usage.input_tokens if usage is not None else None,
                    output_tokens=usage.output_tokens if usage is not None else None,
                    cache_read_tokens=usage.cache_read_tokens if usage is not None else None,
                    cache_write_tokens=usage.cache_write_tokens if usage is not None else None,
                    reasoning_tokens=usage.reasoning_tokens if usage is not None else None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return sequence + 1

    async def snapshot(
        self,
        session_id: str,
        compacted_message: CompactedMessage,
        compacted_through_sequence: int,
        expected_version: int,
    ) -> ContextSnapshot:
        with self._session_scope() as session:
            latest = self._latest(session, session_id)
            version = latest.version if latest else 0
            if version != expected_version:
                raise ValueError("Snapshot conflict; reload the session")
            latest_raw_sequence = self._sequence(session, session_id)
            if compacted_through_sequence < 1 or compacted_through_sequence > latest_raw_sequence:
                raise ValueError("Snapshot boundary must identify an existing Raw Log message")
            if latest is not None and compacted_through_sequence <= latest.compacted_through_sequence:
                raise ValueError("Snapshot boundary must advance beyond the previous checkpoint")
            now = datetime.now(UTC)
            row = ContextSnapshotModel(
                id=new_uuid7(),
                session_id=session_id,
                version=version + 1,
                compacted_through_sequence=compacted_through_sequence,
                message_json=encode_messages([compacted_message]),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return self._snapshot_record(row)
