import base64
import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path

from pydantic import TypeAdapter
from sqlalchemy import Index, Integer, String, Text, create_engine, delete, func, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .compaction import CompactedMessage
from .ids import new_uuid7
from .messages import AnyMessage, AssistantMessage, SystemMessage, ToolMessage, UserMessage
from .persistence import ContextSnapshot, RawMessageRecord, SessionPersistenceExtension, SessionSummary, SessionView


class Base(DeclarativeBase):
    """Schema owned exclusively by kcs-agent."""


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


MESSAGE_TYPES = {
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


class SQLiteSessionStorage:
    """Standalone SQLite session storage shipped with kcs-agent.

    No application session table is required. The first append creates a session's
    history. Only compaction creates snapshots. Pass a temporary path in tests.

    Example:
        storage = SQLiteSessionStorage(Path.home() / ".kcs-agent" / "sessions.sqlite3")
        # Register SessionPersistenceExtension(storage), then close when finished.
        storage.close()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path.home() / ".kcs-agent" / "sessions.sqlite3"
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

    def list_sessions(self, *, limit: int = 100) -> list[SessionSummary]:
        """List sessions by latest activity without requiring a separate session table."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._session_scope() as session:
            first_message_at = func.min(RawLogMessageModel.created_at).label("created_at")
            latest_message_at = func.max(RawLogMessageModel.updated_at).label("updated_at")
            rows = session.execute(
                select(
                    RawLogMessageModel.session_id,
                    func.count(RawLogMessageModel.id).label("message_count"),
                    first_message_at,
                    latest_message_at,
                )
                .group_by(RawLogMessageModel.session_id)
                .order_by(latest_message_at.desc())
                .limit(limit)
            )
            return [
                SessionSummary(
                    session_id=row.session_id,
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

    def delete_session(self, session_id: str) -> bool:
        """Explicitly delete raw messages and snapshots owned by one session."""
        with self._session_scope() as session:
            message_count = session.execute(
                delete(RawLogMessageModel).where(RawLogMessageModel.session_id == session_id)
            ).rowcount
            snapshot_count = session.execute(
                delete(ContextSnapshotModel).where(ContextSnapshotModel.session_id == session_id)
            ).rowcount
            return bool(message_count or snapshot_count)

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
            return SessionView(snapshot=snapshot, raw_tail=[self._record(row) for row in rows])

    async def append(self, session_id: str, request_id: str, message: AnyMessage) -> int:
        with self._session_scope() as session:
            sequence = self._sequence(session, session_id)
            now = datetime.now(UTC)
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


class SQLiteSessionExtension(SessionPersistenceExtension):
    """Restore and save agent sessions in an owned SQLite database."""

    storage: SQLiteSessionStorage

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(SQLiteSessionStorage(path))

    def list_raw_messages(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
        limit: int = 10_000,
        offset: int = 0,
    ) -> list[RawMessageRecord]:
        """Forward a paginated Raw Log query to the owned session storage."""
        return self.storage.list_raw_messages(
            session_id,
            after_sequence=after_sequence,
            through_sequence=through_sequence,
            limit=limit,
            offset=offset,
        )

    def close(self) -> None:
        """Release the SQLite connection pool without deleting history."""
        self.storage.close()
