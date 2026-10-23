"""SQL-backed session storage, imported lazily by :mod:`zett_agent.storage`.

One schema serves SQLite, MySQL, and PostgreSQL: ids and names have explicit
lengths, and long text columns use ``LONGTEXT`` on MySQL where ``TEXT`` would cap a
message at 64 KiB. Declaring the ORM models needs SQLAlchemy at import time, so
this module stays out of ``zett_agent.storage``'s module scope. Importing the
message codec, the message kinds, or the lazy ``SQLSessionStorage`` export costs
nothing until something actually asks for a database.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from weakref import WeakKeyDictionary

from sqlalchemy import Delete, Index, Integer, String, Text, delete, event, func, inspect, select
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.engine import URL, Connection, CursorResult, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from ._compat import UTC
from .extensions.compaction import CompactedMessage
from .extensions.events import MessageTiming
from .extensions.persistence import ContextSnapshot, RawMessageRecord, SessionSummary, SessionView
from .ids import new_uuid7
from .json_types import JsonValue, json_object
from .messages import AnyMessage, AssistantMessage, ToolMessage, UserMessage
from .model import ModelUsage
from .storage import _DEFAULT_SESSION_TYPE, MESSAGE_TYPES, MessageKind, decode_messages, encode_messages
from .sync_runtime import SyncMethodsMixin

#: Long text that holds a whole message or context; MySQL ``TEXT`` stops at 64 KiB.
LongText = Text().with_variant(LONGTEXT(), "mysql")


class Base(DeclarativeBase):
    """Schema owned exclusively by zett-agent."""


class AgentSessionModel(Base):
    """One root or delegated conversation and its provider-neutral identity."""

    __tablename__ = "agent_sessions"
    __table_args__ = (Index("ix_agent_sessions_activity", "updated_at", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_session_id: Mapped[str | None] = mapped_column(String(36), index=True)
    # Every session type is an application-defined integer code.
    session_type: Mapped[int] = mapped_column(
        Integer,
        default=_DEFAULT_SESSION_TYPE,
        server_default=str(_DEFAULT_SESSION_TYPE),
    )
    title: Mapped[str | None] = mapped_column(String(200))
    agent_name: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()


class RawLogMessageModel(Base):
    """Append-only original messages; role is a numeric MessageKind."""

    __tablename__ = "raw_messages"
    __table_args__ = (Index("ux_raw_session_sequence", "session_id", "sequence", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(LongText)
    tool_name: Mapped[str | None] = mapped_column(String(200))
    message_json: Mapped[str] = mapped_column(LongText)
    metadata_json: Mapped[str] = mapped_column(LongText, default="{}")
    tags_json: Mapped[str] = mapped_column(LongText, default="{}")
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
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    version: Mapped[int] = mapped_column(Integer)
    compacted_through_sequence: Mapped[int] = mapped_column(Integer)
    message_json: Mapped[str] = mapped_column(LongText)
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()


def _create_missing_indexes(connection: Connection) -> None:
    """Add indexes that an older application version created without, on any dialect.

    ``CREATE INDEX IF NOT EXISTS`` is not portable (MySQL 8 rejects it), so the
    catalog is inspected and only the missing ones are created.
    """
    inspector = inspect(connection)
    existing = {
        table.name: {index["name"] for index in inspector.get_indexes(table.name)}
        for table in Base.metadata.sorted_tables
    }
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            if index.name not in existing.get(table.name, set()):
                index.create(connection)


def _add_missing_columns(connection: Connection) -> None:
    """Add columns an older database is missing, with a plain ``ALTER TABLE``.

    SQLAlchemy cannot alter a table portably, and the statement below is accepted by
    SQLite, MySQL, and PostgreSQL alike. New databases already have the column.
    """
    inspector = inspect(connection)
    if not inspector.has_table(AgentSessionModel.__tablename__):
        return
    columns = {column["name"] for column in inspector.get_columns(AgentSessionModel.__tablename__)}
    if "session_type" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {AgentSessionModel.__tablename__} ADD COLUMN session_type INTEGER NOT NULL DEFAULT 0"
        )


def _table_names(connection: Connection) -> set[str]:
    """Return the tables this schema owns, for tests and diagnostics."""
    return {name for name in inspect(connection).get_table_names() if name in Base.metadata.tables}


async def _deleted(session: AsyncSession, statement: Delete) -> int:
    """Run a DELETE and return how many rows it removed.

    ``execute`` is typed as returning a generic ``Result``, which has no ``rowcount``; a DELETE
    returns a ``CursorResult``, which does.
    """
    return cast("CursorResult[Any]", await session.execute(statement)).rowcount


def _session_type_code(value: int) -> int:
    """Return one application-defined session type code.

    The runtime stores the integer without interpreting it, so an application
    owns the vocabulary behind each code. Booleans, other types, and negative
    values are rejected to keep the column meaningful.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("session_type must be an integer code")
    if value < 0:
        raise ValueError("session_type cannot be negative")
    return int(value)


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


#: Async dialect each supported driver uses.
DIALECT_URLS = {
    "sqlite": "sqlite+aiosqlite",
    "mysql": "mysql+aiomysql",
    "postgresql": "postgresql+psycopg",
}
#: Extra that installs each server driver, for the error message when it is missing.
DRIVER_EXTRAS = {"mysql": "mysql", "postgresql": "postgres"}
#: Port each server dialect listens on when the caller does not say.
DEFAULT_PORTS = {"mysql": 3306, "postgresql": 5432}


def _sqlite_file(url: URL) -> Path | None:
    """Return the SQLite file a URL names, or None for another dialect or ``:memory:``."""
    database = url.database
    if url.get_backend_name() != "sqlite" or database is None or database == ":memory:":
        return None
    return Path(database).expanduser()


class SQLSessionStorage(SyncMethodsMixin):
    """Session storage in SQLite, MySQL, or PostgreSQL.

    One schema serves every dialect. The first append creates session identity and
    history. Root sessions store no parent; delegated sessions store a plain parent
    ID without a foreign key. Message metadata and tags live in each Raw Log message
    JSON envelope. Only compaction creates snapshots.

    Every operation is asynchronous and runs on SQLAlchemy's asyncio driver, so
    persistence never blocks the Agent event loop. Connection setup happens in the
    constructor; tables are created by the first awaited call, which keeps
    construction usable before a loop exists.

    Args:
        url: A SQLAlchemy URL or string, for example
            ``postgresql+psycopg://agent:secret@db:5432/sessions``. A string without
            a scheme is taken as a SQLite file path. None builds the URL from the
            arguments below.
        driver: ``sqlite``, ``mysql``, or ``postgresql``; used only when ``url`` is
            None.
        path: SQLite file. None uses ``~/.zett-agent/sessions.sqlite3``; parent
            directories are created during construction.
        host: Server host for MySQL or PostgreSQL.
        port: Server port; the dialect default when omitted.
        user: Database user.
        password: Database password. Prefer an environment variable or secret store
            over a literal in source.
        database: Schema or database name; required for a server dialect.
        echo: Log every statement, for debugging a new deployment.
        pool_size: Connections kept open per process for a server dialect.
        max_overflow: Connections opened beyond ``pool_size`` under load.
        pool_recycle: Reconnect after this many seconds, so a server-side idle
            timeout does not surface as a broken connection.
        connect_args: Extra DBAPI arguments passed straight to the driver.

    Note:
        SQLite uses one connection per loop instead of a pool, because a pooled
        SQLite connection cannot cross event loops; WAL mode keeps a second
        runtime process from blocking the first. MySQL and PostgreSQL use a normal
        pooled connection with ``pool_pre_ping``.

    Examples:
        A local file, a MySQL database, and a PostgreSQL database::

            storage = SQLSessionStorage("sessions.sqlite3")
            mysql = SQLSessionStorage.for_mysql(user="agent", password=secret, database="sessions")
            postgres = SQLSessionStorage.for_postgres(user="agent", password=secret, database="sessions")

        A URL, which covers anything SQLAlchemy supports::

            storage = SQLSessionStorage("mysql+aiomysql://agent:secret@db:3306/sessions")

    .. zett-diagram:: session-view

        +--------------------------------+          +---------------------------+
        | Snapshot                       |--------->| SessionView               |
        | compacted context + boundary   |          | active model context      |
        +--------------------------------+          +---------------------------+
                                                                  ^
                                                                  |
                                                                  |
        +--------------------------------+          +---------------------------+
        | Raw Log                        |--------->| Tail                      |
        | immutable UI history           |          | messages after boundary   |
        +--------------------------------+          +---------------------------+

    .. warning::
        The default path writes personal data. Libraries and tests should pass
        an explicit path, and a server deployment should pass credentials from the
        environment rather than source.

    .. seealso::
        :class:`~zett_agent.extensions.sql.SQLSessionExtension` integrates this
        storage with lifecycle hooks, and :doc:`/extending/storage-adapter`
        explains the generic storage boundary.
    """

    def __init__(
        self,
        url: str | URL | None = None,
        *,
        driver: str = "sqlite",
        path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 1800,
        connect_args: Mapping[str, Any] | None = None,
    ) -> None:
        self.url = self.build_url(
            url, driver=driver, path=path, host=host, port=port, user=user, password=password, database=database
        )
        self.dialect = self.url.get_backend_name()
        sqlite_file = _sqlite_file(self.url)
        #: SQLite file this storage owns, or None for a server database.
        self.path = sqlite_file
        if sqlite_file is not None:
            sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {"echo": echo, "connect_args": dict(connect_args or {})}
        if self.dialect == "sqlite":
            # NullPool keeps every connection inside the loop that opened it. One
            # instance may therefore serve an application loop, a SyncRuntime
            # thread, and a test portal without sharing loop-bound connections.
            options["poolclass"] = NullPool
        else:
            options.update(
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,
            )
        try:
            self.engine = create_async_engine(self.url, **options)
        except ModuleNotFoundError as error:
            extra = DRIVER_EXTRAS.get(self.dialect)
            hint = f'; install it with pip install "zett-agent[{extra}]"' if extra else ""
            raise ImportError(f"The {self.dialect} driver is not installed{hint}") from error
        if self.dialect == "sqlite":
            event.listen(self.engine.sync_engine, "connect", self._configure_sqlite_connection)
        self._sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self._schema_ready = False
        self._schema_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()

    @classmethod
    def build_url(
        cls,
        url: str | URL | None = None,
        *,
        driver: str = "sqlite",
        path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> URL:
        """Turn a URL, a file path, or connection parameters into one SQLAlchemy URL.

        Args:
            url: A URL, or a string without a scheme that is taken as a SQLite file.
            driver: ``sqlite``, ``mysql``, or ``postgresql`` when ``url`` is None.
            path: SQLite file for the sqlite driver.
            host: Server host for a server driver.
            port: Server port; the dialect default when omitted.
            user: Database user.
            password: Database password.
            database: Database name for a server driver; required there.

        Returns:
            The URL the storage would connect with.

        Raises:
            ValueError: The driver is unknown, or a server driver has no database.
        """
        """Turn a URL, a file path, or connection parameters into one SQLAlchemy URL."""
        """Turn a URL, a file path, or connection parameters into one SQLAlchemy URL."""
        if isinstance(url, URL):
            return url
        if isinstance(url, str):
            return (
                make_url(url)
                if "://" in url
                else URL.create(DIALECT_URLS["sqlite"], database=str(Path(url).expanduser()))
            )
        if driver not in DIALECT_URLS:
            supported = ", ".join(sorted(DIALECT_URLS))
            raise ValueError(f"driver must be one of {supported}, not {driver!r}")
        if driver == "sqlite":
            target = Path(path) if path is not None else Path.home() / ".zett-agent" / "sessions.sqlite3"
            return URL.create(DIALECT_URLS["sqlite"], database=str(target.expanduser()))
        if not database or not database.strip():
            raise ValueError(f"database is required for {driver}")
        return URL.create(
            DIALECT_URLS[driver],
            username=user,
            password=password,
            host=host or "127.0.0.1",
            port=port or DEFAULT_PORTS[driver],
            database=database,
        )

    @classmethod
    def for_sqlite(cls, path: str | Path | None = None, **options: Any) -> SQLSessionStorage:
        """Open a SQLite file, or the default one when ``path`` is None."""
        return cls(driver="sqlite", path=path, **options)

    @classmethod
    def for_mysql(
        cls,
        *,
        user: str,
        password: str,
        database: str,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORTS["mysql"],
        **options: Any,
    ) -> SQLSessionStorage:
        """Open MySQL from connection parameters instead of a URL."""
        return cls(
            driver="mysql",
            user=user,
            password=password,
            database=database,
            host=host,
            port=port,
            **options,
        )

    @classmethod
    def for_postgres(
        cls,
        *,
        user: str,
        password: str,
        database: str,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORTS["postgresql"],
        **options: Any,
    ) -> SQLSessionStorage:
        """Open PostgreSQL from connection parameters instead of a URL."""
        return cls(
            driver="postgresql",
            user=user,
            password=password,
            database=database,
            host=host,
            port=port,
            **options,
        )

    @staticmethod
    def _configure_sqlite_connection(dbapi_connection: object, _: object) -> None:
        """Use WAL and a bounded busy wait for independent runtime processes."""
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    async def _ensure_schema(self) -> None:
        """Create ORM tables once, before the first awaited statement.

        Schema creation needs the asynchronous engine, so it cannot happen in
        the constructor that callers use before an event loop exists. Every
        statement goes through a session scope, which therefore owns this
        barrier; ``create_all`` is itself idempotent.
        """
        if self._schema_ready:
            return
        async with self._schema_locks.setdefault(asyncio.get_running_loop(), asyncio.Lock()):
            if self._schema_ready:
                return
            async with self.engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
                await connection.run_sync(_add_missing_columns)
                await connection.run_sync(_create_missing_indexes)
            self._schema_ready = True

    @asynccontextmanager
    async def _session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        await self._ensure_schema()
        async with self._sessions() as session, session.begin():
            yield session

    async def close(self) -> None:
        """Release the connection pool without deleting stored data."""
        await self.engine.dispose()

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        title: str | None = None,
        agent_name: str | None = None,
        session_type: int = _DEFAULT_SESSION_TYPE,
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
        resolved_session_type = _session_type_code(session_type)
        now = datetime.now(UTC)
        async with self._session_scope() as session:
            if await session.get(AgentSessionModel, resolved_id) is not None:
                raise ValueError(f"Session already exists: {resolved_id}")
            row = AgentSessionModel(
                id=resolved_id,
                parent_session_id=parent_session_id,
                title=normalized_title,
                agent_name=normalized_agent_name,
                session_type=resolved_session_type,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            await session.flush()
            return await self._summary(session, row)

    async def list_raw_messages(
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

        Args:
            session_id: Conversation to query.
            after_sequence: Exclusive lower sequence bound, defaulting to zero.
            through_sequence: Inclusive upper bound, or None for no upper bound.
            limit: Positive page size.
            offset: Non-negative number of matching messages to skip.

        Returns:
            Typed raw records in ascending sequence order; empty if none match.

        Raises:
            ValueError: If a bound or pagination argument is outside its allowed range.
        """
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        if through_sequence is not None and through_sequence < 1:
            raise ValueError("through_sequence must be positive")
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        async with self._session_scope() as session:
            statement = select(RawLogMessageModel).where(
                RawLogMessageModel.session_id == session_id,
                RawLogMessageModel.sequence > after_sequence,
            )
            if through_sequence is not None:
                statement = statement.where(RawLogMessageModel.sequence <= through_sequence)
            rows = await session.scalars(statement.order_by(RawLogMessageModel.sequence).offset(offset).limit(limit))
            return [self._record(row) for row in rows]

    async def list_sessions(
        self,
        *,
        session_types: Sequence[int] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionSummary]:
        """Return one page of root and subagent sessions by latest activity."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        type_codes = tuple(_session_type_code(value) for value in session_types) if session_types else None
        async with self._session_scope() as session:
            statement = (
                select(
                    AgentSessionModel.id.label("session_id"),
                    AgentSessionModel.parent_session_id,
                    AgentSessionModel.session_type,
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
                    AgentSessionModel.session_type,
                    AgentSessionModel.title,
                    AgentSessionModel.agent_name,
                    AgentSessionModel.created_at,
                    AgentSessionModel.updated_at,
                )
                .order_by(AgentSessionModel.updated_at.desc(), AgentSessionModel.id.desc())
                .offset(offset)
                .limit(limit)
            )
            if type_codes is not None:
                statement = statement.where(AgentSessionModel.session_type.in_(type_codes))
            rows = await session.execute(statement)
            return [
                SessionSummary(
                    session_id=row.session_id,
                    parent_session_id=row.parent_session_id,
                    session_type=row.session_type,
                    title=row.title,
                    agent_name=row.agent_name,
                    message_count=row.message_count,
                    created_at=row.created_at.replace(tzinfo=UTC),
                    updated_at=row.updated_at.replace(tzinfo=UTC),
                )
                for row in rows
            ]

    async def count_messages(self, session_id: str) -> int:
        """Count immutable raw messages without reconstructing their payloads."""
        async with self._session_scope() as session:
            return int(
                await session.scalar(
                    select(func.count(RawLogMessageModel.id)).where(RawLogMessageModel.session_id == session_id)
                )
                or 0
            )

    async def get_session(self, session_id: str) -> SessionSummary | None:
        """Return one session's storage metadata without loading its messages."""
        async with self._session_scope() as session:
            row = await session.get(AgentSessionModel, session_id)
            return await self._summary(session, row) if row is not None else None

    async def update_session(
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
        async with self._session_scope() as session:
            row = await session.get(AgentSessionModel, session_id)
            if row is None:
                return None
            if normalized_title is not None:
                row.title = normalized_title
            if normalized_agent_name is not None:
                row.agent_name = normalized_agent_name
            row.updated_at = datetime.now(UTC)
            await session.flush()
            return await self._summary(session, row)

    @staticmethod
    async def _summary(session: AsyncSession, row: AgentSessionModel) -> SessionSummary:
        """Hydrate one session ORM row and its message count into a read model."""
        message_count = int(
            await session.scalar(
                select(func.count(RawLogMessageModel.id)).where(RawLogMessageModel.session_id == row.id)
            )
            or 0
        )
        return SessionSummary(
            session_id=row.id,
            parent_session_id=row.parent_session_id,
            session_type=row.session_type,
            title=row.title,
            agent_name=row.agent_name,
            message_count=message_count,
            created_at=row.created_at.replace(tzinfo=UTC),
            updated_at=row.updated_at.replace(tzinfo=UTC),
        )

    async def delete_session(self, session_id: str) -> bool:
        """Explicitly delete raw messages and snapshots owned by one session."""
        async with self._session_scope() as session:
            message_count = await _deleted(
                session, delete(RawLogMessageModel).where(RawLogMessageModel.session_id == session_id)
            )
            snapshot_count = await _deleted(
                session, delete(ContextSnapshotModel).where(ContextSnapshotModel.session_id == session_id)
            )
            session_count = await _deleted(session, delete(AgentSessionModel).where(AgentSessionModel.id == session_id))
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
    async def _sequence(session: AsyncSession, session_id: str) -> int:
        return (
            await session.scalar(
                select(func.max(RawLogMessageModel.sequence)).where(RawLogMessageModel.session_id == session_id)
            )
            or 0
        )

    @staticmethod
    async def _latest(session: AsyncSession, session_id: str) -> ContextSnapshotModel | None:
        return await session.scalar(
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
        """Restore the latest checkpoint and the original messages after its boundary.

        Args:
            session_id: Stable conversation identifier.

        Returns:
            A typed SessionView containing the snapshot and ordered raw tail.
            An unknown session produces an empty view without creating records.
        """
        async with self._session_scope() as session:
            session_row = await session.get(AgentSessionModel, session_id)
            snapshot_row = await self._latest(session, session_id)
            snapshot = self._snapshot_record(snapshot_row) if snapshot_row is not None else None
            compacted_through_sequence = snapshot.compacted_through_sequence if snapshot is not None else 0
            rows = await session.scalars(
                select(RawLogMessageModel)
                .where(
                    RawLogMessageModel.session_id == session_id,
                    RawLogMessageModel.sequence > compacted_through_sequence,
                )
                .order_by(RawLogMessageModel.sequence)
            )
            return SessionView(
                parent_session_id=session_row.parent_session_id if session_row is not None else None,
                session_type=session_row.session_type if session_row is not None else _DEFAULT_SESSION_TYPE,
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
        """Append one original message and return its new per-session sequence.

        Args:
            session_id: Owning conversation, created on first append if needed.
            request_id: Correlation ID; repeated IDs do not deduplicate writes.
            message: Original provider-neutral user, assistant, tool, or agent input.
            timing: Observed operation timing; None records an instantaneous append.
            parent_session_id: Immutable parent link for delegated conversations.
            title: Optional display title used when creating the session.
            agent_name: Optional display name used when creating the session.
            metadata: Request-scoped JSON data, separate from model message content.
            tags: JSON classifications with non-empty keys.
            usage: Normalized provider counters, allowed only for assistant messages.

        Returns:
            The strictly increasing sequence assigned to the stored record.

        Raises:
            ValueError: If metadata, session display values, parent linkage, or
                the message/usage combination is invalid.
        """
        if title is not None and (not title.strip() or len(title) > 200):
            raise ValueError("Session title must contain between 1 and 200 characters")
        if agent_name is not None and (not agent_name.strip() or len(agent_name) > 64):
            raise ValueError("Agent name must contain between 1 and 64 characters")
        if usage is not None and not isinstance(message, AssistantMessage):
            raise ValueError("Model usage belongs only to assistant messages")
        encoded_metadata = _encode_context_data(metadata, field_name="Message metadata")
        encoded_tags = _encode_context_data(tags, field_name="Message tags", nonempty_keys=True)
        async with self._session_scope() as session:
            sequence = await self._sequence(session, session_id)
            now = datetime.now(UTC)
            session_row = await session.get(AgentSessionModel, session_id)
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
                    content=message.text if isinstance(message, (UserMessage, ToolMessage)) else message.content,
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
            await session.flush()
            return sequence + 1

    async def snapshot(
        self,
        session_id: str,
        compacted_message: CompactedMessage,
        compacted_through_sequence: int,
        expected_version: int,
    ) -> ContextSnapshot:
        """Create an immutable compaction checkpoint using optimistic version checking.

        Args:
            session_id: Conversation whose context was compacted.
            compacted_message: Summary replacing older dialogue in model context.
            compacted_through_sequence: Last Raw Log message represented by the
                summary; later records remain the replay tail.
            expected_version: Last snapshot version seen by the caller, or zero
                before the first checkpoint.

        Returns:
            The new checkpoint with its incremented version and UTC timestamps.

        Raises:
            ValueError: If the version is stale, the boundary is outside the Raw
                Log, or the boundary does not advance beyond the previous snapshot.

        Note:
            This operation never changes or deletes original Raw Log messages.
        """
        async with self._session_scope() as session:
            latest = await self._latest(session, session_id)
            version = latest.version if latest else 0
            if version != expected_version:
                raise ValueError("Snapshot conflict; reload the session")
            latest_raw_sequence = await self._sequence(session, session_id)
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
            await session.flush()
            return self._snapshot_record(row)


class SQLiteSessionStorage(SQLSessionStorage):
    """SQLite file storage; the original name, kept for existing callers.

    Args:
        path: SQLite file, or None for ``~/.zett-agent/sessions.sqlite3``.

    Examples:
        Use a temporary database in tests::

            from pathlib import Path
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                storage = SQLiteSessionStorage(Path(directory) / "sessions.sqlite3")
                try:
                    session = await storage.create_session(title="Documentation review")
                    assert await storage.get_session(session.session_id) == session
                finally:
                    await storage.close()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(driver="sqlite", path=path)
