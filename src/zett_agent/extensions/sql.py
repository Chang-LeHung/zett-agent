"""Session persistence over a SQL database: SQLite, MySQL, or PostgreSQL."""

from collections.abc import Sequence

from ..storage import _DEFAULT_SESSION_TYPE, SQLSessionStorage
from .persistence import BaseSessionPersistenceExtension, RawMessageRecord, SessionSummary


class SQLSessionExtension(BaseSessionPersistenceExtension[SQLSessionStorage]):
    """Restore and save agent sessions in a database you configured yourself.

    Construct the storage with a URL or with `SQLSessionStorage.for_mysql`,
    `.for_postgres`, or `.for_sqlite`, then hand it over:

    Args:
        storage: An open :class:`~zett_agent.storage.SQLSessionStorage`. The
            extension owns it and closes it with :meth:`close`.

    Examples:
        Persist a conversation in PostgreSQL::

            storage = SQLSessionStorage.for_postgres(
                user="agent", password=os.environ["PGPASSWORD"], database="sessions"
            )
            persistence = SQLSessionExtension(storage)
            try:
                client = await create_agent(
                    model,
                    config=AgentRunConfig(session_id="demo"),
                    extensions=[persistence, ToolGuidelinesExtension()],
                )
                await client.run("Remember the project requirements.")
                messages = await persistence.list_raw_messages("demo", limit=20)
            finally:
                await persistence.close()

    Note:
        The storage creates its tables on the first awaited call. For a server
        database the application's role needs ``CREATE`` on the schema the first
        time; afterwards it only needs read and write on those three tables.
    """

    storage: SQLSessionStorage

    def __init__(self, storage: SQLSessionStorage) -> None:
        super().__init__(storage)

    async def list_raw_messages(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
        limit: int = 10_000,
        offset: int = 0,
    ) -> list[RawMessageRecord]:
        """Forward a paginated Raw Log query to the owned session storage.

        Args:
            session_id: Conversation whose immutable messages are requested.
            after_sequence: Exclusive lower sequence bound; zero starts at the beginning.
            through_sequence: Inclusive upper bound; None leaves it unbounded.
            limit: Maximum number of records returned.
            offset: Records to skip after applying sequence filters.

        Returns:
            Original messages in ascending sequence order, with timing and usage.
        """
        return await self.storage.list_raw_messages(
            session_id,
            after_sequence=after_sequence,
            through_sequence=through_sequence,
            limit=limit,
            offset=offset,
        )

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        title: str | None = None,
        agent_name: str | None = None,
        session_type: int = _DEFAULT_SESSION_TYPE,
    ) -> SessionSummary:
        """Create an empty session in the owned storage."""
        return await self.storage.create_session(
            session_id=session_id,
            parent_session_id=parent_session_id,
            title=title,
            agent_name=agent_name,
            session_type=session_type,
        )

    async def list_sessions(
        self,
        *,
        session_types: Sequence[int] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionSummary]:
        """Forward a paginated session-summary query to owned storage."""
        return await self.storage.list_sessions(session_types=session_types, limit=limit, offset=offset)

    async def get_session(self, session_id: str) -> SessionSummary | None:
        """Forward a typed session metadata lookup to the owned storage."""
        return await self.storage.get_session(session_id)

    async def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        agent_name: str | None = None,
    ) -> SessionSummary | None:
        """Forward a typed session update to the owned storage."""
        return await self.storage.update_session(session_id, title=title, agent_name=agent_name)

    async def delete_session(self, session_id: str) -> bool:
        """Delete one session and its Raw Log and snapshot records explicitly."""
        return await self.storage.delete_session(session_id)

    async def close(self) -> None:
        """Release the connection pool without deleting history."""
        await self.storage.close()
