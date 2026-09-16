"""SQLite-backed session persistence exposed as an Agent extension."""

from pathlib import Path

from ..storage import SQLiteSessionStorage
from .persistence import BaseSessionPersistenceExtension, RawMessageRecord, SessionSummary


class SQLiteSessionExtension(BaseSessionPersistenceExtension[SQLiteSessionStorage]):
    """Restore and save agent sessions in an owned SQLite database.

    Args:
        path: Database file or None for the storage default under the user data
            directory. Use an explicit temporary path in tests.

    Examples:
        Persist and query a conversation::

            persistence = SQLiteSessionExtension("sessions.sqlite")
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
        Explicit extensions replace Agent defaults. Storage restores context;
        do not add a second history restorer to the same request unnecessarily.
    """

    storage: SQLiteSessionStorage

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(SQLiteSessionStorage(path))

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
    ) -> SessionSummary:
        """Create an empty session in the owned storage."""
        return await self.storage.create_session(
            session_id=session_id,
            parent_session_id=parent_session_id,
            title=title,
            agent_name=agent_name,
        )

    async def list_sessions(self, *, limit: int = 100, offset: int = 0) -> list[SessionSummary]:
        """Forward a paginated session-summary query to owned storage."""
        return await self.storage.list_sessions(limit=limit, offset=offset)

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
        """Release the SQLite connection pool without deleting history."""
        await self.storage.close()
