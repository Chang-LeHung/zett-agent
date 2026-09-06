"""SQLite-backed session persistence exposed as an Agent extension."""

from pathlib import Path

from ..storage import SQLiteSessionStorage
from .persistence import BaseSessionPersistenceExtension, RawMessageRecord, SessionSummary


class SQLiteSessionExtension(BaseSessionPersistenceExtension[SQLiteSessionStorage]):
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

    def create_session(
        self,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        title: str | None = None,
        agent_name: str | None = None,
    ) -> SessionSummary:
        """Create an empty session in the owned storage."""
        return self.storage.create_session(
            session_id=session_id,
            parent_session_id=parent_session_id,
            title=title,
            agent_name=agent_name,
        )

    def list_sessions(self, *, limit: int = 100, offset: int = 0) -> list[SessionSummary]:
        """Forward a paginated session-summary query to owned storage."""
        return self.storage.list_sessions(limit=limit, offset=offset)

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Forward a typed session metadata lookup to the owned storage."""
        return self.storage.get_session(session_id)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        agent_name: str | None = None,
    ) -> SessionSummary | None:
        """Forward a typed session update to the owned storage."""
        return self.storage.update_session(session_id, title=title, agent_name=agent_name)

    def delete_session(self, session_id: str) -> bool:
        """Delete one session and its Raw Log and snapshot records explicitly."""
        return self.storage.delete_session(session_id)

    def close(self) -> None:
        """Release the SQLite connection pool without deleting history."""
        self.storage.close()
