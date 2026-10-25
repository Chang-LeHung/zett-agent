"""SQLite-backed session persistence exposed as an Agent extension."""

from pathlib import Path

from ..storage import SQLiteSessionStorage
from .sql import SQLSessionExtension


class SQLiteSessionExtension(SQLSessionExtension):
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
