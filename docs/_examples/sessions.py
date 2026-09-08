"""Restore a conversation through a new SQLite connection and a new Agent."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent import (
    AgentConfig,
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    SQLiteSessionExtension,
    UserMessage,
    create_agent,
)


class TurnModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        turn = sum(isinstance(message, UserMessage) for message in request.messages)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=f"Stored turn {turn}")))


async def main() -> None:
    # No demo touches the default database under the user's home directory.
    with TemporaryDirectory(prefix="zett-docs-") as directory:
        path = Path(directory) / "sessions.sqlite"
        config = AgentConfig(session_id="project-notes")
        first_storage = SQLiteSessionExtension(path)
        try:
            first = await create_agent(TurnModel(), config=config, extensions=[first_storage])
            print((await first.run("Remember Python", metadata={"origin": "tutorial"})).content)
        finally:
            first_storage.close()

        restored_storage = SQLiteSessionExtension(path)
        try:
            restored = await create_agent(TurnModel(), config=config, extensions=[restored_storage])
            print((await restored.run("Continue")).content)
            rows = restored_storage.list_raw_messages(config.session_id)
            assert [row.message.role.value for row in rows] == ["user", "assistant", "user", "assistant"]
            assert rows[0].metadata == {"origin": "tutorial"}
            assert len(restored_storage.list_raw_messages(config.session_id, limit=2, offset=2)) == 2
            view = await restored_storage.storage.load(config.session_id)
            assert view.snapshot is None and len(view.raw_tail) == 4
            restored_storage.update_session(config.session_id, title="Project notes")
            assert restored_storage.get_session(config.session_id).title == "Project notes"
            assert restored_storage.delete_session(config.session_id)
            assert restored_storage.list_raw_messages(config.session_id) == []
            print("Raw roles: user, assistant, user, assistant")
            print("Session restored, renamed, and deleted in a temporary database")
        finally:
            restored_storage.close()


if __name__ == "__main__":
    asyncio.run(main())
