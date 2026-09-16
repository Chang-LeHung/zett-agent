"""Adapt a storage implementation without reimplementing persistence hooks."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent import (
    AgentRunConfig,
    AnyMessage,
    AssistantMessage,
    BaseSessionPersistenceExtension,
    ContextSnapshot,
    JsonValue,
    MessageTiming,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RetryOptions,
    SessionView,
    SQLiteSessionStorage,
    create_agent,
)
from zett_agent.extensions.compaction import CompactedMessage


class AuditedStorage:
    """A transparent adapter with an audit trail around an existing SQLite store."""

    def __init__(self, database: SQLiteSessionStorage) -> None:
        self.database = database
        self.operations: list[str] = []

    async def load(self, session_id: str) -> SessionView:
        self.operations.append("load")
        return await self.database.load(session_id)

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
        self.operations.append(f"append:{message.role.value}")
        return await self.database.append(
            session_id, request_id, message, timing, parent_session_id, title, agent_name, metadata, tags, usage
        )

    async def snapshot(
        self,
        session_id: str,
        compacted_message: CompactedMessage,
        compacted_through_sequence: int,
        expected_version: int,
    ) -> ContextSnapshot:
        self.operations.append("snapshot")
        return await self.database.snapshot(session_id, compacted_message, compacted_through_sequence, expected_version)


class AuditedPersistence(BaseSessionPersistenceExtension[AuditedStorage]):
    """Only construct the adapter; the base owns all lifecycle integration."""

    def __init__(self, database: SQLiteSessionStorage) -> None:
        super().__init__(AuditedStorage(database))


class FixedModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Saved.")))


async def main() -> None:
    with TemporaryDirectory(prefix="zett-storage-adapter-") as directory:
        database = SQLiteSessionStorage(Path(directory) / "sessions.sqlite")
        try:
            extension = AuditedPersistence(database)
            client = await create_agent(FixedModel(), config=AgentRunConfig(session_id="audit"), extensions=[extension])
            await client.run("Remember this")
            assert extension.storage.operations == ["load", "append:user", "append:assistant"]
            print(", ".join(extension.storage.operations))
            assert len(await database.list_raw_messages("audit")) == 2
        finally:
            await database.close()


if __name__ == "__main__":
    asyncio.run(main())
