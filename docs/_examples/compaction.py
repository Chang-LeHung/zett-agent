"""Compact old turns and verify Raw Log preservation in a temporary database."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent import (
    AgentEvent,
    AgentEventDispatcher,
    AgentRunConfig,
    AnyMessage,
    AssistantMessage,
    CompactionExtension,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    SQLiteSessionExtension,
    create_agent,
)


def demo_tokens(messages: Sequence[AnyMessage]) -> int:
    """A reproducible word-token estimate for this ASCII-only teaching example."""
    return sum(len(str(message.content).split()) for message in messages)


class AnswerModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Noted.")))


class SummaryModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.reasoning("Keep the durable preference.")
        yield ModelEvent.text("The user prefers Python.")
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="The user prefers Python.")))


class CompactionEvents(AgentEventDispatcher):
    async def on_compaction_started_event(self, event: AgentEvent) -> None:
        print("Compaction started")

    async def on_compaction_completed_event(self, event: AgentEvent) -> None:
        print(f"Compaction applied: {event.applied}")


async def main() -> None:
    with TemporaryDirectory(prefix="zett-compaction-") as directory:
        storage = SQLiteSessionExtension(Path(directory) / "sessions.sqlite")
        try:
            client = await create_agent(
                AnswerModel(),
                config=AgentRunConfig(session_id="compaction-demo"),
                extensions=[
                    storage,
                    CompactionExtension(
                        SummaryModel(), max_tokens=100, keep_recent_tokens=10, count_tokens=demo_tokens
                    ),
                ],
                event_dispatcher=CompactionEvents(),
            )
            original = "I prefer Python. " * 100
            await client.run(original)
            await client.run("Continue with Python and keep all the current request words in the recent context.")
            raw = await storage.list_raw_messages("compaction-demo")
            view = await storage.storage.load("compaction-demo")
            assert len(raw) == 6 and raw[1].message.content == original
            assert view.snapshot is not None
            assert view.snapshot.compacted_through_sequence == 3
            assert len(view.raw_tail) == 3
            print("Raw messages: 6; checkpoint boundary: 3; raw tail: 3")
        finally:
            await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
