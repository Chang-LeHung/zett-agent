"""Observe persisted-message timing and provider usage without changing the loop."""

import asyncio
from collections.abc import AsyncIterator

from zett_agent.agent import AgentRunContext
from zett_agent.client import create_agent
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.events import (
    ExtensionEvent,
    MessageAppendedEvent,
)
from zett_agent.messages import AssistantMessage
from zett_agent.model import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RetryOptions,
)


class UsageObserver(AgentExtension):
    """Retain immutable observations for a demo; use a bounded sink in production."""

    def __init__(self) -> None:
        self.records: list[tuple[str, MessageAppendedEvent]] = []

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        if isinstance(event, MessageAppendedEvent) and event.usage is not None:
            self.records.append((context.config.session_id, event))


class UsageModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.reasoning("Inspect the request.")
        await asyncio.sleep(0)
        yield ModelEvent.text("Done")
        yield ModelEvent.completed(
            ModelResponse(
                AssistantMessage(content="Done", reasoning="Inspect the request."),
                usage=ModelUsage(input_tokens=100, output_tokens=20, cache_read_tokens=60, reasoning_tokens=5),
            )
        )


async def main() -> None:
    observer = UsageObserver()
    client = await create_agent(UsageModel(), extensions=[observer])
    await client.run("Inspect usage")
    _, record = observer.records[0]
    assert record.timing.duration_ns >= 0
    assert record.timing.reasoning_duration_ns is not None
    assert record.timing.content_duration_ns is not None
    assert record.usage.total_tokens == 120
    assert record.usage.cache_hit_rate == 0.6
    print("Tokens: 120; cache hit: 60%")
    print("Reasoning and content boundaries recorded")


if __name__ == "__main__":
    asyncio.run(main())
