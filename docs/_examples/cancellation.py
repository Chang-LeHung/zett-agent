"""Close a streaming request and then safely reuse the same Agent."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from zett_agent.agent import AgentRunContext
from zett_agent.client import create_agent
from zett_agent.events import (
    AgentEventType,
    AgentPhase,
)
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.events import (
    ExtensionEvent,
    RunCancelledEvent,
)
from zett_agent.messages import AssistantMessage
from zett_agent.model import (
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
)


class WorkTracker(AgentExtension):
    def __init__(self) -> None:
        self.active: set[AgentRunContext] = set()

    async def before_run(self, context: AgentRunContext) -> None:
        self.active.add(context)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        if isinstance(event, RunCancelledEvent):
            self.active.discard(context)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        self.active.discard(context)

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        self.active.discard(context)


class StreamingModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.text("Partial")
        yield ModelEvent.text(" answer")
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Partial answer")))


async def main() -> None:
    tracker = WorkTracker()
    client = await create_agent(StreamingModel(), extensions=[tracker])
    async with aclosing(client.stream("Start")) as events:
        async for event in events:
            if event.type == AgentEventType.TEXT_DELTA:
                break
    assert client.agent.state.phase == AgentPhase.CANCELLED and not tracker.active
    print("Cancelled request; extension state cleared")
    print((await client.run("Try again")).content)
    assert client.agent.state.phase == AgentPhase.COMPLETED and not tracker.active


if __name__ == "__main__":
    asyncio.run(main())
