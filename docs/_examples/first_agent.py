"""A complete offline conversation through the real Agent runtime."""

import asyncio
from collections.abc import AsyncIterator

from zett_agent import (
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    UserMessage,
    create_agent,
)


class EchoModel:
    """Deterministic teaching adapter: count user turns and echo the latest one."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        users = [message for message in request.messages if isinstance(message, UserMessage)]
        answer = f"Turn {len(users)}: {users[-1].text}"
        yield ModelEvent.text(answer)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))


async def main() -> None:
    client = await create_agent(EchoModel())
    first = await client.run("Hello")
    second = await client.run("Remember this conversation")
    assert first.content == "Turn 1: Hello"
    assert second.content == "Turn 2: Remember this conversation"
    print(first.content)
    print(second.content)


if __name__ == "__main__":
    asyncio.run(main())
