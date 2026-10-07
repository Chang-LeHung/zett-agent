"""Verify independent sessions, timeout cleanup, and reuse of a cancelled session."""

import asyncio
from collections.abc import AsyncIterator

from zett_agent.agent import AgentRunConfig
from zett_agent.client import create_agent
from zett_agent.events import AgentPhase
from zett_agent.messages import AssistantMessage, UserMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions


class ApplicationModel:
    retry = RetryOptions(max_retries=0)

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        users = [message for message in request.messages if isinstance(message, UserMessage)]
        if users[-1].text == "wait":
            try:
                self.started.set()
                await asyncio.Event().wait()
            finally:
                self.closed.set()
        else:
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content=f"{users[-1].text}{len(users)}")))


async def main() -> None:
    model = ApplicationModel()
    client = await create_agent(model)
    replies = await asyncio.gather(
        client.run("A", config=AgentRunConfig(session_id="chat-a")),
        client.run("B", config=AgentRunConfig(session_id="chat-b")),
    )
    assert [reply.content for reply in replies] == ["A1", "B1"]
    print("Separate sessions: A1, B1")
    assert (await client.run("A", config=AgentRunConfig(session_id="chat-a"))).content == "A2"

    waiting = asyncio.create_task(client.run("wait", config=AgentRunConfig(session_id="timeout")))
    await model.started.wait()
    try:
        await asyncio.wait_for(waiting, timeout=0)
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("The waiting request should time out")
    assert model.closed.is_set() and not client.agent.is_conflict("timeout")
    assert client.agent.get_state("timeout").phase == AgentPhase.CANCELLED
    # Cancellation releases the session; it does not remove the recorded input.
    assert (await client.run("retry", config=AgentRunConfig(session_id="timeout"))).content == "retry2"
    print("Timeout cleaned up; session reused.")


if __name__ == "__main__":
    asyncio.run(main())
