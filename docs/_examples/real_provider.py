"""Opt-in network example. Requires credentials and may incur provider charges."""

import asyncio
import os

from zett_agent.client import create_agent
from zett_agent.dispatcher import AgentEventDispatcher
from zett_agent.events import AgentEvent
from zett_agent.model import RetryOptions
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.tools.base import tool


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers exactly."""
    return left + right


class ConsoleEvents(AgentEventDispatcher):
    async def on_text_delta_event(self, event: AgentEvent) -> None:
        print(event.delta, end="", flush=True)


async def main() -> None:
    model = OpenAIProvider(
        model=os.environ["OPENAI_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        retry=RetryOptions(base_delay=0.5, max_delay=8, max_retries=3),
    )
    try:
        client = await create_agent(model, tools=[add], event_dispatcher=ConsoleEvents())
        await client.run("Use the add tool to calculate 20 plus 22, then explain the result.")
        print()
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
