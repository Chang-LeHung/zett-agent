"""Reload project guidance between requests without reading any user files."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent.client import create_agent
from zett_agent.extensions.agents_md import AgentsMdExtension
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.messages import AssistantMessage, SystemMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions


class GuidanceModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        instructions = "\n".join(message.content for message in request.messages if isinstance(message, SystemMessage))
        style = "concise" if "Use concise reviews." in instructions else "careful"
        assert f"Use {style} reviews." in instructions
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=f"Project guidance: {style} reviews")))


async def main() -> None:
    with TemporaryDirectory(prefix="zett-instructions-") as directory:
        workspace = Path(directory).resolve()
        path = workspace / "AGENTS.md"
        path.write_text("Use careful reviews.\n", encoding="utf-8")
        instructions = AgentsMdExtension(directory=workspace, include_parents=False)
        assert instructions.sources() == (path,)
        client = await create_agent(GuidanceModel(), extensions=[InMemoryMessageAccumulator(), instructions])
        first = await client.run("Review the change.")
        assert first.content == "Project guidance: careful reviews"
        print(first.content)
        path.write_text("Use concise reviews.\n", encoding="utf-8")
        second = await client.run("Review again.")
        assert second.content == "Project guidance: concise reviews"
        print(second.content)


if __name__ == "__main__":
    asyncio.run(main())
