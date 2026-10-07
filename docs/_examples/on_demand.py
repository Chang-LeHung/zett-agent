"""Load one skill through a real, deterministic tool round trip."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from zett_agent.client import create_agent
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.skill import SkillExtension
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions


class ReviewModel:
    """Ask for instructions explicitly, without pretending to perform a review."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            assert "Check correctness, edge cases" in request.messages[-1].content
            answer = "Review correctness, edge cases, and tests."
            yield ModelEvent.text(answer)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))
        else:
            assert any(definition.name == "read_skill" for definition in request.tools)
            yield ModelEvent.completed(
                ModelResponse(
                    AssistantMessage(tool_calls=(ToolCall("review-skill", "read_skill", {"name": "code-review"}),))
                )
            )


async def main() -> None:
    skills = SkillExtension([Path(__file__).parent / "skills"])
    client = await create_agent(
        ReviewModel(),
        extensions=[InMemoryMessageAccumulator(), ToolGuidelinesExtension(), skills],
    )
    reply = await client.run("Load the review instructions.")
    assert reply.content == "Review correctness, edge cases, and tests."
    print("Skill loaded: code-review")
    print(reply.content)


if __name__ == "__main__":
    asyncio.run(main())
