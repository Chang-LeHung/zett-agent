"""Run a complete agent/tool loop without credentials: uv run python examples/basic.py."""

import asyncio

from kcs_agent import (
    Agent,
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool,
)


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


class DemoModel:
    """A deterministic model: first request a tool, then read its result."""

    async def stream(self, request: ModelRequest):
        match request.messages[-1]:
            case ToolMessage(content=result):
                yield ModelEvent.text(result)
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content=result)))
            case _:
                call = ToolCall("add-1", "add", {"left": 2, "right": 3})
                yield ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(call,))))


async def main() -> None:
    agent = Agent(DemoModel(), tools=[add])
    reply = await agent.run(UserMessage(content="What is 2 + 3?"))
    print(reply.content)


if __name__ == "__main__":
    asyncio.run(main())
