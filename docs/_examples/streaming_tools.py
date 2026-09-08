"""Run a typed tool and display reasoning, tool results, and final text."""

import asyncio
from collections.abc import AsyncIterator

from zett_agent import (
    AgentEvent,
    AgentEventDispatcher,
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    ToolCall,
    ToolMessage,
    create_agent,
    tool,
)


@tool
def add(left: int, right: int) -> int:
    """Add two integers exactly.

    Args:
        left: First operand.
        right: Second operand.

    Snippet:
        add(left=20, right=22)

    Guidelines:
        - Use this tool for exact integer addition.
    """
    return left + right


class MathModel:
    """Script a model/tool/model round trip without making a network request."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            answer = f"Answer: {request.messages[-1].content}"
            yield ModelEvent.text(answer)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))
        else:
            assert any(definition.name == "add" for definition in request.tools)
            yield ModelEvent.reasoning("Use exact arithmetic.")
            yield ModelEvent.completed(
                ModelResponse(AssistantMessage(tool_calls=(ToolCall("sum-1", "add", {"left": 20, "right": 22}),)))
            )


class ConsoleEvents(AgentEventDispatcher):
    async def on_reasoning_delta_event(self, event: AgentEvent) -> None:
        print(f"Reasoning: {event.delta}")

    async def on_tool_started_event(self, event: AgentEvent) -> None:
        call = event.tool_calls[0]
        print(f"Tool: {call.name} {call.arguments}")

    async def on_tool_completed_event(self, event: AgentEvent) -> None:
        print(f"Result: {event.message.content}")

    async def on_text_delta_event(self, event: AgentEvent) -> None:
        print(event.delta)


async def main() -> None:
    client = await create_agent(MathModel(), tools=[add], event_dispatcher=ConsoleEvents())
    reply = await client.run("What is 20 plus 22?")
    assert reply.content == "Answer: 42"


if __name__ == "__main__":
    asyncio.run(main())
