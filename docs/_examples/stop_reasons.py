"""See a refusal for what it is: reported, not run, and never mistaken for an answer."""

import asyncio
from collections.abc import AsyncIterator

from zett_agent.agent import AgentRunConfig
from zett_agent.client import create_agent
from zett_agent.events import AgentEventType
from zett_agent.messages import AssistantMessage, ToolCall
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.stop import StopInfo, StopReason
from zett_agent.tools.base import tool

RAN: list[str] = []


@tool(guidelines="Use to delete a file.")
async def delete_file(path: str) -> str:
    """Delete one file.

    Args:
        path: File to delete.
    """
    RAN.append(path)
    return "deleted"


class RefusingModel:
    """Stream part of an answer and a tool call, then decline, as a provider can."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.text("I will start by")
        call = ToolCall("call-1", "delete_file", {"path": "/important"})
        # A real adapter fills this in from the provider's own report; here it is written out.
        stop = StopInfo(StopReason.REFUSAL, raw="refusal", category="cyber", message="This request was declined.")
        message = AssistantMessage(content="I will start by", tool_calls=(call,))
        yield ModelEvent.completed(ModelResponse(message, finish_reason="refusal", stop=stop))


async def main() -> None:
    agent = await create_agent(RefusingModel(), config=AgentRunConfig("stop-reasons"), tools=[delete_file])
    heard = []
    async for event in agent.stream("Clean up the project"):
        if event.type is AgentEventType.MODEL_STOPPED:
            assert event.stop is not None
            heard.append(event.stop.describe())
        if event.type is AgentEventType.RUN_COMPLETED:
            assert isinstance(event.message, AssistantMessage)
            message = event.message

    # The run completed, and the message says it was not an answer.
    assert message.stop is not None and message.stop.reason is StopReason.REFUSAL
    assert message.stop.category == "cyber" and not message.stop.ok
    assert heard == ["refusal (cyber): This request was declined."]

    # The tool call that came with the refusal was withheld: the tool never started.
    assert RAN == [] and message.tool_calls == ()
    assert message.stop.detail["withheld_tool_calls"] == ["delete_file"]

    # A script that prefers an exception to checking can ask for one.
    try:
        message.raise_for_stop()
    except Exception as error:
        assert type(error).__name__ == "AgentStopError"
    print("Refusal reported; the tool call was not run")


if __name__ == "__main__":
    asyncio.run(main())
