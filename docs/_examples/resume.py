"""Pick a session up after a failure and after a crash, without running a tool twice."""

import asyncio
from collections.abc import AsyncIterator

import httpx

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.sqlite import SQLiteSessionExtension
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

WRITTEN: list[str] = []


@tool(guidelines="Use to write a file.")
async def write_file(name: str) -> str:
    """Write one file.

    Args:
        name: File name.
    """
    WRITTEN.append(name)
    return f"wrote {name}"


class FlakyModel:
    """Ask for a file, then lose the connection, then answer once it is back."""

    retry = RetryOptions(max_retries=0)

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        if self.calls == 1:
            call = ToolCall("call-1", "write_file", {"name": "notes.txt"})
            yield ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(call,))))
        elif self.calls == 2:
            raise httpx.ConnectError("the network went away")
        else:
            seen = [message for message in request.messages if isinstance(message, ToolMessage)]
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content=f"Done: {seen[0].content}")))


class AnswerModel:
    """Answer in one line, saying what the model was shown about the interrupted tool."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        results = [message for message in request.messages if isinstance(message, ToolMessage)]
        interrupted = [message for message in results if not message.success]
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=f"{len(interrupted)} tool was interrupted")))


async def main() -> None:
    storage = SQLiteSessionExtension("resume.sqlite")

    # A failure: the tool ran, then the connection dropped before the model saw the result.
    agent = await Agent.create(FlakyModel(), config=AgentRunConfig("failure"), tools=[write_file], extensions=[storage])
    try:
        await agent.run("Write notes.txt")
    except httpx.ConnectError:
        pass
    point = await agent.resume_point()
    assert point is not None and point.kind.value == "call_model"
    answer = await agent.resume()
    assert answer.content == "Done: wrote notes.txt"
    assert WRITTEN == ["notes.txt"]  # resuming did not run the tool a second time
    assert await agent.resume_point() is None  # and there is nothing left to resume

    # A crash: the process died after the model asked for two files and one was written.
    # Nothing could be saved for the other, so its result is missing from the history.
    calls = (ToolCall("a", "write_file", {"name": "a.txt"}), ToolCall("b", "write_file", {"name": "b.txt"}))
    await storage.storage.create_session(session_id="crash")
    for message in (
        UserMessage(content="Write a.txt and b.txt"),
        AssistantMessage(tool_calls=calls),
        ToolMessage(tool_call_id="a", name="write_file", content="wrote a.txt", success=True),
    ):
        await storage.storage.append("crash", "killed-request", message)

    agent = await Agent.create(AnswerModel(), config=AgentRunConfig("crash"), tools=[write_file], extensions=[storage])
    point = await agent.resume_point()
    assert point is not None and point.kind.value == "recover_tools"
    assert [call.id for call in point.unanswered_tool_calls] == ["b"]
    answer = await agent.resume()
    assert answer.content == "1 tool was interrupted"  # the model was told, and decides what to do
    assert WRITTEN == ["notes.txt"]  # still nothing was run on its own
    await storage.storage.close()
    print("Resumed after a failure and a crash; no tool ran twice")


if __name__ == "__main__":
    asyncio.run(main())
