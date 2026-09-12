"""Delegate a task to a child profile and inspect its durable parent linkage."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent import (
    AgentRunConfig,
    AssistantMessage,
    FileSystemExtension,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    SQLiteSessionExtension,
    SubAgentDefinition,
    SubAgentExtension,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
    create_agent,
)


class ParentModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            report = json.loads(request.messages[-1].content)
            message = AssistantMessage(content=f"Child report: {report['content']}")
        else:
            message = AssistantMessage(
                tool_calls=(
                    ToolCall(
                        "task-1",
                        "task",
                        {"description": "Review design", "prompt": "Review the design", "subagent_type": "review"},
                    ),
                )
            )
        yield ModelEvent.completed(ModelResponse(message))


class ChildModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        assert {tool.name for tool in request.tools} == {"read_file", "glob", "grep"}
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Keep the runtime small.")))


async def main() -> None:
    with TemporaryDirectory(prefix="zett-subagent-") as directory:
        persistence = SQLiteSessionExtension(Path(directory) / "sessions.sqlite")
        try:
            definition = SubAgentDefinition(
                name="review",
                description="review a design with read-only tools",
                system_prompt="Review the supplied design and return a concise report.",
                model=ChildModel(),
                extensions=(persistence, FileSystemExtension(read_only=True), ToolGuidelinesExtension()),
            )
            client = await create_agent(
                ParentModel(),
                config=AgentRunConfig(session_id="parent"),
                extensions=[persistence, SubAgentExtension([definition]), ToolGuidelinesExtension()],
            )
            print((await client.run("Review the design")).content)
            sessions = persistence.list_sessions()
            children = [session for session in sessions if session.parent_session_id == "parent"]
            assert len(sessions) == 2 and len(children) == 1
            assert children[0].session_id != "parent"
            print("Two isolated sessions; child links to parent")
        finally:
            persistence.close()


if __name__ == "__main__":
    asyncio.run(main())
