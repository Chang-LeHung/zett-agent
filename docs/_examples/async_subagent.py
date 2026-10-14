"""Start a subagent in the background, keep working, and read its report later."""

import asyncio
import json
from collections.abc import AsyncIterator

from zett_agent.agent import AgentRunConfig
from zett_agent.client import create_agent
from zett_agent.extensions.async_subagent import AsyncSubAgentExtension, SubAgentTaskStatus
from zett_agent.extensions.subagent import SubAgentDefinition
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions


class ParentModel:
    """Start a task, do other work, then wait for the report."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        results = [message for message in request.messages if isinstance(message, ToolMessage)]
        if not results:
            call = ToolCall(
                "start-1",
                "task_start",
                {"description": "Audit auth", "prompt": "List the auth entry points", "subagent_type": "audit"},
            )
            message = AssistantMessage(tool_calls=(call,))
        elif len(results) == 1:
            # The task is running; ask for its report and wait up to ten seconds.
            task_id = json.loads(results[0].content)["task_id"]
            call = ToolCall("output-1", "task_output", {"task_id": task_id, "wait_seconds": 10})
            message = AssistantMessage(tool_calls=(call,))
        else:
            report = json.loads(results[-1].content)["content"]
            message = AssistantMessage(content=f"Background report: {report}")
        yield ModelEvent.completed(ModelResponse(message))


class ChildModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        await asyncio.sleep(0.05)  # stands in for real work the parent does not wait for
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="login, logout, refresh")))


async def main() -> None:
    definition = SubAgentDefinition(
        name="audit",
        description="audit a code area and report what it finds.",
        system_prompt="You audit code and return a short report.",
        model=ChildModel(),
    )
    extension = AsyncSubAgentExtension([definition])
    client = await create_agent(
        ParentModel(),
        config=AgentRunConfig(session_id="parent"),
        extensions=[extension, ToolGuidelinesExtension()],
    )
    try:
        print((await client.run("Audit the login code")).content)
        [task] = extension.tasks("parent")
        assert task.status is SubAgentTaskStatus.COMPLETED and task.session_id != "parent"
        print("Background task finished; child session is separate")
    finally:
        await extension.close()


if __name__ == "__main__":
    asyncio.run(main())
