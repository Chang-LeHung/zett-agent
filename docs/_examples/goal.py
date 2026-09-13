"""Run a bounded goal review using the normal Agent as a private evaluator."""

import asyncio
from collections.abc import AsyncIterator

from zett_agent import (
    GOAL_MODE_EVENT_NAME,
    AgentEvent,
    AgentEventDispatcher,
    AgentRunConfig,
    AssistantMessage,
    ExternalEvent,
    GoalExtension,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    SubAgentDefinition,
    ToolCall,
    ToolMessage,
    create_agent,
)


class DraftModel:
    retry = RetryOptions(max_retries=0)

    def __init__(self) -> None:
        self.attempt = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.attempt += 1
        answer = "A draft without tests." if self.attempt == 1 else "Implementation and tests are complete."
        yield ModelEvent.text(answer)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))


class ReviewModel:
    """Script review decisions; real evaluators must inspect actual evidence."""

    retry = RetryOptions(max_retries=0)

    def __init__(self) -> None:
        self.reviews = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            message = AssistantMessage(content="Decision submitted.")
        else:
            self.reviews += 1
            achieved = self.reviews > 1
            message = AssistantMessage(
                tool_calls=(
                    ToolCall(
                        "review-1",
                        "report_goal_evaluation",
                        {
                            "achieved": achieved,
                            "summary": "Verified" if achieved else "Tests missing",
                            "remaining_work": [] if achieved else ["Add tests"],
                            "next_instruction": None if achieved else "Add and run the tests.",
                        },
                    ),
                )
            )
        yield ModelEvent.completed(ModelResponse(message))


class GoalEvents(AgentEventDispatcher):
    async def on_internal_message_started_event(self, event: AgentEvent) -> None:
        print("Continuing after private review")


async def main() -> None:
    reviewer = ReviewModel()
    definition = SubAgentDefinition(
        name="review",
        description="verify goal completion",
        system_prompt="Inspect evidence and call report_goal_evaluation.",
        model=reviewer,
    )
    config = AgentRunConfig(session_id="goal-example", request_id="request-1")
    client = await create_agent(
        DraftModel(),
        config=config,
        extensions=[GoalExtension(definition, max_iterations=2)],
        event_dispatcher=GoalEvents(),
    )
    accepted = client.agent.emit_external_event(ExternalEvent(GOAL_MODE_EVENT_NAME, {}), config=config)
    assert accepted == ["GoalExtension"]
    result = await client.run("Implement the change with tests", config=config)
    assert reviewer.reviews == 2
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
