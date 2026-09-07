"""End-to-end and corner-case coverage for private goal subagents."""

import asyncio
from collections import deque

import pytest

from zett_agent import (
    GOAL_EVALUATION_TOOL_NAME,
    Agent,
    AgentConfig,
    AgentEventType,
    AgentIterationLimitError,
    AgentMessage,
    AgentProtocolError,
    AssistantMessage,
    CodingExtension,
    GoalEvaluation,
    GoalExtension,
    ImageContent,
    ImageUrlSource,
    InternalMessageExtension,
    ModelEvent,
    ModelResponse,
    ReasoningEffort,
    SubAgentDefinition,
    TextContent,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
    UserMessage,
    default_goal_subagent,
)


def decision(
    achieved: bool,
    *,
    summary: str = "Verified",
    remaining_work: list[str] | None = None,
    next_instruction: str | None = None,
) -> dict:
    """Build valid arguments for the evaluator report tool."""
    return {
        "achieved": achieved,
        "summary": summary,
        "remaining_work": remaining_work or [],
        "next_instruction": next_instruction,
    }


class PrimaryModel:
    """Return scripted parent answers and retain complete requests."""

    def __init__(self, *answers: str) -> None:
        self.answers = deque(answers)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        answer = self.answers.popleft()
        yield ModelEvent.text(answer)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))


class EvaluatorModel:
    """Drive one report tool call and one final response per child Agent."""

    def __init__(self, *decisions: dict, private_deltas: bool = False) -> None:
        self.decisions = deque(decisions)
        self.requests = []
        self.private_deltas = private_deltas

    async def stream(self, request):
        self.requests.append(request)
        if self.private_deltas:
            yield ModelEvent.reasoning("private reasoning")
            yield ModelEvent.text("private text")
        if isinstance(request.messages[-1], ToolMessage):
            self.decisions.popleft()
            message = AssistantMessage(content="Decision submitted")
        else:
            message = AssistantMessage(
                tool_calls=(ToolCall("goal-report", GOAL_EVALUATION_TOOL_NAME, self.decisions[0]),)
            )
        yield ModelEvent.completed(ModelResponse(message))


def definition(model, *, extensions=None, effort=ReasoningEffort.HIGH, max_iterations=6):
    """Build one configurable goal subagent definition for tests."""
    return SubAgentDefinition(
        name="goal",
        description="verify work",
        system_prompt="Inspect evidence and report the goal decision.",
        model=model,
        extensions=tuple(extensions or ()),
        reasoning_effort=effort,
        max_iterations=max_iterations,
    )


async def test_incomplete_result_is_injected_then_verified_with_hidden_child_events() -> None:
    primary = PrimaryModel("Draft result", "Completed result")
    evaluator = EvaluatorModel(
        decision(
            False,
            summary="Tests are missing",
            remaining_work=["Run the test suite"],
            next_instruction="Run tests and fix failures.",
        ),
        decision(True),
        private_deltas=True,
    )
    extension = GoalExtension(definition(evaluator))
    original = UserMessage(content="/goal Implement and test", attributes={"source": "editor"})
    agent = await Agent.create(
        primary,
        config=AgentConfig(session_id="parent-session"),
        extensions=[extension],
    )

    events = [event async for event in agent.stream(original)]

    assert events[-1].type is AgentEventType.RUN_COMPLETED
    assert events[-1].message.content == "Completed result"
    assert all(event.type is not AgentEventType.CUSTOM for event in events)
    assert len(primary.requests) == 2
    assert len(evaluator.requests) == 4
    internal = primary.requests[1].messages[-1]
    assert isinstance(internal, AgentMessage)
    assert "Tests are missing" in internal.content
    assert "Run the test suite" in internal.content
    details = internal.attributes["goal_extension"]
    assert details["iteration"] == 1
    assert details["max_iterations"] == 8
    assert details["max_decision_retries"] == 3
    assert details["raw_content"] == original.content
    transformed = primary.requests[0].messages[-1]
    assert isinstance(transformed, UserMessage)
    assert transformed.content.startswith("You are operating in Goal Mode.")
    assert "Target:\nImplement and test" in transformed.content
    assert transformed.attributes["source"] == "editor"
    assert transformed.attributes["goal_extension"] == {
        "raw_content": "/goal Implement and test",
        "target": "Implement and test",
    }
    assert original.content == "/goal Implement and test"
    assert sum(event.type is AgentEventType.INTERNAL_MESSAGE_STARTED for event in events) == 1
    assert sum(event.type is AgentEventType.INTERNAL_MESSAGE_COMPLETED for event in events) == 1
    assert extension._runs == {}


@pytest.mark.parametrize("message", ["Hello", "/goal", " /goal Work", "/Goal Work", "Explain /goal mode"])
async def test_non_goal_messages_bypass_evaluator(message: str) -> None:
    primary = PrimaryModel("Normal answer")
    evaluator = EvaluatorModel()
    agent = await Agent.create(
        primary,
        config=AgentConfig(session_id="parent-session"),
        extensions=[GoalExtension(definition(evaluator))],
    )

    await agent.run(message)

    assert evaluator.requests == []


async def test_goal_command_rejects_empty_text_and_cleans_state() -> None:
    extension = GoalExtension(definition(EvaluatorModel()))
    agent = await Agent.create(
        PrimaryModel("unused"),
        config=AgentConfig(session_id="parent-session"),
        extensions=[extension],
    )

    with pytest.raises(AgentProtocolError, match="non-empty goal text"):
        await agent.run("/goal    ")

    assert extension._runs == {}


async def test_multimodal_goal_preserves_raw_content_copy_and_images() -> None:
    original = UserMessage(
        content=[
            TextContent("/goal Explain this diagram"),
            ImageContent(ImageUrlSource("https://example.com/diagram.png")),
        ],
        attributes={"attachment": "diagram"},
    )
    primary = PrimaryModel("Draft", "Done")
    evaluator = EvaluatorModel(
        decision(False, summary="Missing detail", remaining_work=["Add detail"], next_instruction="Add detail."),
        decision(True),
    )
    agent = await Agent.create(
        primary,
        config=AgentConfig(session_id="parent-session"),
        extensions=[GoalExtension(definition(evaluator))],
    )

    await agent.run(original)

    internal = primary.requests[1].messages[-1]
    raw_content = internal.attributes["goal_extension"]["raw_content"]
    assert raw_content == original.content
    assert raw_content is not original.content
    transformed = primary.requests[0].messages[-1]
    assert isinstance(transformed.content[0], TextContent)
    assert transformed.content[0].text.startswith("You are operating in Goal Mode.")
    assert transformed.content[1] == original.content[1]
    assert transformed.content[1] is not original.content[1]


async def test_default_definition_reuses_parent_model_and_coding_extensions() -> None:
    class CombinedModel:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            tool_names = {item.name for item in request.tools}
            if GOAL_EVALUATION_TOOL_NAME in tool_names:
                if isinstance(request.messages[-1], ToolMessage):
                    message = AssistantMessage(content="Submitted")
                else:
                    message = AssistantMessage(
                        tool_calls=(ToolCall("report", GOAL_EVALUATION_TOOL_NAME, decision(True)),)
                    )
            else:
                message = AssistantMessage(content="Parent result")
            yield ModelEvent.completed(ModelResponse(message))

    model = CombinedModel()
    agent = await Agent.create(
        model,
        config=AgentConfig(session_id="parent-session"),
        extensions=[GoalExtension()],
    )

    await agent.run("/goal Complete the task")

    child_request = next(request for request in model.requests if request.tool_choice is None and request.tools)
    names = {item.name for item in child_request.tools}
    assert {"read_file", "write_file", "replace_in_file", "delete_file", "glob", "grep", "run_shell"} <= names
    assert GOAL_EVALUATION_TOOL_NAME in names
    assert child_request.reasoning_effort is ReasoningEffort.HIGH


def test_default_goal_subagent_uses_existing_agent_configuration() -> None:
    model = EvaluatorModel()

    configured = default_goal_subagent(model)

    assert configured.model is model
    assert configured.reasoning_effort is ReasoningEffort.HIGH
    assert configured.max_iterations == 18
    assert any(isinstance(extension, CodingExtension) for extension in configured.extensions)
    assert any(isinstance(extension, ToolGuidelinesExtension) for extension in configured.extensions)


async def test_custom_definition_controls_model_effort_extensions_and_iteration_limit() -> None:
    evaluator = EvaluatorModel(decision(True))
    configured = definition(
        evaluator,
        extensions=[CodingExtension(), ToolGuidelinesExtension()],
        effort=ReasoningEffort.MINIMAL,
        max_iterations=3,
    )
    agent = await Agent.create(
        PrimaryModel("Done"),
        config=AgentConfig(session_id="parent-session"),
        extensions=[GoalExtension(configured)],
    )

    await agent.run("/goal Verify it")

    assert evaluator.requests[0].reasoning_effort is ReasoningEffort.MINIMAL
    assert GOAL_EVALUATION_TOOL_NAME in {item.name for item in evaluator.requests[0].tools}
    assert "run_shell" in {item.name for item in evaluator.requests[0].tools}


async def test_goal_continuation_limit_is_exact_and_cleans_state() -> None:
    incomplete = decision(
        False,
        summary="Still incomplete",
        remaining_work=["One requirement remains"],
        next_instruction="Complete the remaining requirement.",
    )
    evaluator = EvaluatorModel(incomplete, incomplete, incomplete)
    extension = GoalExtension(definition(evaluator), max_iterations=2)
    primary = PrimaryModel("initial", "retry one", "retry two")
    agent = await Agent.create(
        primary,
        config=AgentConfig(session_id="parent-session"),
        extensions=[extension],
        max_internal_messages=10,
    )

    with pytest.raises(AgentIterationLimitError, match="after 2 continuation iterations"):
        await agent.run("/goal Never complete")

    assert len(primary.requests) == 3
    assert len(evaluator.requests) == 6
    assert extension._runs == {}


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_goal_extension_rejects_invalid_iteration_limits(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GoalExtension(max_iterations=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "3"])
def test_goal_extension_rejects_invalid_decision_retry_limits(value) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        GoalExtension(max_decision_retries=value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"achieved": True, "summary": "Done", "remaining_work": ["extra"], "next_instruction": None},
            "cannot contain remaining_work",
        ),
        (
            {"achieved": True, "summary": "Done", "remaining_work": [], "next_instruction": "Continue"},
            "cannot contain next_instruction",
        ),
        (
            {"achieved": False, "summary": "No", "remaining_work": [], "next_instruction": None},
            "requires remaining_work",
        ),
    ],
)
def test_goal_evaluation_rejects_inconsistent_decisions(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        GoalEvaluation.model_validate(payload)


async def test_child_agent_without_report_is_rejected() -> None:
    class NoDecisionModel:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="No report")))

    evaluator = NoDecisionModel()
    extension = GoalExtension(definition(evaluator))
    agent = await Agent.create(
        PrimaryModel("Parent result"),
        config=AgentConfig(session_id="parent-session"),
        extensions=[extension],
    )

    with pytest.raises(AgentProtocolError, match="after 4 attempts"):
        await agent.run("/goal Verify this")

    assert len(evaluator.requests) == 4
    assert "previous evaluator completed" not in evaluator.requests[0].messages[-1].text.lower()
    assert "previous evaluator completed" in evaluator.requests[1].messages[-1].text.lower()
    assert extension._runs == {}


async def test_missing_decisions_are_retried_until_one_is_reported() -> None:
    class EventuallyDecidesModel:
        def __init__(self) -> None:
            self.requests = []
            self.runs = 0

        async def stream(self, request):
            self.requests.append(request)
            if isinstance(request.messages[-1], ToolMessage):
                message = AssistantMessage(content="Submitted")
            else:
                self.runs += 1
                message = (
                    AssistantMessage(content="No decision")
                    if self.runs < 3
                    else AssistantMessage(tool_calls=(ToolCall("report", GOAL_EVALUATION_TOOL_NAME, decision(True)),))
                )
            yield ModelEvent.completed(ModelResponse(message))

    evaluator = EventuallyDecidesModel()
    agent = await Agent.create(
        PrimaryModel("Parent result"),
        config=AgentConfig(session_id="parent-session"),
        extensions=[GoalExtension(definition(evaluator), max_decision_retries=3)],
    )

    result = await agent.run("/goal Verify this")

    assert result.content == "Parent result"
    assert evaluator.runs == 3
    assert len(evaluator.requests) == 4


async def test_zero_decision_retries_runs_only_initial_attempt() -> None:
    class NoDecisionModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            self.calls += 1
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="No report")))

    evaluator = NoDecisionModel()
    agent = await Agent.create(
        PrimaryModel("Parent result"),
        config=AgentConfig(session_id="parent-session"),
        extensions=[GoalExtension(definition(evaluator), max_decision_retries=0)],
    )

    with pytest.raises(AgentProtocolError, match="after 1 attempts"):
        await agent.run("/goal Verify this")

    assert evaluator.calls == 1


async def test_tool_call_candidate_is_not_evaluated_until_parent_final_answer() -> None:
    class ToolPrimary:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            self.calls += 1
            message = (
                AssistantMessage(tool_calls=(ToolCall("1", "noop", {}),))
                if self.calls == 1
                else AssistantMessage(content="Finished after tool")
            )
            yield ModelEvent.completed(ModelResponse(message))

    from zett_agent import tool

    @tool
    def noop() -> str:
        """Perform a no-op operation.

        Snippet:
            noop()

        Guidelines:
            - Use only when a no-op is explicitly useful.
        """
        return "ok"

    evaluator = EvaluatorModel(decision(True))
    agent = await Agent.create(
        ToolPrimary(),
        config=AgentConfig(session_id="parent-session"),
        tools=[noop],
        extensions=[GoalExtension(definition(evaluator))],
    )

    await agent.run("/goal Use the tool")

    assert len(evaluator.requests) == 2


async def test_goal_state_is_cleared_when_parent_request_is_cancelled() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingModel:
        async def stream(self, request):
            started.set()
            await release.wait()
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Done")))

    extension = GoalExtension(definition(EvaluatorModel(decision(True))))
    agent = await Agent.create(
        BlockingModel(),
        config=AgentConfig(session_id="parent-session"),
        extensions=[extension],
    )

    async def consume() -> None:
        async for _ in agent.stream("/goal Goal"):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert extension._runs == {}


def test_goal_extension_coexists_with_agent_owned_internal_extension() -> None:
    agent = Agent(PrimaryModel("unused"), extensions=[GoalExtension()])

    assert len([extension for extension in agent.extensions if isinstance(extension, InternalMessageExtension)]) == 1
