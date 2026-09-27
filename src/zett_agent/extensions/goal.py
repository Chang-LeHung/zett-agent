"""Opt-in goal verification through an ordinary isolated Agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agent import Agent, AgentRunConfig, AgentRunContext
from ..exceptions import AgentIterationLimitError, AgentProtocolError
from ..ids import new_uuid7
from ..messages import AgentMessage, AssistantMessage, ImageContent, TextContent, UserContent, UserMessage
from ..model import AgentModel, ModelResponse, ReasoningEffort
from ..tools.base import tool
from .base import AgentExtension
from .coding import CodingExtension
from .events import ExtensionEvent, InternalMessageEvent, RunCancelledEvent
from .external import ExternalEvent
from .memory import InMemoryMessageAccumulator
from .subagent import SubAgentDefinition
from .tool_guidelines import ToolGuidelinesExtension

GOAL_MODE_EVENT_NAME = "goal_mode"
GOAL_EVALUATION_TOOL_NAME = "report_goal_evaluation"

GoalSummary = Annotated[str, Field(min_length=1, max_length=8_000)]
RemainingWork = Annotated[list[str], Field(max_length=100)]
NextInstruction = Annotated[str | None, Field(max_length=16_000)]


class GoalEvaluation(BaseModel):
    """Structured decision reported by the private evaluator Agent."""

    model_config = ConfigDict(extra="forbid")

    achieved: bool = Field(description="Whether the current result fully satisfies the original goal")
    summary: GoalSummary = Field(description="Concise evidence-based explanation of the decision")
    remaining_work: RemainingWork = Field(
        default_factory=list,
        description="Concrete unmet requirements; empty only when achieved is true",
    )
    next_instruction: NextInstruction = Field(
        description="Actionable instruction for the parent Agent; required when achieved is false"
    )

    @model_validator(mode="after")
    def validate_decision(self) -> GoalEvaluation:
        """Keep terminal and continuation decisions internally consistent."""
        if self.achieved:
            if self.remaining_work:
                raise ValueError("An achieved goal cannot contain remaining_work")
            if self.next_instruction is not None:
                raise ValueError("An achieved goal cannot contain next_instruction")
        elif not self.remaining_work or self.next_instruction is None or not self.next_instruction.strip():
            raise ValueError("An incomplete goal requires remaining_work and next_instruction")
        return self


@dataclass(slots=True)
class _GoalRun:
    """Request-local goal command and continuation count."""

    armed_key: tuple[str, str | None]
    goal: str | None = None
    raw_content: UserContent | None = None
    continuation_count: int = 0


def default_goal_subagent(model: AgentModel) -> SubAgentDefinition:
    """Build the standard goal evaluator with complete coding workspace access."""
    return SubAgentDefinition(
        name="goal",
        description="verify the current execution result against the original goal",
        system_prompt=(
            "You are an independent goal-verification subagent. Inspect the current workspace and execution evidence "
            "before deciding. Judge observable results, not the parent agent's claims. You may use filesystem and "
            "shell tools when verification requires them. Do not communicate with the user. Finish every task by "
            "calling report_goal_evaluation exactly once. Mark achieved=true only when every material requirement is "
            "satisfied and no required work remains. Otherwise report concrete gaps and one actionable next instruction."
        ),
        model=model,
        extensions=(CodingExtension(), ToolGuidelinesExtension()),
        reasoning_effort=ReasoningEffort.HIGH,
        max_iterations=18,
    )


class GoalExtension(AgentExtension):
    """Continue externally selected Goal Mode requests until verification passes.

    This extension does not implement another agent loop. It creates the normal
    :class:`Agent` using a :class:`SubAgentDefinition`, exactly like delegated
    subagents. The default definition installs ``CodingExtension`` and therefore
    provides read, write, replace, glob, grep, and shell tools. Its model,
    extensions, reasoning effort, and model-iteration limit all come from the
    definition and can be replaced as one unit.

    An application first sends a ``goal_mode`` :class:`ExternalEvent` with the
    :class:`AgentRunConfig` of a future request. The extension arms that exact
    session/request identity. When the matching request reaches ``on_message()``,
    its complete user input becomes the goal and is replaced with an explicit
    Goal Mode prompt before it reaches the model or Raw Log. Selection is
    consumed once; unrelated sessions and request IDs remain in normal mode.

    The transformed UserMessage
    keeps a deep copy of its original content under
    ``attributes["goal_extension"]["raw_content"]``. Evaluator AgentEvents remain
    private. An incomplete decision becomes an ``AgentMessage`` published through
    ``InternalMessageEvent``; a successful decision injects nothing and lets the
    parent request complete normally.

    ``max_iterations`` is the independent limit for goal-driven continuations;
    it defaults to eight. The parent Agent's ``max_internal_messages`` remains a
    global limit across all extensions and may stop processing earlier. When an
    evaluator returns normally without calling the decision tool,
    ``max_decision_retries`` continues the same evaluator with its history. Its default of three
    means one initial attempt plus three retries.

    Examples:
        Usage::

            definition = SubAgentDefinition(
                name="goal",
                description="verify coding work",
                system_prompt="Inspect the result, then call report_goal_evaluation.",
                model=reviewer_model,
                extensions=(CodingExtension(), ToolGuidelinesExtension()),
                reasoning_effort=ReasoningEffort.HIGH,
                max_iterations=12,
            )
            agent = await Agent.create(
                primary_model,
                config=AgentRunConfig(session_id="session-42"),
                extensions=[GoalExtension(definition, max_iterations=8, max_decision_retries=3)],
                max_internal_messages=8,
            )
            config = AgentRunConfig(session_id="session-42", request_id="request-7")
            agent.emit_external_event(ExternalEvent(GOAL_MODE_EVENT_NAME, {}), config=config)
            result = await agent.run("Implement the feature and verify its tests.", config=config)
    """

    def __init__(
        self,
        definition: SubAgentDefinition | None = None,
        *,
        max_iterations: int = 36,
        max_decision_retries: int = 3,
    ) -> None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if (
            isinstance(max_decision_retries, bool)
            or not isinstance(max_decision_retries, int)
            or max_decision_retries < 0
        ):
            raise ValueError("max_decision_retries must be a non-negative integer")
        self._definition = definition
        self.max_iterations = max_iterations
        self.max_decision_retries = max_decision_retries
        self._runs: dict[AgentRunContext, _GoalRun] = {}
        self._armed_requests: set[tuple[str, str | None]] = set()
        self._armed_lock = Lock()

    def accept(self, config: AgentRunConfig | None, event: ExternalEvent) -> bool:
        """Arm Goal Mode once for the next request matching ``config``.

        The event payload is intentionally application-owned and ignored. Mode
        routing comes only from the separately supplied AgentRunConfig, avoiding
        duplicate or conflicting session identifiers inside the payload.
        Repeating the same event before consumption is idempotent.
        """
        if event.name != GOAL_MODE_EVENT_NAME or config is None:
            return False
        with self._armed_lock:
            self._armed_requests.add((config.session_id, config.request_id))
        return True

    async def on_message(self, context: AgentRunContext) -> None:
        """Consume a matching external selection and inject the Goal Mode prompt."""
        message = context.input_message
        session_id = context.config.session_id
        request_id = context.config.request_id
        exact_key = (session_id, request_id)
        wildcard_key = (session_id, None)
        with self._armed_lock:
            armed_key = exact_key if exact_key in self._armed_requests else wildcard_key
            armed = armed_key in self._armed_requests
        if not armed:
            return
        # Register cleanup before validation so an invalid first message still
        # releases the session-level selection through on_error.
        self._runs[context] = _GoalRun(armed_key=armed_key)
        if message is None:
            raise AgentProtocolError("Goal Mode requires a user message")
        goal = message.text.strip()
        if not goal:
            raise AgentProtocolError("Goal Mode requires non-empty goal text")
        raw_content = deepcopy(message.content)
        attributes = deepcopy(message.attributes)
        attributes["goal_extension"] = {
            "raw_content": deepcopy(raw_content),
            "target": goal,
        }
        context.input_message = UserMessage(
            content=self._goal_prompt(goal, message),
            attributes=attributes,
        )
        # Keep the selection armed for the complete request lifecycle. Terminal
        # callbacks remove it together with _runs; consuming it here would lose
        # Goal Mode before a model/tool failure or cancellation is finalized.
        self._runs[context] = _GoalRun(armed_key=armed_key, goal=goal, raw_content=raw_content)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Release request-local state when cancellation bypasses run callbacks."""
        match event:
            case RunCancelledEvent():
                self._finish_run(context)

    async def after_model(
        self,
        context: AgentRunContext,
        response: ModelResponse,
    ) -> None:
        """Privately evaluate a candidate answer and enqueue one continuation."""
        if response.message.tool_calls:
            return
        run = self._runs.get(context)
        if run is None or run.goal is None or run.raw_content is None:
            return
        decision = await self._evaluate(context, run, response.message)
        if decision.achieved:
            return
        if run.continuation_count >= self.max_iterations:
            raise AgentIterationLimitError(
                f"Goal remained incomplete after {self.max_iterations} continuation iterations"
            )
        run.continuation_count += 1
        remaining = "\n".join(f"- {item}" for item in decision.remaining_work)
        await context.publish(
            InternalMessageEvent(
                AgentMessage(
                    content=(
                        "An independent goal review found that the original request is not complete.\n\n"
                        f"Decision: {decision.summary}\n\n"
                        f"Remaining work:\n{remaining}\n\n"
                        f"Next instruction: {decision.next_instruction}\n\n"
                        "Continue working on the original goal. Perform the required work and verify the result."
                    ),
                    attributes={
                        "goal_extension": {
                            "iteration": run.continuation_count,
                            "max_iterations": self.max_iterations,
                            "max_decision_retries": self.max_decision_retries,
                            "raw_content": deepcopy(run.raw_content),
                        }
                    },
                )
            )
        )

    @staticmethod
    def _goal_prompt(goal: str, message: UserMessage) -> UserContent:
        """Make Goal Mode explicit while retaining attached image evidence."""
        prompt = (
            "You are operating in Goal Mode.\n\n"
            f"Target:\n{goal}\n\n"
            "Work autonomously toward this target using the available tools. Do not stop after only describing a "
            "plan. Continue until the target is implemented and verified, then report the concrete result."
        )
        if isinstance(message.content, str):
            return prompt
        images = [deepcopy(part) for part in message.content if isinstance(part, ImageContent)]
        return [TextContent(prompt), *images]

    async def _evaluate(
        self,
        context: AgentRunContext,
        run: _GoalRun,
        result: AssistantMessage,
    ) -> GoalEvaluation:
        """Continue one child session when it finishes without submitting a decision."""
        definition = self._definition
        if definition is None:
            if context.model is None:
                raise AgentProtocolError("GoalExtension requires a model")
            definition = default_goal_subagent(context.model)
        decision: GoalEvaluation | None = None

        @tool(name=GOAL_EVALUATION_TOOL_NAME)
        def report_goal_evaluation(
            achieved: bool,
            summary: GoalSummary,
            remaining_work: RemainingWork,
            next_instruction: NextInstruction = None,
        ) -> GoalEvaluation:
            """Report whether the parent Agent's current result satisfies its goal.

            Args:
                achieved: True only when all material requirements are observably complete.
                summary: Concise evidence-based explanation of this decision.
                remaining_work: Specific unmet requirements, or an empty list when achieved.
                next_instruction: Actionable next step for the parent, or null when achieved.

            Snippet:
                report_goal_evaluation(achieved=False, summary="Tests fail", remaining_work=["Fix test_x"], next_instruction="Inspect and fix test_x.")

            Guidelines:
                - Call exactly once after inspecting enough evidence.
                - Do not report success based only on the parent Agent's claim.
                - Return concrete remaining work and one executable instruction when incomplete.
            """
            nonlocal decision
            if decision is not None:
                raise AgentProtocolError("Goal decision was already reported")
            decision = GoalEvaluation(
                achieved=achieved,
                summary=summary,
                remaining_work=remaining_work,
                next_instruction=next_instruction,
            )
            return decision

        # Keep retry history in memory for this evaluator's lifetime.
        evaluator = await Agent.create(
            definition.model,
            config=AgentRunConfig(
                session_id=new_uuid7(),
                parent_session_id=context.config.session_id,
            ),
            system_prompt=definition.system_prompt,
            tools=(*definition.tools, report_goal_evaluation),
            extensions=(*definition.extensions, InMemoryMessageAccumulator()),
            reasoning_effort=definition.reasoning_effort,
            max_iterations=definition.max_iterations,
        )
        prompt = (
            f"Original goal:\n{run.goal}\n\n"
            f"Parent Agent's current answer:\n{result.content}\n\n"
            "Inspect the workspace and relevant evidence as needed, then report your decision."
        )
        attempts = self.max_decision_retries + 1
        for _ in range(attempts):
            await evaluator.run(prompt)
            if decision is not None:
                return decision
            prompt = (
                "You completed without submitting the required decision. "
                "Continue from your existing findings and tool results; "
                "call report_goal_evaluation to submit your decision."
            )
        raise AgentProtocolError(f"Goal evaluator completed without reporting a decision after {attempts} attempts")

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Release request-local goal state after verified success."""
        _ = result
        self._finish_run(context)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Release request-local goal state after any failed request."""
        _ = error
        self._finish_run(context)

    def _finish_run(self, context: AgentRunContext) -> None:
        """Clear one activated selection only when its request reaches a terminal path."""
        run = self._runs.pop(context, None)
        if run is None:
            return
        with self._armed_lock:
            self._armed_requests.discard(run.armed_key)
