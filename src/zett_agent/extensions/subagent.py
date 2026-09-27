"""Foreground subagent delegation through an isolated child session."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field

from ..agent import Agent, AgentRunConfig, AgentRunContext
from ..ids import new_uuid7
from ..model import AgentModel, ReasoningEffort
from ..tools.base import (
    AgentTool,
    tool,
)
from .base import AgentExtension
from .file_system import FileSystemExtension
from .sqlite import SQLiteSessionExtension
from .tool_guidelines import ToolGuidelinesExtension

TASK_TOOL_NAME = "task"

TaskDescription = Annotated[str, Field(min_length=1, max_length=120)]
TaskPrompt = Annotated[str, Field(min_length=1, max_length=100_000)]
SubAgentName = Annotated[str, Field(min_length=1, max_length=64)]


@dataclass(frozen=True, slots=True)
class SubAgentDefinition:
    """Configure one isolated agent exposed through the task tool.

    Attributes:
        name: Unique lowercase profile name selected by the parent model.
        description: Profile purpose exposed to the parent's tool guidance.
        system_prompt: Instructions used only by this child agent.
        model: Adapter used for delegated inference; caller owns its resources.
        tools: Static tools explicitly granted to the child.
        extensions: Child lifecycle extensions, including its history storage.
            The parent does not inject a separate persistence parameter.
        reasoning_effort: Default reasoning level for delegated work.
        max_iterations: Model-call budget per child input.
    """

    name: str
    description: str
    system_prompt: str
    model: AgentModel
    tools: tuple[AgentTool, ...] = ()
    extensions: tuple[AgentExtension, ...] = ()
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    max_iterations: int = 18

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_-]*", self.name) is None:
            raise ValueError("Subagent name must use lowercase letters, digits, hyphens, or underscores")
        if not self.description.strip():
            raise ValueError("Subagent description cannot be empty")
        if not self.system_prompt.strip():
            raise ValueError("Subagent system_prompt cannot be empty")
        if self.max_iterations < 1:
            raise ValueError("Subagent max_iterations must be positive")
        if len({registered.name for registered in self.tools}) != len(self.tools):
            raise ValueError("Subagent tool names must be unique")


class SubAgentResult(BaseModel):
    """Final report and durable child-session identity returned to the parent."""

    session_id: str = Field(description="New child session created for this task")
    parent_session_id: str = Field(description="Parent session that delegated this task")
    subagent_type: str = Field(description="Selected subagent definition")
    content: str = Field(description="Subagent's final report for the parent agent")


def default_subagents(model: AgentModel) -> tuple[SubAgentDefinition, ...]:
    """Return built-in profiles with their complete extension configuration."""
    return (
        SubAgentDefinition(
            name="reasoning",
            description="analyze complex decisions, tradeoffs, plans, and ambiguous problems without changing files.",
            system_prompt=(
                "You are a reasoning subagent. Analyze the delegated task independently, check assumptions, compare "
                "alternatives, and return a concise report with conclusions, risks, and unresolved questions. You do "
                "not communicate directly with the user."
            ),
            model=model,
            extensions=(SQLiteSessionExtension(), ToolGuidelinesExtension()),
            reasoning_effort=ReasoningEffort.HIGH,
        ),
        SubAgentDefinition(
            name="explore",
            description="explore a codebase broadly with read-only file search and return evidence with paths.",
            system_prompt=(
                "You are a read-only code exploration subagent. Use the available search and read tools to inspect "
                "the current workspace. Do not modify files. Return concise findings with relevant paths, symbols, "
                "and uncertainties. You do not communicate directly with the user."
            ),
            model=model,
            extensions=(SQLiteSessionExtension(), FileSystemExtension(read_only=True), ToolGuidelinesExtension()),
            reasoning_effort=ReasoningEffort.LOW,
        ),
    )


class SubAgentExtension(AgentExtension):
    """Register a task tool that runs one configured subagent in a child session.

    Child agents receive an isolated context and persist their own Raw Log under
    a UUIDv7 session whose parent_session_id points to the calling session. The
    first version is foreground-only: the parent tool waits for one final report.

    Examples:
        Usage::

            extension = SubAgentExtension()
            agent = await Agent.create(
                model,
                config=AgentRunConfig(session_id="parent"),
                extensions=[SessionPersistenceExtension(storage), extension, ToolGuidelinesExtension()],
            )
    """

    def __init__(self, subagents: Sequence[SubAgentDefinition] | None = None) -> None:
        if subagents is not None and not subagents:
            raise ValueError("At least one subagent must be configured")
        configured = None if subagents is None else {definition.name: definition for definition in subagents}
        if configured is not None and len(configured) != len(subagents):
            raise ValueError("Subagent names must be unique")
        self._configured_subagents = configured

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register the request-bound task tool and its available-agent schema."""
        subagents = self._resolve_subagents(context)
        context.register_tool(self._build_tool(context, subagents))

    def _resolve_subagents(self, context: AgentRunContext) -> Mapping[str, SubAgentDefinition]:
        """Resolve configured profiles or create defaults using the parent model."""
        if self._configured_subagents is not None:
            return self._configured_subagents
        if context.model is None:
            raise ValueError("Default subagents require an AgentRunContext model")
        return {definition.name: definition for definition in default_subagents(context.model)}

    def _build_tool(
        self,
        context: AgentRunContext,
        subagents: Mapping[str, SubAgentDefinition],
    ) -> AgentTool:
        """Build one task tool whose child is linked to the calling session."""

        @tool(name=TASK_TOOL_NAME)
        async def task(
            description: TaskDescription,
            prompt: TaskPrompt,
            subagent_type: SubAgentName,
        ) -> SubAgentResult:
            """Delegate a complex task to an isolated specialized subagent.

            Args:
                description: Short human-readable summary of the delegated task.
                prompt: Complete task context, constraints, and expected output.
                subagent_type: Name listed by the task schema and tool guidance.

            Snippet:
                task(description="Trace auth flow", prompt="Find the entry points and return paths.", subagent_type="explore")

            Guidelines:
                - Delegate only complex work that benefits from an isolated context.
                - Use direct tools for a precise file read, glob, or grep operation.
                - Include all necessary context and the expected report format in prompt.
                - Do not duplicate delegated work; use the returned report in your answer.
                - The subagent report is not shown directly to the user; summarize it yourself.
            """
            _ = description
            definition = subagents.get(subagent_type)
            if definition is None:
                available = ", ".join(subagents)
                raise ValueError(f"Unknown subagent type {subagent_type!r}; available: {available}")
            child_session_id = new_uuid7()
            child = await Agent.create(
                definition.model,
                config=AgentRunConfig(
                    session_id=child_session_id,
                    parent_session_id=context.config.session_id,
                ),
                system_prompt=definition.system_prompt,
                tools=definition.tools,
                extensions=definition.extensions,
                reasoning_effort=definition.reasoning_effort,
                max_iterations=definition.max_iterations,
            )
            result = await child.run(prompt)
            return SubAgentResult(
                session_id=child_session_id,
                parent_session_id=context.config.session_id,
                subagent_type=definition.name,
                content=result.content,
            )

        parameters = deepcopy(task.parameters)
        parameters["properties"]["subagent_type"]["enum"] = list(subagents)
        return AgentTool(
            name=task.name,
            description=task.description,
            parameters=parameters,
            handler=task.handler,
            guidelines=(
                *task.guidelines,
                *(f"Use {item.name!r} when you need to {item.description}" for item in subagents.values()),
            ),
            snippet=task.snippet,
            execution_mode=task.execution_mode,
        )
