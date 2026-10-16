"""Background subagent delegation with tools to start, inspect, wait for, and cancel tasks."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field

from .._compat import StrEnum, TypeAliasType
from ..agent import AgentRunContext
from ..ids import new_uuid7
from ..messages import AgentMessage
from ..model import ModelRequest
from ..stop import StopInfo
from ..tools.base import AgentTool, tool
from .subagent import SubAgentDefinition, SubAgentExtension, subagent_report

logger = logging.getLogger(__name__)

TASK_START_TOOL_NAME = "task_start"
TASK_LIST_TOOL_NAME = "task_list"
TASK_STATUS_TOOL_NAME = "task_status"
TASK_OUTPUT_TOOL_NAME = "task_output"
TASK_CANCEL_TOOL_NAME = "task_cancel"

TaskId = TypeAliasType("TaskId", Annotated[str, Field(min_length=1, max_length=64)])


class SubAgentTaskStatus(StrEnum):
    """Lifecycle of one background subagent task."""

    #: The child agent is still working.
    RUNNING = "running"
    #: The child finished and its report is available.
    COMPLETED = "completed"
    #: The child raised; ``error`` says why and there is no report.
    FAILED = "failed"
    #: The task was cancelled before it finished.
    CANCELLED = "cancelled"


@dataclass(eq=False, slots=True)
class SubAgentTask:
    """One background task owned by an :class:`AsyncSubAgentExtension`.

    The extension mutates a task as the child runs. Tools and hosts read
    :class:`SubAgentTaskInfo` snapshots instead of holding this object.

    Attributes:
        task_id: Identifier the parent model uses, unique within its session.
        parent_session_id: Session that started the task and may inspect it.
        session_id: Child session created for the work, for storage and replay.
        subagent_type: Name of the :class:`SubAgentDefinition` that runs it.
        description: Short summary written by the parent model.
        prompt: Complete instructions given to the child.
        status: Current lifecycle state.
        started_at: Monotonic start time, in seconds.
        finished_at: Monotonic end time, or None while running.
        content: The child's final report once completed.
        stop: Why the child stopped early, when it did not finish normally; None otherwise.
            A task whose child was refused or cut off is still ``completed``: the status says
            the child ended, and this says it did not finish.
        error: Failure description once failed.
        notified: Whether the parent was told that the task finished.
    """

    task_id: str
    parent_session_id: str
    session_id: str
    subagent_type: str
    description: str
    prompt: str
    status: SubAgentTaskStatus = SubAgentTaskStatus.RUNNING
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    content: str | None = None
    stop: StopInfo | None = None
    error: str | None = None
    notified: bool = False
    _runner: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def finished(self) -> bool:
        """Whether the task reached a terminal status."""
        return self.status is not SubAgentTaskStatus.RUNNING

    @property
    def elapsed(self) -> float:
        """Seconds the task ran, or has been running so far."""
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


class SubAgentTaskInfo(BaseModel):
    """Model-facing snapshot of one background task."""

    task_id: str = Field(description="Identifier to pass to task_status, task_output, or task_cancel")
    subagent_type: str = Field(description="Subagent definition running the task")
    description: str = Field(description="Short summary of the delegated work")
    status: SubAgentTaskStatus = Field(description="running, completed, failed, or cancelled")
    session_id: str = Field(description="Child session created for this task")
    elapsed_seconds: float = Field(description="How long the task has run or ran")
    content: str | None = Field(default=None, description="The subagent's final report, once completed")
    stop: StopInfo | None = Field(
        default=None,
        description="Why the subagent stopped early, when it did; the report may be missing or partial",
    )
    error: str | None = Field(default=None, description="Why the task failed, when it did")
    note: str | None = Field(default=None, description="What to do next with this task")


class AsyncSubAgentExtension(SubAgentExtension):
    """Run subagents in the background and let the model manage them with tools.

    The foreground :class:`~zett_agent.extensions.subagent.SubAgentExtension`
    blocks its parent until one report arrives. This extension returns at once
    and registers five tools instead:

    ``task_start``
        Start a task from a configured subagent; returns its ``task_id``.
    ``task_list``
        Show every task of the session with its status.
    ``task_status``
        Show one task without its report.
    ``task_output``
        Return the report, optionally waiting up to ``wait_seconds`` for it.
    ``task_cancel``
        Stop a running task.

    Children are ordinary isolated agents with their own session, built from the
    same :class:`~zett_agent.extensions.subagent.SubAgentDefinition` objects the
    foreground extension uses. Tasks live in memory for the lifetime of this
    extension object and are visible only to the session that started them. The
    event loop that started a task must keep running for it to finish.

    A finished task is announced to its parent at the next opportunity: before
    the next model call of a request that is still running, or when the
    session's next request starts. The notice is an ordinary
    :class:`~zett_agent.messages.AgentMessage`, so it needs no budget of
    internal continuations and can never fail the parent's request. A task whose
    result the model already read through a task tool is not announced again.
    An application that wants to wake an idle parent publishes its own
    :class:`~zett_agent.extensions.events.InternalMessageEvent` from
    :meth:`task_finished`.

    Args:
        subagents: Profiles the parent may choose; defaults to the built-in
            ``reasoning`` and ``explore`` profiles on the parent's model.
        max_concurrent: Running tasks allowed per session at once.
        max_retained: Finished tasks kept per session for later inspection;
            the oldest finished tasks are forgotten first.
        notify_parent: Announce finished tasks to the parent.

    Examples:
        Offer background delegation next to the foreground tool::

            extensions = [SubAgentExtension(), AsyncSubAgentExtension(), ToolGuidelinesExtension()]

        Change what a finished task tells the parent::

            class TerseNotices(AsyncSubAgentExtension):
                def completion_notice(self, task):
                    return f"{task.task_id} is {task.status.value}."

    .. note::
        Customize through the overridable methods instead of copying the
        extension: :meth:`build_tools` chooses the tools, :meth:`task_info`
        shapes what a tool returns, :meth:`completion_notice` writes the
        notification, :meth:`task_started` and :meth:`task_finished` observe
        the lifecycle, and :meth:`_create_child` builds each child agent.

    .. warning::
        Children run with the tools and permissions their definition grants.
        Background work keeps running after the request that started it ended,
        so call :meth:`close` when the application shuts down.
    """

    def __init__(
        self,
        subagents: Sequence[SubAgentDefinition] | None = None,
        *,
        max_concurrent: int = 4,
        max_retained: int = 100,
        notify_parent: bool = True,
    ) -> None:
        super().__init__(subagents)
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if isinstance(max_retained, bool) or not isinstance(max_retained, int) or max_retained < 0:
            raise ValueError("max_retained must be a non-negative integer")
        self.max_concurrent = max_concurrent
        self.max_retained = max_retained
        self.notify_parent = notify_parent
        self._tasks: dict[str, dict[str, SubAgentTask]] = {}
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------ hooks

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register the task tools for this request."""
        subagents = self._resolve_subagents(context)
        for registered in self.build_tools(context, subagents):
            context.register_tool(registered)

    async def on_state(self, context: AgentRunContext) -> None:
        """Announce tasks that finished while no request was active."""
        self._announce(context)

    async def before_model(self, context: AgentRunContext, request: ModelRequest) -> None:
        """Announce tasks that finished since the previous model call of this request."""
        self._announce(context)

    def _announce(self, context: AgentRunContext) -> None:
        """Add one notice per finished task the model has not been told about."""
        if not self.notify_parent:
            return
        for task in self.tasks(context.session_id):
            if task.finished and not task.notified:
                task.notified = True
                notice = self.completion_notice(task)
                if notice:
                    context.add_message(AgentMessage(content=notice))

    # -------------------------------------------------------- public registry

    def tasks(self, session_id: str) -> tuple[SubAgentTask, ...]:
        """Return the tasks one session started, oldest first."""
        return tuple(self._tasks.get(session_id, {}).values())

    def get_task(self, session_id: str, task_id: str) -> SubAgentTask:
        """Return one task of a session.

        Raises:
            ValueError: If the session has no such task. The message lists the
                known identifiers so the model can correct itself.
        """
        found = self._tasks.get(session_id, {}).get(task_id)
        if found is None:
            known = ", ".join(self._tasks.get(session_id, {})) or "none"
            raise ValueError(f"Unknown task {task_id!r}; this session has: {known}")
        return found

    async def start(
        self,
        context: AgentRunContext,
        definition: SubAgentDefinition,
        *,
        description: str,
        prompt: str,
    ) -> SubAgentTask:
        """Start one background task and return it while it runs.

        Raises:
            ValueError: If the session already runs ``max_concurrent`` tasks.
        """
        session_id = context.session_id
        running = sum(not task.finished for task in self.tasks(session_id))
        if running >= self.max_concurrent:
            raise ValueError(
                f"{running} tasks are already running (limit {self.max_concurrent}); "
                "wait for one to finish with task_output or stop one with task_cancel"
            )
        number = self._counters.get(session_id, 0) + 1
        self._counters[session_id] = number
        task = SubAgentTask(
            task_id=f"task-{number}",
            parent_session_id=session_id,
            session_id=new_uuid7(),
            subagent_type=definition.name,
            description=description,
            prompt=prompt,
        )
        self._tasks.setdefault(session_id, {})[task.task_id] = task
        task._runner = asyncio.create_task(self._execute(task, definition, context), name=f"subagent-{task.task_id}")
        return task

    async def wait(self, task: SubAgentTask, timeout: float | None = None) -> SubAgentTask:
        """Wait until a task finishes or ``timeout`` seconds pass, then return it.

        Cancelling the caller does not cancel the task.
        """
        if task._runner is not None and not task.finished:
            await asyncio.wait({task._runner}, timeout=timeout)
        return task

    async def cancel(self, task: SubAgentTask) -> SubAgentTask:
        """Cancel a running task and wait for it to settle; a finished task is unchanged."""
        runner = task._runner
        if runner is not None and not task.finished:
            runner.cancel()
            await asyncio.wait({runner})
        return task

    async def close(self) -> None:
        """Cancel every running task of every session and wait for them to settle."""
        running = [task for tasks in self._tasks.values() for task in tasks.values() if not task.finished]
        for task in running:
            if task._runner is not None:
                task._runner.cancel()
        runners = {task._runner for task in running if task._runner is not None}
        if runners:
            await asyncio.wait(runners)

    # ----------------------------------------------------- extension points

    def build_tools(
        self,
        context: AgentRunContext,
        subagents: Mapping[str, SubAgentDefinition],
    ) -> tuple[AgentTool, ...]:
        """Return the tools to register for one request.

        Override to drop a tool, add one, or replace one, calling
        ``super().build_tools(...)`` for the rest.
        """
        session_id = context.session_id
        return (
            self._start_tool(context, subagents),
            self._list_tool(session_id),
            self._status_tool(session_id),
            self._output_tool(session_id),
            self._cancel_tool(session_id),
        )

    def task_info(self, task: SubAgentTask, *, include_output: bool = False) -> SubAgentTaskInfo:
        """Describe one task for the model.

        Override to add detail. ``include_output`` is true only for the tool
        that returns a report, so listings stay short.
        """
        note: str | None = None
        match task.status:
            case SubAgentTaskStatus.RUNNING:
                note = "Still running; call task_output with wait_seconds to wait, or check back later."
            case SubAgentTaskStatus.COMPLETED if task.stop is not None:
                note = f"The subagent stopped before finishing ({task.stop.describe()}); its report is incomplete."
            case SubAgentTaskStatus.COMPLETED if not include_output:
                note = "Finished; call task_output to read the report."
            case SubAgentTaskStatus.FAILED:
                note = "The task failed and produced no report."
            case _:
                pass
        return SubAgentTaskInfo(
            task_id=task.task_id,
            subagent_type=task.subagent_type,
            description=task.description,
            status=task.status,
            session_id=task.session_id,
            elapsed_seconds=round(task.elapsed, 1),
            content=task.content if include_output else None,
            stop=task.stop,
            error=task.error,
            note=note,
        )

    def completion_notice(self, task: SubAgentTask) -> str | None:
        """Return the message that tells the parent a task finished.

        Return None to stay silent for this task; the model can still find it
        with ``task_list``.
        """
        outcome = {
            SubAgentTaskStatus.COMPLETED: "finished" if task.stop is None else f"ended early ({task.stop.describe()})",
            SubAgentTaskStatus.FAILED: f"failed: {task.error}",
            SubAgentTaskStatus.CANCELLED: "was cancelled",
        }.get(task.status, task.status.value)
        follow_up = " Call task_output to read its report." if task.status is SubAgentTaskStatus.COMPLETED else ""
        return f"Background task {task.task_id} ({task.subagent_type}: {task.description}) {outcome}.{follow_up}"

    async def task_started(self, task: SubAgentTask) -> None:
        """Observe a task that just started; the default does nothing."""

    async def task_finished(self, task: SubAgentTask) -> None:
        """Observe a task that reached a terminal status; the default does nothing.

        Its errors are logged, never raised into the task. This is the place to
        publish an :class:`~zett_agent.extensions.events.InternalMessageEvent`
        to a parent that should wake up instead of waiting for its next turn.
        """

    # --------------------------------------------------------------- running

    async def _execute(self, task: SubAgentTask, definition: SubAgentDefinition, context: AgentRunContext) -> None:
        """Run one child to completion and record how it ended."""
        cancelled = False
        try:
            await self._observe(self.task_started, task)
            child = await self._create_child(definition, task.session_id, context)
            result = await child.run(task.prompt)
            task.content = subagent_report(result)
            task.stop = result.stop if result.stop is not None and not result.stop.ok else None
            task.status = SubAgentTaskStatus.COMPLETED
        except asyncio.CancelledError:
            task.status = SubAgentTaskStatus.CANCELLED
            cancelled = True
        except Exception as error:
            task.error = f"{type(error).__name__}: {error}"
            task.status = SubAgentTaskStatus.FAILED
        task.finished_at = time.monotonic()
        await self._observe(self.task_finished, task)
        self._evict(task.parent_session_id)
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    async def _observe(hook: Callable[[SubAgentTask], Awaitable[None]], task: SubAgentTask) -> None:
        """Call a lifecycle hook so a broken override cannot break the task."""
        try:
            await hook(task)
        except Exception:
            logger.exception("Subagent task hook %s failed for %s", hook.__name__, task.task_id)

    @staticmethod
    def _seen(task: SubAgentTask) -> SubAgentTask:
        """Record that the model just read this task's outcome, so it is not announced again."""
        if task.finished:
            task.notified = True
        return task

    def _evict(self, session_id: str) -> None:
        """Forget the oldest finished tasks beyond ``max_retained``."""
        tasks = self._tasks.get(session_id, {})
        finished = [task_id for task_id, task in tasks.items() if task.finished]
        for task_id in finished[: max(0, len(finished) - self.max_retained)]:
            del tasks[task_id]

    # ----------------------------------------------------------------- tools

    def _start_tool(self, context: AgentRunContext, subagents: Mapping[str, SubAgentDefinition]) -> AgentTool:
        extension = self

        @tool(name=TASK_START_TOOL_NAME)
        async def start(
            description: Annotated[str, Field(min_length=1, max_length=120)],
            prompt: Annotated[str, Field(min_length=1, max_length=100_000)],
            subagent_type: Annotated[str, Field(min_length=1, max_length=64)],
        ) -> SubAgentTaskInfo:
            """Start a subagent in the background and return immediately with its task id.

            Args:
                description: Short human-readable summary of the delegated task.
                prompt: Complete task context, constraints, and expected output.
                subagent_type: Name listed by the task schema and tool guidance.

            Snippet:
                task_start(description="Audit auth", prompt="List every auth entry point with paths.", subagent_type="explore")

            Guidelines:
                - Use for independent work that can run while you continue; start several tasks to work in parallel.
                - The prompt must stand alone: the subagent sees none of this conversation.
                - Do not wait by polling; keep working, or call task_output with wait_seconds when you need the report.
                - You are told when a task finishes; read its report with task_output.
            """
            definition = subagents.get(subagent_type)
            if definition is None:
                raise ValueError(f"Unknown subagent type {subagent_type!r}; available: {', '.join(subagents)}")
            started = await extension.start(context, definition, description=description, prompt=prompt)
            return extension.task_info(started)

        parameters = deepcopy(start.parameters)
        parameters["properties"]["subagent_type"]["enum"] = list(subagents)
        return AgentTool(
            name=start.name,
            description=start.description,
            parameters=parameters,
            handler=start.handler,
            guidelines=(
                *start.guidelines,
                *(f"Use {item.name!r} when you need to {item.description}" for item in subagents.values()),
            ),
            snippet=start.snippet,
            execution_mode=start.execution_mode,
        )

    def _list_tool(self, session_id: str) -> AgentTool:
        extension = self

        @tool(name=TASK_LIST_TOOL_NAME)
        async def list_tasks() -> list[SubAgentTaskInfo]:
            """List this session's background tasks with their status.

            Snippet:
                task_list()

            Guidelines:
                - Use to recover task ids or to see which tasks are still running.
                - Listings omit reports; read one with task_output.
            """
            return [extension.task_info(extension._seen(task)) for task in extension.tasks(session_id)]

        return list_tasks

    def _status_tool(self, session_id: str) -> AgentTool:
        extension = self

        @tool(name=TASK_STATUS_TOOL_NAME)
        async def status(task_id: TaskId) -> SubAgentTaskInfo:
            """Show one background task's status without its report.

            Args:
                task_id: Identifier returned by task_start.

            Snippet:
                task_status(task_id="task-1")

            Guidelines:
                - Cheap to call; it never waits.
                - Use task_output once the status is completed.
            """
            return extension.task_info(extension._seen(extension.get_task(session_id, task_id)))

        return status

    def _output_tool(self, session_id: str) -> AgentTool:
        extension = self

        @tool(name=TASK_OUTPUT_TOOL_NAME)
        async def output(task_id: TaskId, wait_seconds: Annotated[int, Field(ge=0, le=300)] = 0) -> SubAgentTaskInfo:
            """Return a background task's report, waiting for it when asked.

            Args:
                task_id: Identifier returned by task_start.
                wait_seconds: Seconds to wait for a running task to finish; 0 returns at once.

            Snippet:
                task_output(task_id="task-1", wait_seconds=30)

            Guidelines:
                - A running task returns its status; call again or pass wait_seconds.
                - A failed or cancelled task has no report; read its error.
                - The report is not shown to the user; summarize it yourself.
            """
            found = extension.get_task(session_id, task_id)
            if wait_seconds and not found.finished:
                await extension.wait(found, wait_seconds)
            return extension.task_info(extension._seen(found), include_output=True)

        return output

    def _cancel_tool(self, session_id: str) -> AgentTool:
        extension = self

        @tool(name=TASK_CANCEL_TOOL_NAME)
        async def cancel(task_id: TaskId) -> SubAgentTaskInfo:
            """Stop a running background task.

            Args:
                task_id: Identifier returned by task_start.

            Snippet:
                task_cancel(task_id="task-1")

            Guidelines:
                - Use when the work is no longer needed; a finished task is left unchanged.
                - Work the task already did is not undone.
            """
            return extension.task_info(extension._seen(await extension.cancel(extension.get_task(session_id, task_id))))

        return cancel
