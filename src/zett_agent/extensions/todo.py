"""Session-scoped sequential todo tracking exposed through one model tool."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .._compat import StrEnum
from ..agent import AgentRunContext
from ..messages import AssistantMessage
from ..tools.base import (
    AgentTool,
    tool,
)
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent

TODO_WRITE_TOOL_NAME = "todo_write"

TodoContent = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000)]


class TodoStatus(StrEnum):
    """Allowed lifecycle states for one ordered todo item.

    The ``in_progress`` name matches the convention used by mainstream todo
    tools, so models select it without extra prompting.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TodoItem(BaseModel):
    """One ordered task in the request-scoped list submitted through todo_write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: TodoContent = Field(description="Task text; fixed once the list is first created.")
    status: TodoStatus = Field(
        description=(
            'Lifecycle state: "pending" (not started), "in_progress" (the one current task), or "completed" (finished).'
        )
    )


TodoList = Annotated[list[TodoItem], Field(min_length=1, max_length=100)]


class TodoWriteResult(BaseModel):
    """Validated list state returned to the model after each update."""

    model_config = ConfigDict(frozen=True)

    todos: tuple[TodoItem, ...] = Field(description="Complete ordered list after this update.")
    in_progress_index: int | None = Field(
        description="Zero-based index of the single in_progress task, or null when every task is completed"
    )
    in_progress: TodoItem | None = Field(
        description="The single in_progress task, or null when every task is completed"
    )
    completed: bool = Field(description="Whether every task in the list is completed")


class TodoWriteExtension(AgentExtension):
    """Register todo_write and enforce strictly sequential task progress.

    The model submits the complete ordered list on every call. The first write
    starts the first task as ``in_progress`` and leaves every later task
    ``pending``. Each later write may preserve the state or complete exactly one
    task, in which case the next task becomes ``in_progress``; after the last
    task every item is ``completed``. Task content and order are fixed once the
    list is created.

    The list is request-working state, not durable task storage. It is cleared
    after the request succeeds, fails, or is cancelled, so the model builds a
    fresh list on the next request.

    Example progression::

        [in_progress, pending,     pending]
        [completed,   in_progress, pending]
        [completed,   completed,   in_progress]
        [completed,   completed,   completed]

    Examples:
        Inspect progress from application code while a request is active::

            extension = TodoWriteExtension()
            client = await create_agent(model, extensions=[extension])
            async for event in client.stream("Implement and test the change"):
                if event.type is AgentEventType.TOOL_COMPLETED:
                    progress = extension.todos(event.session_id)
                    if progress is not None and progress.in_progress is not None:
                        print(progress.in_progress.content)

    .. note::
        The list is request-working state, not durable task storage. It is
        cleared after success, failure, or cancellation.

    .. seealso::
        :class:`~zett_agent.extensions.todo.TodoItem` and :class:`~zett_agent.extensions.todo.TodoWriteResult`
        expose validated progress to application code.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[TodoItem, ...]] = {}

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register a request-scoped todo_write tool bound to this session."""
        context.register_tool(self._build_tool(context))

    async def after_run(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Release todo state after a successful request finishes."""
        self.clear(context.config.session_id)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Release todo state when a request terminates with an error."""
        self.clear(context.config.session_id)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Release todo state when cancellation terminates a request."""
        if isinstance(event, RunCancelledEvent):
            self.clear(context.config.session_id)

    def todos(self, session_id: str) -> TodoWriteResult | None:
        """Return the latest validated list for one session, if it has one."""
        self._validate_session_id(session_id)
        items = self._sessions.get(session_id)
        return None if items is None else self._result(items)

    def clear(self, session_id: str) -> None:
        """Forget one session's todo list without affecting other sessions."""
        self._validate_session_id(session_id)
        self._sessions.pop(session_id, None)

    def _build_tool(self, context: AgentRunContext) -> AgentTool:
        """Create a validated tool whose state is isolated by session ID."""

        @tool(name=TODO_WRITE_TOOL_NAME)
        async def todo_write(todos: TodoList) -> TodoWriteResult:
            """Create or advance the ordered todo list for the current request.
            Allowed status values are exactly "pending", "in_progress", and "completed".
            The list is request-scoped: it is discarded when the request ends and is not
            remembered in later requests, so send the complete list, starting over each request.
            Exactly one task stays in_progress until every task is completed, and one call may
            complete at most one task. Task text and order are fixed after the first call.

            Args:
                todos: Complete ordered list with the current status of every
                    task, not just the tasks that changed.

            Snippet:
                todo_write(todos=[{"content": "Inspect code", "status": "in_progress"}])

            Guidelines:
                - The list is request-scoped: it is discarded when the request ends, so recreate it and resend every task next time.
                - Always send the complete list; never change task text or order after the first call.
                - Keep exactly one task in_progress; every later task stays pending.
                - Complete at most one task per call; then the next pending task becomes in_progress.
                - Mark a task completed only after the work is done; finish with every task completed.
            """
            return self._write(context.config.session_id, tuple(todos))

        return todo_write

    def _write(self, session_id: str, todos: tuple[TodoItem, ...]) -> TodoWriteResult:
        previous = self._sessions.get(session_id)
        completed_count = self._validate_shape(todos)
        if previous is None:
            if completed_count != 0:
                raise ValueError(
                    "The first todo_write call must start with the first task in_progress and no "
                    f"completed tasks, but {completed_count} task(s) are already completed."
                )
        else:
            self._validate_transition(previous, todos, completed_count)
        self._sessions[session_id] = todos
        return self._result(todos)

    @staticmethod
    def _validate_shape(todos: tuple[TodoItem, ...]) -> int:
        completed_count = 0
        while completed_count < len(todos) and todos[completed_count].status is TodoStatus.COMPLETED:
            completed_count += 1
        if completed_count == len(todos):
            return completed_count
        if todos[completed_count].status is not TodoStatus.IN_PROGRESS:
            raise ValueError(
                f"Exactly one task must be in_progress, but the first unfinished task (index "
                f"{completed_count}) is {todos[completed_count].status.value!r}. Mark it "
                '"in_progress" and every later task "pending".'
            )
        if any(item.status is not TodoStatus.PENDING for item in todos[completed_count + 1 :]):
            raise ValueError("Every task after the in_progress task must be pending")
        return completed_count

    @staticmethod
    def _validate_transition(
        previous: tuple[TodoItem, ...],
        todos: tuple[TodoItem, ...],
        completed_count: int,
    ) -> None:
        if len(todos) != len(previous):
            raise ValueError(
                f"A todo list cannot gain or lose tasks after the first write: resubmit all "
                f"{len(previous)} tasks in the original order."
            )
        if any(current.content != old.content for old, current in zip(previous, todos, strict=True)):
            raise ValueError(
                "Task text and order are fixed after the first write: resubmit the original content "
                "in the original order."
            )

        previous_completed = sum(item.status is TodoStatus.COMPLETED for item in previous)
        if previous_completed == len(previous):
            if todos != previous:
                raise ValueError("A completed todo list cannot be reopened or changed")
            return
        if completed_count < previous_completed:
            raise ValueError("A completed task cannot be reopened: keep every completed task completed")
        if completed_count > previous_completed + 1:
            raise ValueError(
                "todo_write completes at most one task per call, but this call completed "
                f"{completed_count - previous_completed} tasks. Resubmit the list with only the "
                "current in_progress task marked completed."
            )

    @staticmethod
    def _result(todos: tuple[TodoItem, ...]) -> TodoWriteResult:
        in_progress_index = next(
            (index for index, item in enumerate(todos) if item.status is TodoStatus.IN_PROGRESS),
            None,
        )
        in_progress = None if in_progress_index is None else todos[in_progress_index]
        return TodoWriteResult(
            todos=todos,
            in_progress_index=in_progress_index,
            in_progress=in_progress,
            completed=in_progress_index is None,
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id cannot be empty")
