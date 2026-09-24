"""Session-scoped sequential todo tracking exposed through one model tool."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .._compat import StrEnum
from ..agent import AgentRunContext
from ..messages import AssistantMessage
from ..tools import AgentTool, tool
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent

TODO_WRITE_TOOL_NAME = "todo_write"

TodoContent = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000)]


class TodoStatus(StrEnum):
    """Allowed lifecycle states for one ordered todo item."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


class TodoItem(BaseModel):
    """One immutable-position task submitted through todo_write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: TodoContent
    status: TodoStatus


TodoList = Annotated[list[TodoItem], Field(min_length=1, max_length=100)]


class TodoWriteResult(BaseModel):
    """Validated list state returned to the model after each update."""

    model_config = ConfigDict(frozen=True)

    todos: tuple[TodoItem, ...]
    processing_index: int | None = Field(
        description="Zero-based index of the only processing task, or null after every task completes"
    )
    processing: TodoItem | None = Field(description="Current processing task, or null after every task completes")
    completed: bool = Field(description="Whether every task in the list is completed")


class TodoWriteExtension(AgentExtension):
    """Register todo_write and enforce strictly sequential task progress.

    The first write starts the first task and leaves every later task pending.
    Each subsequent write submits the complete list and may either preserve the
    current state or complete exactly one task. When a task completes, the next
    item must become processing; after the final task, every item is completed.
    Existing task content and order cannot change during the session.

    Example progression::

        [processing, pending,    pending]
        [completed,  processing, pending]
        [completed,  completed,  processing]
        [completed,  completed,  completed]

    Examples:
        Inspect progress from application code while a request is active::

            extension = TodoWriteExtension()
            client = await create_agent(model, extensions=[extension])
            async for event in client.stream("Implement and test the change"):
                if event.type is AgentEventType.TOOL_COMPLETED:
                    progress = extension.todos(event.session_id)
                    if progress is not None and progress.processing is not None:
                        print(progress.processing.content)

    .. note::
        The list is request-working state, not durable task storage. It is
        cleared after success, failure, or cancellation.

    .. seealso::
        :class:`~zett_agent.TodoItem` and :class:`~zett_agent.TodoWriteResult`
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
            """Create or advance the ordered todo list for the current session.

            Args:
                todos: Complete ordered list with the current status of every task.

            Snippet:
                todo_write(todos=[{"content": "Inspect code", "status": "processing"}])

            Guidelines:
                - On the first call, mark only the first task as processing and every other task as pending.
                - Keep task content and order unchanged after creating the list.
                - Complete only the current task, then mark exactly the next task as processing.
                - When the final task completes, mark every task as completed with none processing.
            """
            return self._write(context.config.session_id, tuple(todos))

        return todo_write

    def _write(self, session_id: str, todos: tuple[TodoItem, ...]) -> TodoWriteResult:
        previous = self._sessions.get(session_id)
        completed_count = self._validate_shape(todos)
        if previous is None:
            if completed_count != 0:
                raise ValueError("The first todo_write call must start with the first task processing")
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
        if todos[completed_count].status is not TodoStatus.PROCESSING:
            raise ValueError("The first incomplete todo must be processing")
        if any(item.status is not TodoStatus.PENDING for item in todos[completed_count + 1 :]):
            raise ValueError("Todos after the processing task must be pending")
        return completed_count

    @staticmethod
    def _validate_transition(
        previous: tuple[TodoItem, ...],
        todos: tuple[TodoItem, ...],
        completed_count: int,
    ) -> None:
        if len(todos) != len(previous):
            raise ValueError("A todo list cannot add or remove tasks after its first write")
        if any(current.content != old.content for old, current in zip(previous, todos, strict=True)):
            raise ValueError("Todo content and order cannot change after the first write")

        previous_completed = sum(item.status is TodoStatus.COMPLETED for item in previous)
        if previous_completed == len(previous):
            if todos != previous:
                raise ValueError("A completed todo list cannot transition to another state")
            return
        if completed_count not in (previous_completed, previous_completed + 1):
            raise ValueError("todo_write can complete at most the current processing task")

    @staticmethod
    def _result(todos: tuple[TodoItem, ...]) -> TodoWriteResult:
        processing_index = next(
            (index for index, item in enumerate(todos) if item.status is TodoStatus.PROCESSING),
            None,
        )
        processing = None if processing_index is None else todos[processing_index]
        return TodoWriteResult(
            todos=todos,
            processing_index=processing_index,
            processing=processing,
            completed=processing_index is None,
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id cannot be empty")
