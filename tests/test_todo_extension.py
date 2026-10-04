"""Sequential todo-write tool behavior across complete Agent runs."""

import asyncio
import json

import pytest

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.events import AgentEventType
from zett_agent.extensions.todo import (
    TODO_WRITE_TOOL_NAME,
    TodoStatus,
    TodoWriteExtension,
)
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)


class ScriptedModel:
    def __init__(self, *responses: AssistantMessage) -> None:
        self.responses = list(responses)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(self.responses.pop(0)))


def todo_call(call_id: str, *items: tuple[str, str]) -> AssistantMessage:
    return AssistantMessage(
        tool_calls=(
            ToolCall(
                call_id,
                TODO_WRITE_TOOL_NAME,
                {"todos": [{"content": content, "status": status} for content, status in items]},
            ),
        )
    )


def tool_result(request) -> dict:
    message = next(message for message in reversed(request.messages) if isinstance(message, ToolMessage))
    return json.loads(message.content)


async def test_todo_write_advances_every_task_in_order_until_complete():
    first = ("Inspect code", TodoStatus.IN_PROGRESS)
    second = ("Implement change", TodoStatus.PENDING)
    third = ("Run tests", TodoStatus.PENDING)
    model = ScriptedModel(
        todo_call("todo-1", first, second, third),
        todo_call(
            "todo-2",
            (first[0], TodoStatus.COMPLETED),
            (second[0], TodoStatus.IN_PROGRESS),
            third,
        ),
        todo_call(
            "todo-3",
            (first[0], TodoStatus.COMPLETED),
            (second[0], TodoStatus.COMPLETED),
            (third[0], TodoStatus.IN_PROGRESS),
        ),
        todo_call(
            "todo-4",
            (first[0], TodoStatus.COMPLETED),
            (second[0], TodoStatus.COMPLETED),
            (third[0], TodoStatus.COMPLETED),
        ),
        AssistantMessage(content="All tasks completed"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("todo-lifecycle"),
        extensions=[extension, ToolGuidelinesExtension()],
    )

    reply = await agent.run("Complete this task")

    assert reply.content == "All tasks completed"
    assert [tool_result(request)["in_progress_index"] for request in model.requests[1:]] == [0, 1, 2, None]
    assert [tool_result(request)["completed"] for request in model.requests[1:]] == [False, False, False, True]
    assert tool_result(model.requests[1])["in_progress"] == {
        "content": "Inspect code",
        "status": "in_progress",
    }
    assert tool_result(model.requests[4])["in_progress"] is None
    assert extension.todos("todo-lifecycle") is None

    definitions = {registered.name: registered for registered in model.requests[0].tools}
    assert set(definitions) == {TODO_WRITE_TOOL_NAME}
    schema = definitions[TODO_WRITE_TOOL_NAME].parameters
    assert schema["properties"]["todos"]["minItems"] == 1
    guidance = "\n".join(
        message.content for message in model.requests[0].messages if isinstance(message, SystemMessage)
    )
    assert "## todo_write" in guidance


@pytest.mark.parametrize(
    "items",
    [
        (),
        (("", TodoStatus.IN_PROGRESS),),
        (("   ", TodoStatus.IN_PROGRESS),),
        (("First", TodoStatus.PENDING),),
        (("First", TodoStatus.COMPLETED),),
        (("First", TodoStatus.IN_PROGRESS), ("Second", TodoStatus.IN_PROGRESS)),
        (("First", TodoStatus.PENDING), ("Second", TodoStatus.IN_PROGRESS)),
    ],
)
async def test_invalid_initial_todo_list_fails_without_storing_state(items):
    model = ScriptedModel(todo_call("invalid-initial", *items), AssistantMessage(content="Recovered"))
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("invalid-initial"), extensions=[extension])

    events = [event async for event in agent.stream("Start tasks")]

    failures = [event for event in events if event.type is AgentEventType.TOOL_FAILED]
    assert len(failures) == 1
    assert failures[0].message.success is False
    assert extension.todos("invalid-initial") is None


@pytest.mark.parametrize(
    "updated",
    [
        (("First", TodoStatus.COMPLETED),),
        (
            ("First", TodoStatus.COMPLETED),
            ("Second", TodoStatus.COMPLETED),
            ("Third", TodoStatus.IN_PROGRESS),
        ),
        (
            ("First", TodoStatus.PENDING),
            ("Second", TodoStatus.IN_PROGRESS),
            ("Third", TodoStatus.PENDING),
        ),
        (
            ("Changed", TodoStatus.IN_PROGRESS),
            ("Second", TodoStatus.PENDING),
            ("Third", TodoStatus.PENDING),
        ),
        (
            ("Second", TodoStatus.IN_PROGRESS),
            ("First", TodoStatus.PENDING),
            ("Third", TodoStatus.PENDING),
        ),
    ],
)
async def test_invalid_update_fails_atomically_and_preserves_current_task(updated):
    initial = (
        ("First", TodoStatus.IN_PROGRESS),
        ("Second", TodoStatus.PENDING),
        ("Third", TodoStatus.PENDING),
    )
    model = ScriptedModel(
        todo_call("initial", *initial),
        todo_call("invalid-update", *updated),
        todo_call(
            "valid-update",
            ("First", TodoStatus.COMPLETED),
            ("Second", TodoStatus.IN_PROGRESS),
            ("Third", TodoStatus.PENDING),
        ),
        AssistantMessage(content="Recovered"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("invalid-update"), extensions=[extension])

    events = [event async for event in agent.stream("Work through tasks")]

    successful = [event for event in events if event.type is AgentEventType.TOOL_COMPLETED]
    assert len(successful) == 2
    assert len([event for event in events if event.type is AgentEventType.TOOL_FAILED]) == 1
    result = tool_result(model.requests[3])
    assert result["in_progress_index"] == 1
    assert result["in_progress"]["content"] == "Second"
    assert extension.todos("invalid-update") is None


async def test_successful_requests_clear_session_state_and_clear_handles_empty_state():
    extension = TodoWriteExtension()
    first_model = ScriptedModel(
        todo_call("first", ("First session", TodoStatus.IN_PROGRESS)),
        AssistantMessage(content="done"),
    )
    second_model = ScriptedModel(
        todo_call("second", ("Second session", TodoStatus.IN_PROGRESS)),
        AssistantMessage(content="done"),
    )
    first_agent = await Agent.create(first_model, config=AgentRunConfig("first"), extensions=[extension])
    second_agent = await Agent.create(second_model, config=AgentRunConfig("second"), extensions=[extension])

    await first_agent.run("First")
    await second_agent.run("Second")

    assert extension.todos("first") is None
    assert extension.todos("second") is None
    assert extension.todos("missing") is None
    extension.clear("missing")


async def test_model_error_clears_active_todo_state():
    class FailingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield ModelEvent.completed(ModelResponse(todo_call("start", ("Task", TodoStatus.IN_PROGRESS))))
                return
            raise RuntimeError("model failed")
            yield

    extension = TodoWriteExtension()
    agent = await Agent.create(FailingModel(), config=AgentRunConfig("error-cleanup"), extensions=[extension])

    with pytest.raises(RuntimeError, match="model failed"):
        await agent.run("Start")

    assert extension.todos("error-cleanup") is None


async def test_stream_keeps_todos_until_successful_terminal_cleanup():
    model = ScriptedModel(
        todo_call("start", ("Task", TodoStatus.IN_PROGRESS)),
        AssistantMessage(content="Finished"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("stream-cleanup"), extensions=[extension])
    state_at_tool_completion = None
    state_at_run_completion = object()

    async for event in agent.stream("Start"):
        if event.type is AgentEventType.TOOL_COMPLETED:
            state_at_tool_completion = extension.todos("stream-cleanup")
        elif event.type is AgentEventType.RUN_COMPLETED:
            state_at_run_completion = extension.todos("stream-cleanup")

    assert state_at_tool_completion is not None
    assert state_at_tool_completion.in_progress is not None
    assert state_at_tool_completion.in_progress.content == "Task"
    assert state_at_run_completion is None


async def test_successful_cleanup_allows_a_fresh_list_in_the_next_run():
    model = ScriptedModel(
        todo_call("first", ("Old task", TodoStatus.IN_PROGRESS)),
        AssistantMessage(content="First run finished"),
        todo_call(
            "second",
            ("New task one", TodoStatus.IN_PROGRESS),
            ("New task two", TodoStatus.PENDING),
        ),
        AssistantMessage(content="Second run finished"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("success-reuse"), extensions=[extension])

    first_reply = await agent.run("First run")
    second_reply = await agent.run("Second run")

    assert first_reply.content == "First run finished"
    assert second_reply.content == "Second run finished"
    second_result = tool_result(model.requests[3])
    assert [item["content"] for item in second_result["todos"]] == ["New task one", "New task two"]
    assert second_result["in_progress_index"] == 0
    assert extension.todos("success-reuse") is None


async def test_cancellation_clears_active_todo_state():
    class BlockingModel:
        def __init__(self) -> None:
            self.calls = 0
            self.blocked = asyncio.Event()
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                yield ModelEvent.completed(ModelResponse(todo_call("start", ("Task", TodoStatus.IN_PROGRESS))))
                return
            self.blocked.set()
            await asyncio.Event().wait()
            yield

    model = BlockingModel()
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("cancel-cleanup"), extensions=[extension])
    task = asyncio.create_task(agent.run("Start"))
    await asyncio.wait_for(model.blocked.wait(), timeout=1)

    active = extension.todos("cancel-cleanup")
    assert active is not None and active.in_progress is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert extension.todos("cancel-cleanup") is None


async def test_cancelled_agent_can_restart_with_a_fresh_todo_list():
    class CancelThenResumeModel:
        def __init__(self) -> None:
            self.calls = 0
            self.blocked = asyncio.Event()
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            self.calls += 1
            match self.calls:
                case 1:
                    yield ModelEvent.completed(ModelResponse(todo_call("old", ("Old task", TodoStatus.IN_PROGRESS))))
                case 2:
                    self.blocked.set()
                    await asyncio.Event().wait()
                case 3:
                    yield ModelEvent.completed(
                        ModelResponse(todo_call("new", ("Replacement task", TodoStatus.IN_PROGRESS)))
                    )
                case 4:
                    yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Restarted")))

    model = CancelThenResumeModel()
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("cancel-reuse"), extensions=[extension])
    cancelled_run = asyncio.create_task(agent.run("Start old work"))
    await asyncio.wait_for(model.blocked.wait(), timeout=1)
    cancelled_run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_run

    reply = await agent.run("Start replacement work")

    assert reply.content == "Restarted"
    new_result = tool_result(model.requests[3])
    assert new_result["in_progress"]["content"] == "Replacement task"
    assert extension.todos("cancel-reuse") is None


async def test_failed_agent_can_restart_with_a_fresh_todo_list():
    class FailThenResumeModel:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            self.calls += 1
            match self.calls:
                case 1:
                    yield ModelEvent.completed(ModelResponse(todo_call("old", ("Old task", TodoStatus.IN_PROGRESS))))
                case 2:
                    raise RuntimeError("temporary model failure")
                case 3:
                    yield ModelEvent.completed(
                        ModelResponse(todo_call("new", ("Replacement task", TodoStatus.IN_PROGRESS)))
                    )
                case 4:
                    yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Recovered")))

    model = FailThenResumeModel()
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("error-reuse"), extensions=[extension])
    with pytest.raises(RuntimeError, match="temporary model failure"):
        await agent.run("Start old work")

    reply = await agent.run("Start replacement work")

    assert reply.content == "Recovered"
    new_result = tool_result(model.requests[3])
    assert new_result["in_progress"]["content"] == "Replacement task"
    assert extension.todos("error-reuse") is None


async def test_request_without_todo_calls_completes_with_empty_state():
    extension = TodoWriteExtension()
    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="No task list needed")),
        config=AgentRunConfig("no-todos"),
        extensions=[extension],
    )

    reply = await agent.run("Answer directly")

    assert reply.content == "No task list needed"
    assert extension.todos("no-todos") is None


async def test_completed_list_can_be_repeated_but_cannot_be_reopened():
    processing = (("Only task", TodoStatus.IN_PROGRESS),)
    completed = (("Only task", TodoStatus.COMPLETED),)
    model = ScriptedModel(
        todo_call("start", *processing),
        todo_call("still-processing", *processing),
        todo_call("complete", *completed),
        todo_call("repeat", *completed),
        todo_call("reopen", *processing),
        AssistantMessage(content="Finished"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("completed-list"), extensions=[extension])

    events = [event async for event in agent.stream("Finish one task")]

    assert len([event for event in events if event.type is AgentEventType.TOOL_COMPLETED]) == 4
    failures = [event for event in events if event.type is AgentEventType.TOOL_FAILED]
    assert len(failures) == 1
    assert "cannot be reopened" in failures[0].message.content
    assert extension.todos("completed-list") is None


def test_todo_extension_rejects_empty_session_ids():
    extension = TodoWriteExtension()
    with pytest.raises(ValueError, match="session_id"):
        extension.todos(" ")
    with pytest.raises(ValueError, match="session_id"):
        extension.clear("")


async def test_model_facing_docs_publish_scope_statuses_and_single_completion():
    model = ScriptedModel(AssistantMessage(content="No task list needed"))
    extension = TodoWriteExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("tool-docs"),
        extensions=[extension, ToolGuidelinesExtension()],
    )

    await agent.run("Answer directly")

    definition = next(item for item in model.requests[0].tools if item.name == TODO_WRITE_TOOL_NAME)
    assert '"pending"' in definition.description
    assert '"in_progress"' in definition.description
    assert '"completed"' in definition.description
    assert "request-scoped" in definition.description
    assert "at most one task" in definition.description
    assert definition.parameters["$defs"]["TodoStatus"]["enum"] == ["pending", "in_progress", "completed"]

    guidance = "\n".join(
        message.content for message in model.requests[0].messages if isinstance(message, SystemMessage)
    )
    assert "request-scoped" in guidance
    assert "Complete at most one task per call" in guidance
    assert '"in_progress"' in guidance


async def test_batching_two_completions_fails_with_actionable_recovery_hint():
    model = ScriptedModel(
        todo_call(
            "start",
            ("First", TodoStatus.IN_PROGRESS),
            ("Second", TodoStatus.PENDING),
            ("Third", TodoStatus.PENDING),
        ),
        todo_call(
            "batch",
            ("First", TodoStatus.COMPLETED),
            ("Second", TodoStatus.COMPLETED),
            ("Third", TodoStatus.IN_PROGRESS),
        ),
        todo_call(
            "single",
            ("First", TodoStatus.COMPLETED),
            ("Second", TodoStatus.IN_PROGRESS),
            ("Third", TodoStatus.PENDING),
        ),
        AssistantMessage(content="Recovered"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentRunConfig("batch"), extensions=[extension])

    state_at_failure = None
    async for event in agent.stream("Work through tasks"):
        if event.type is AgentEventType.TOOL_FAILED:
            state_at_failure = extension.todos("batch")
            assert "at most one task per call" in event.message.content
            assert "completed 2 tasks" in event.message.content

    assert state_at_failure is not None and state_at_failure.in_progress is not None
    assert state_at_failure.in_progress.content == "First"
    assert tool_result(model.requests[3])["in_progress"]["content"] == "Second"
