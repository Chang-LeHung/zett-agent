"""Sequential todo-write tool behavior across complete Agent runs."""

import asyncio
import json

import pytest

from zett_agent import (
    TODO_WRITE_TOOL_NAME,
    Agent,
    AgentConfig,
    AgentEventType,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    SystemMessage,
    TodoStatus,
    TodoWriteExtension,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
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
    first = ("Inspect code", TodoStatus.PROCESSING)
    second = ("Implement change", TodoStatus.PENDING)
    third = ("Run tests", TodoStatus.PENDING)
    model = ScriptedModel(
        todo_call("todo-1", first, second, third),
        todo_call(
            "todo-2",
            (first[0], TodoStatus.COMPLETED),
            (second[0], TodoStatus.PROCESSING),
            third,
        ),
        todo_call(
            "todo-3",
            (first[0], TodoStatus.COMPLETED),
            (second[0], TodoStatus.COMPLETED),
            (third[0], TodoStatus.PROCESSING),
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
        config=AgentConfig("todo-lifecycle"),
        extensions=[extension, ToolGuidelinesExtension()],
    )

    reply = await agent.run("Complete this task")

    assert reply.content == "All tasks completed"
    assert [tool_result(request)["processing_index"] for request in model.requests[1:]] == [0, 1, 2, None]
    assert [tool_result(request)["completed"] for request in model.requests[1:]] == [False, False, False, True]
    assert tool_result(model.requests[1])["processing"] == {
        "content": "Inspect code",
        "status": "processing",
    }
    assert tool_result(model.requests[4])["processing"] is None
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
        (("", TodoStatus.PROCESSING),),
        (("   ", TodoStatus.PROCESSING),),
        (("First", TodoStatus.PENDING),),
        (("First", TodoStatus.COMPLETED),),
        (("First", TodoStatus.PROCESSING), ("Second", TodoStatus.PROCESSING)),
        (("First", TodoStatus.PENDING), ("Second", TodoStatus.PROCESSING)),
    ],
)
async def test_invalid_initial_todo_list_fails_without_storing_state(items):
    model = ScriptedModel(todo_call("invalid-initial", *items), AssistantMessage(content="Recovered"))
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("invalid-initial"), extensions=[extension])

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
            ("Third", TodoStatus.PROCESSING),
        ),
        (
            ("First", TodoStatus.PENDING),
            ("Second", TodoStatus.PROCESSING),
            ("Third", TodoStatus.PENDING),
        ),
        (
            ("Changed", TodoStatus.PROCESSING),
            ("Second", TodoStatus.PENDING),
            ("Third", TodoStatus.PENDING),
        ),
        (
            ("Second", TodoStatus.PROCESSING),
            ("First", TodoStatus.PENDING),
            ("Third", TodoStatus.PENDING),
        ),
    ],
)
async def test_invalid_update_fails_atomically_and_preserves_current_task(updated):
    initial = (
        ("First", TodoStatus.PROCESSING),
        ("Second", TodoStatus.PENDING),
        ("Third", TodoStatus.PENDING),
    )
    model = ScriptedModel(
        todo_call("initial", *initial),
        todo_call("invalid-update", *updated),
        todo_call(
            "valid-update",
            ("First", TodoStatus.COMPLETED),
            ("Second", TodoStatus.PROCESSING),
            ("Third", TodoStatus.PENDING),
        ),
        AssistantMessage(content="Recovered"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("invalid-update"), extensions=[extension])

    events = [event async for event in agent.stream("Work through tasks")]

    successful = [event for event in events if event.type is AgentEventType.TOOL_COMPLETED]
    assert len(successful) == 2
    assert len([event for event in events if event.type is AgentEventType.TOOL_FAILED]) == 1
    result = tool_result(model.requests[3])
    assert result["processing_index"] == 1
    assert result["processing"]["content"] == "Second"
    assert extension.todos("invalid-update") is None


async def test_successful_requests_clear_session_state_and_clear_handles_empty_state():
    extension = TodoWriteExtension()
    first_model = ScriptedModel(
        todo_call("first", ("First session", TodoStatus.PROCESSING)),
        AssistantMessage(content="done"),
    )
    second_model = ScriptedModel(
        todo_call("second", ("Second session", TodoStatus.PROCESSING)),
        AssistantMessage(content="done"),
    )
    first_agent = await Agent.create(first_model, config=AgentConfig("first"), extensions=[extension])
    second_agent = await Agent.create(second_model, config=AgentConfig("second"), extensions=[extension])

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
                yield ModelEvent.completed(ModelResponse(todo_call("start", ("Task", TodoStatus.PROCESSING))))
                return
            raise RuntimeError("model failed")
            yield

    extension = TodoWriteExtension()
    agent = await Agent.create(FailingModel(), config=AgentConfig("error-cleanup"), extensions=[extension])

    with pytest.raises(RuntimeError, match="model failed"):
        await agent.run("Start")

    assert extension.todos("error-cleanup") is None


async def test_stream_keeps_todos_until_successful_terminal_cleanup():
    model = ScriptedModel(
        todo_call("start", ("Task", TodoStatus.PROCESSING)),
        AssistantMessage(content="Finished"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("stream-cleanup"), extensions=[extension])
    state_at_tool_completion = None
    state_at_run_completion = object()

    async for event in agent.stream("Start"):
        if event.type is AgentEventType.TOOL_COMPLETED:
            state_at_tool_completion = extension.todos("stream-cleanup")
        elif event.type is AgentEventType.RUN_COMPLETED:
            state_at_run_completion = extension.todos("stream-cleanup")

    assert state_at_tool_completion is not None
    assert state_at_tool_completion.processing is not None
    assert state_at_tool_completion.processing.content == "Task"
    assert state_at_run_completion is None


async def test_successful_cleanup_allows_a_fresh_list_in_the_next_run():
    model = ScriptedModel(
        todo_call("first", ("Old task", TodoStatus.PROCESSING)),
        AssistantMessage(content="First run finished"),
        todo_call(
            "second",
            ("New task one", TodoStatus.PROCESSING),
            ("New task two", TodoStatus.PENDING),
        ),
        AssistantMessage(content="Second run finished"),
    )
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("success-reuse"), extensions=[extension])

    first_reply = await agent.run("First run")
    second_reply = await agent.run("Second run")

    assert first_reply.content == "First run finished"
    assert second_reply.content == "Second run finished"
    second_result = tool_result(model.requests[3])
    assert [item["content"] for item in second_result["todos"]] == ["New task one", "New task two"]
    assert second_result["processing_index"] == 0
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
                yield ModelEvent.completed(ModelResponse(todo_call("start", ("Task", TodoStatus.PROCESSING))))
                return
            self.blocked.set()
            await asyncio.Event().wait()
            yield

    model = BlockingModel()
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("cancel-cleanup"), extensions=[extension])
    task = asyncio.create_task(agent.run("Start"))
    await asyncio.wait_for(model.blocked.wait(), timeout=1)

    active = extension.todos("cancel-cleanup")
    assert active is not None and active.processing is not None
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
                    yield ModelEvent.completed(ModelResponse(todo_call("old", ("Old task", TodoStatus.PROCESSING))))
                case 2:
                    self.blocked.set()
                    await asyncio.Event().wait()
                case 3:
                    yield ModelEvent.completed(
                        ModelResponse(todo_call("new", ("Replacement task", TodoStatus.PROCESSING)))
                    )
                case 4:
                    yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Restarted")))

    model = CancelThenResumeModel()
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("cancel-reuse"), extensions=[extension])
    cancelled_run = asyncio.create_task(agent.run("Start old work"))
    await asyncio.wait_for(model.blocked.wait(), timeout=1)
    cancelled_run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_run

    reply = await agent.run("Start replacement work")

    assert reply.content == "Restarted"
    new_result = tool_result(model.requests[3])
    assert new_result["processing"]["content"] == "Replacement task"
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
                    yield ModelEvent.completed(ModelResponse(todo_call("old", ("Old task", TodoStatus.PROCESSING))))
                case 2:
                    raise RuntimeError("temporary model failure")
                case 3:
                    yield ModelEvent.completed(
                        ModelResponse(todo_call("new", ("Replacement task", TodoStatus.PROCESSING)))
                    )
                case 4:
                    yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Recovered")))

    model = FailThenResumeModel()
    extension = TodoWriteExtension()
    agent = await Agent.create(model, config=AgentConfig("error-reuse"), extensions=[extension])
    with pytest.raises(RuntimeError, match="temporary model failure"):
        await agent.run("Start old work")

    reply = await agent.run("Start replacement work")

    assert reply.content == "Recovered"
    new_result = tool_result(model.requests[3])
    assert new_result["processing"]["content"] == "Replacement task"
    assert extension.todos("error-reuse") is None


async def test_request_without_todo_calls_completes_with_empty_state():
    extension = TodoWriteExtension()
    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="No task list needed")),
        config=AgentConfig("no-todos"),
        extensions=[extension],
    )

    reply = await agent.run("Answer directly")

    assert reply.content == "No task list needed"
    assert extension.todos("no-todos") is None


async def test_completed_list_can_be_repeated_but_cannot_be_reopened():
    processing = (("Only task", TodoStatus.PROCESSING),)
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
    agent = await Agent.create(model, config=AgentConfig("completed-list"), extensions=[extension])

    events = [event async for event in agent.stream("Finish one task")]

    assert len([event for event in events if event.type is AgentEventType.TOOL_COMPLETED]) == 4
    failures = [event for event in events if event.type is AgentEventType.TOOL_FAILED]
    assert len(failures) == 1
    assert "cannot transition" in failures[0].message.content
    assert extension.todos("completed-list") is None


def test_todo_extension_rejects_empty_session_ids():
    extension = TodoWriteExtension()
    with pytest.raises(ValueError, match="session_id"):
        extension.todos(" ")
    with pytest.raises(ValueError, match="session_id"):
        extension.clear("")
