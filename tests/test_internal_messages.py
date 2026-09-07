"""End-to-end coverage for internal message delivery and processing limits."""

import asyncio

import pytest

from zett_agent import (
    INTERNAL_MESSAGE_EVENT_NAME,
    Agent,
    AgentConfig,
    AgentEventType,
    AgentExtension,
    AgentIterationLimitError,
    AgentMessage,
    AssistantMessage,
    ExternalEvent,
    InternalMessageEvent,
    InternalMessageExtension,
    ModelEvent,
    ModelResponse,
    SQLiteSessionExtension,
    ToolCall,
)


@pytest.mark.parametrize("name", ["", " ", None, 42])
def test_invalid_extension_names_are_rejected(name):
    extension = AgentExtension()
    extension.name = name
    with pytest.raises(ValueError, match="non-empty string"):
        Agent(PausingModel(), extensions=[extension])


def test_extension_names_are_unique_including_builtins():
    first, second = AgentExtension(), AgentExtension()
    with pytest.raises(ValueError, match="Duplicate extension name"):
        Agent(PausingModel(), extensions=[first, second])
    second.name = "second"
    Agent(PausingModel(), extensions=[first, second])
    first.name = "InternalMessageExtension"
    with pytest.raises(ValueError, match="Duplicate extension name"):
        Agent(PausingModel(), extensions=[first])


async def test_internal_message_tool_loop_still_obeys_its_own_budget():
    class Publisher(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "initial answer":
                await context.publish(InternalMessageEvent(AgentMessage(content="continue")))

    class ToolLoopModel:
        calls = 0

        async def stream(self, request):
            self.calls += 1
            message = (
                AssistantMessage(content="initial answer")
                if self.calls == 1
                else AssistantMessage(tool_calls=(ToolCall("call", "missing"),))
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = ToolLoopModel()
    agent = await Agent.create(model, config=AgentConfig("budget"), extensions=[Publisher()], max_iterations=2)
    with pytest.raises(AgentIterationLimitError):
        await agent.run("initial")
    assert model.calls == 3


class PausingModel:
    """Pause the first response so a caller can enqueue concurrent input."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.requests = []
        self.first_request_started = asyncio.Event()
        self.release_first_request = asyncio.Event()

    async def stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_request_started.set()
            await self.release_first_request.wait()
        answer = self.answers.pop(0)
        yield ModelEvent.text(answer)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))


@pytest.mark.parametrize("limit", [-1, 1.5, True, "8", None])
def test_internal_message_limit_rejects_invalid_values(limit):
    with pytest.raises(ValueError, match="non-negative integer"):
        Agent(PausingModel(), max_internal_messages=limit)


@pytest.mark.parametrize("limit,queued", [(0, 0), (0, 1), (2, 2), (2, 3), (8, 8), (8, 9)])
@pytest.mark.parametrize("use_stream", [True, False])
async def test_request_internal_message_limit_and_reuse(limit, queued, use_stream):
    class Publisher(AgentExtension):
        async def before_run(self, context):
            for index in range(queued):
                await context.publish(InternalMessageEvent(AgentMessage(content=f"internal {index}")))

    model = PausingModel(*(["answer"] * 30))
    model.release_first_request.set()
    options = {} if limit == 8 else {"max_internal_messages": limit}
    agent = await Agent.create(
        model, config=AgentConfig("limits"), extensions=[Publisher()], max_iterations=1, **options
    )
    assert agent.max_internal_messages == limit
    for _ in range(2):
        previous_calls = len(model.requests)
        operation = _collect(agent, "initial") if use_stream else agent.run("initial")
        if queued > limit:
            with pytest.raises(AgentIterationLimitError, match="internal messages"):
                await operation
        else:
            await operation
        assert len(model.requests) - previous_calls == 1 + min(limit, queued)


async def _collect(agent: Agent, message: str):
    return [event async for event in agent.stream(message)]


async def test_external_internal_message_continues_loop_and_persists_complete_history(tmp_path) -> None:
    model = PausingModel("initial answer", "internal answer")
    persistence = SQLiteSessionExtension(tmp_path / "sessions.db")
    agent = await Agent.create(
        model,
        config=AgentConfig("session-1", request_id="request-1"),
        extensions=[persistence],
    )
    task = asyncio.create_task(_collect(agent, "initial question"))
    await model.first_request_started.wait()

    accepted = await asyncio.to_thread(
        agent.emit_external_event,
        ExternalEvent(
            INTERNAL_MESSAGE_EVENT_NAME,
            {
                "session_id": "session-1",
                "request_id": "request-1",
                "content": "internal question",
            },
        ),
    )
    assert accepted
    model.release_first_request.set()
    events = await task

    assert [event.type for event in events] == [
        AgentEventType.MODEL_STARTED,
        AgentEventType.TEXT_DELTA,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.INTERNAL_MESSAGE_STARTED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.TEXT_DELTA,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.INTERNAL_MESSAGE_COMPLETED,
        AgentEventType.RUN_COMPLETED,
    ]
    started = next(event for event in events if event.type is AgentEventType.INTERNAL_MESSAGE_STARTED)
    completed = next(event for event in events if event.type is AgentEventType.INTERNAL_MESSAGE_COMPLETED)
    assert [started.internal_message.content] == ["internal question"]
    assert completed.internal_message == started.internal_message
    assert completed.message == AssistantMessage(content="internal answer")
    assert [message.content for message in model.requests[1].messages[-3:]] == [
        "initial question",
        "initial answer",
        "internal question",
    ]

    records = persistence.list_raw_messages("session-1")
    assert [record.message.role.value for record in records] == ["user", "assistant", "agent", "assistant"]
    assert [record.message.content for record in records] == [
        "initial question",
        "initial answer",
        "internal question",
        "internal answer",
    ]
    persistence.close()


async def test_internal_internal_message_event_queues_messages_in_publish_order() -> None:
    class Publisher(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "initial":
                await context.publish(InternalMessageEvent(AgentMessage(content="first")))
                await context.publish(InternalMessageEvent(AgentMessage(content="second")))

    model = PausingModel("initial", "first answer", "second answer")
    model.release_first_request.set()
    agent = await Agent.create(model, config=AgentConfig("internal"), extensions=[Publisher()])

    events = await _collect(agent, "question")

    started = next(event for event in events if event.type is AgentEventType.INTERNAL_MESSAGE_STARTED)
    assert [started.internal_message.content] == ["first"]
    assert model.requests[1].messages[-1].content == "first"
    assert model.requests[2].messages[-1].content == "second"
    assert len(model.requests) == 3
    assert events[-2].type is AgentEventType.INTERNAL_MESSAGE_COMPLETED
    assert events[-1].type is AgentEventType.RUN_COMPLETED


async def test_internal_message_arriving_during_internal_message_processing_creates_a_second_batch() -> None:
    class ThreeStepModel:
        def __init__(self) -> None:
            self.requests = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.second_started = asyncio.Event()
            self.release_second = asyncio.Event()

        async def stream(self, request):
            self.requests.append(request)
            number = len(self.requests)
            if number == 1:
                self.first_started.set()
                await self.release_first.wait()
            elif number == 2:
                self.second_started.set()
                await self.release_second.wait()
            answer = f"answer-{number}"
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))

    model = ThreeStepModel()
    agent = await Agent.create(model, config=AgentConfig("batches"))
    events = []

    async def consume() -> None:
        async for event in agent.stream("initial"):
            events.append(event)

    task = asyncio.create_task(consume())
    await model.first_started.wait()
    assert agent.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "batches", "content": "first internal"})
    )
    model.release_first.set()
    await model.second_started.wait()
    assert agent.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "batches", "content": "second internal"})
    )
    model.release_second.set()
    await task

    assert sum(event.type is AgentEventType.INTERNAL_MESSAGE_STARTED for event in events) == 2
    assert sum(event.type is AgentEventType.INTERNAL_MESSAGE_COMPLETED for event in events) == 2
    assert len(model.requests) == 3


@pytest.mark.parametrize("renamed", [False, True])
def test_internal_extension_cannot_be_supplied_by_caller(renamed):
    extension = InternalMessageExtension()
    if renamed:
        extension.name = "custom-internal"
    with pytest.raises(ValueError, match="Agent built-in"):
        Agent(PausingModel(), extensions=[extension])


async def test_builtin_internal_inboxes_are_isolated_between_agents() -> None:
    first_model = PausingModel("one", "internal answer")
    second_model = PausingModel("two")
    first = await Agent.create(first_model, config=AgentConfig("same"), extensions=[])
    second = await Agent.create(second_model, config=AgentConfig("same"), extensions=[])

    assert not first.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "same", "content": "too early"})
    )
    first_task = asyncio.create_task(_collect(first, "first"))
    second_task = asyncio.create_task(_collect(second, "second"))
    await asyncio.gather(first_model.first_request_started.wait(), second_model.first_request_started.wait())
    assert first.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "same", "content": "first only"})
    )
    assert not first.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"content": "bad"}),
        config=AgentConfig("same", request_id="wrong-request"),
    )
    first_model.release_first_request.set()
    second_model.release_first_request.set()
    await asyncio.gather(first_task, second_task)
    assert len(first_model.requests) == 2
    assert len(second_model.requests) == 1
    assert first_model.requests[-1].messages[-1] == AgentMessage(content="first only")
    assert not first.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "same", "content": "too late"})
    )


async def test_each_internal_message_has_an_independent_iteration_budget() -> None:
    class Publisher(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "only":
                await context.publish(InternalMessageEvent(AgentMessage(content="continue")))

    model = PausingModel("only", "next run")
    model.release_first_request.set()
    agent = await Agent.create(
        model,
        config=AgentConfig("limited"),
        extensions=[Publisher()],
        max_iterations=1,
    )

    assert (await agent.run("initial")).content == "next run"
    assert len(model.requests) == 2
    assert not agent.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "limited", "content": "stale"})
    )


async def test_cancellation_discards_queued_internal_message_before_agent_reuse() -> None:
    class CancelThenAnswerModel:
        def __init__(self) -> None:
            self.requests = []
            self.first_started = asyncio.Event()

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_started.set()
                await asyncio.Event().wait()
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="fresh answer")))

    model = CancelThenAnswerModel()
    agent = await Agent.create(model, config=AgentConfig("cancelled"), extensions=[])
    task = asyncio.create_task(agent.run("cancel this"))
    await model.first_started.wait()
    assert agent.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "cancelled", "content": "stale internal"})
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not agent.emit_external_event(
        ExternalEvent(INTERNAL_MESSAGE_EVENT_NAME, {"session_id": "cancelled", "content": "too late"})
    )

    result = await agent.run("new request")
    assert result.content == "fresh answer"
    assert [message.content for message in model.requests[1].messages if message.role == "user"] == ["new request"]
