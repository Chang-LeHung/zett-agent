"""Ask-user event protocol, suspension, routing, and cancellation behavior."""

import asyncio
import json
from datetime import UTC, datetime
from time import monotonic_ns

import pytest

from zett_agent import (
    ASK_USER_RESPONSE_EVENT_NAME,
    Agent,
    AgentEventType,
    AgentExtension,
    AgentPhase,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AskUserEvent,
    AskUserExtension,
    AssistantMessage,
    ExtensionEvent,
    ExternalEvent,
    ExternalEventExtension,
    ModelEvent,
    ModelResponse,
    RunCancelledEvent,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
)


class AskModel:
    def __init__(self, call_id: str = "question-1") -> None:
        self.call_id = call_id
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        message = (
            AssistantMessage(
                tool_calls=(
                    ToolCall(
                        self.call_id,
                        "ask_user",
                        {
                            "question": "Which format?",
                            "options": ["Markdown", "Plain text"],
                            "allow_multiple": False,
                        },
                    ),
                )
            )
            if len(self.requests) == 1
            else AssistantMessage(content="Markdown selected")
        )
        yield ModelEvent.completed(ModelResponse(message))


class ExternalEventRecorder(AgentExtension):
    def __init__(self, *, accepts: bool) -> None:
        self.accepts = accepts
        self.events: list[ExternalEvent] = []

    def accept(self, config: AgentRunConfig | None, event: ExternalEvent) -> bool:
        self.events.append(event)
        return self.accepts


class ContextRecorder(AgentExtension):
    def __init__(self) -> None:
        self.context = None

    async def on_state(self, context) -> None:
        self.context = context


def _context(session_id: str = "base-extension") -> AgentRunContext:
    return AgentRunContext(AgentRunConfig(session_id), AgentState(), {})


async def _wait_for_ask(events: list, ready: asyncio.Event, agent: Agent) -> None:
    async for event in agent.stream("Prepare the document"):
        events.append(event)
        if isinstance(event, AskUserEvent):
            ready.set()


def test_agent_broadcasts_external_events_to_every_extension() -> None:
    first = ExternalEventRecorder(accepts=True)
    second = ExternalEventRecorder(accepts=False)
    first.name = "first-recorder"
    second.name = "second-recorder"
    agent = Agent(AskModel(), extensions=[first, second])
    event = ExternalEvent("ui_action", {"value": 1})

    assert agent.emit_external_event(event)
    assert first.events == [event]
    assert second.events == [event]


def test_agent_reports_an_unaccepted_external_event() -> None:
    extension = ExternalEventRecorder(accepts=False)
    agent = Agent(AskModel(), extensions=[extension])

    assert not agent.emit_external_event(ExternalEvent("unknown", {}))


@pytest.mark.parametrize(
    ("response_event_name", "correlation_field", "message"),
    [
        ("", "operation_id", "response_event_name cannot be empty"),
        ("response", "", "correlation_field cannot be empty"),
    ],
)
def test_external_event_extension_rejects_empty_protocol_fields(
    response_event_name: str,
    correlation_field: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ExternalEventExtension(
            response_event_name=response_event_name,
            correlation_field=correlation_field,
        )


async def test_external_event_extension_rejects_an_empty_correlation_id() -> None:
    extension = ExternalEventExtension(response_event_name="response", correlation_field="operation_id")
    with pytest.raises(ValueError, match="correlation_id cannot be empty"):
        async with extension._wait_for_external_event(_context(), ""):
            pass


async def test_external_event_extension_rejects_duplicate_pending_routes() -> None:
    extension = ExternalEventExtension(response_event_name="response", correlation_field="operation_id")
    context = _context()

    async def wait() -> None:
        async with extension._wait_for_external_event(context, "operation-1"):
            pass

    first = asyncio.create_task(wait())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already pending"):
        await wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


def test_external_event_extension_rejects_consuming_without_a_response() -> None:
    extension = ExternalEventExtension(response_event_name="response", correlation_field="operation_id")
    with pytest.raises(RuntimeError, match="without an accepted external response"):
        extension._take_external_event(_context())


async def test_ask_user_ignores_other_tool_calls() -> None:
    extension = AskUserExtension()
    context = _context()
    await extension.before_tool(context, ToolCall("call-1", "another_tool", {}))
    assert context.event_queue.empty


async def test_ask_user_event_pauses_tool_until_accept_and_returns_payload():
    extension = AskUserExtension()
    model = AskModel()
    config = AgentRunConfig("session-1", request_id="request-1")
    agent = await Agent.create(model, config=config, extensions=[extension])
    events, ready = [], asyncio.Event()
    task = asyncio.create_task(_wait_for_ask(events, ready, agent))

    await asyncio.wait_for(ready.wait(), timeout=1)
    ask = next(event for event in events if isinstance(event, AskUserEvent))
    assert ask.type == AgentEventType.CUSTOM
    assert ask.name == "ask_user"
    assert ask.session_id == "session-1"
    assert ask.phase == AgentPhase.READY
    assert [call.id for call in ask.tool_calls] == ["question-1"]
    assert ask.payload == {
        "session_id": "session-1",
        "tool_call_id": "question-1",
        "question": "Which format?",
        "options": ["Markdown", "Plain text"],
        "allow_multiple": False,
        "response_event": ASK_USER_RESPONSE_EVENT_NAME,
    }
    assert not task.done()
    assert all(event.type != AgentEventType.TOOL_STARTED for event in events)

    response = ExternalEvent(
        ASK_USER_RESPONSE_EVENT_NAME,
        {"tool_call_id": "question-1", "answer": "Markdown"},
    )
    assert agent.emit_external_event(response, config=AgentRunConfig("other-session")) == []
    assert agent.emit_external_event(response, config=AgentRunConfig("session-1", request_id="stale")) == []
    assert not task.done()
    assert agent.emit_external_event(response, config=config) == ["AskUserExtension"]
    assert agent.emit_external_event(response, config=config) == []
    await asyncio.wait_for(task, timeout=1)

    assert agent.state.phase == AgentPhase.COMPLETED
    assert [event.type for event in events if event.type in (AgentEventType.CUSTOM, AgentEventType.TOOL_STARTED)] == [
        AgentEventType.CUSTOM,
        AgentEventType.TOOL_STARTED,
    ]
    result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert json.loads(result.content) == {"name": ASK_USER_RESPONSE_EVENT_NAME, "payload": response.payload}


@pytest.mark.parametrize(
    "event",
    [
        ExternalEvent("unrelated", {}),
        ExternalEvent(ASK_USER_RESPONSE_EVENT_NAME, {}),
        ExternalEvent(ASK_USER_RESPONSE_EVENT_NAME, {"session_id": "session-1", "tool_call_id": 7}),
        ExternalEvent(ASK_USER_RESPONSE_EVENT_NAME, {"session_id": "other", "tool_call_id": "question-1"}),
    ],
)
def test_accept_rejects_unrelated_or_malformed_events(event):
    assert not AskUserExtension().accept(AgentRunConfig("missing"), event)


def test_external_event_rejects_an_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        ExternalEvent("  ", {})


@pytest.mark.parametrize("payload", [None, {1: "value"}])
def test_external_event_requires_a_string_keyed_dictionary(payload):
    with pytest.raises(TypeError, match="dictionary with string keys"):
        ExternalEvent("event", payload)


async def test_accept_can_resume_from_another_thread():
    extension = AskUserExtension()
    agent = await Agent.create(AskModel(), config=AgentRunConfig("thread-session"), extensions=[extension])
    events, ready = [], asyncio.Event()
    task = asyncio.create_task(_wait_for_ask(events, ready, agent))
    await asyncio.wait_for(ready.wait(), timeout=1)
    accepted = await asyncio.to_thread(
        agent.emit_external_event,
        ExternalEvent(
            ASK_USER_RESPONSE_EVENT_NAME,
            {"session_id": "thread-session", "tool_call_id": "question-1", "answer": "Markdown"},
        ),
    )
    assert accepted
    await asyncio.wait_for(task, timeout=1)


async def test_same_tool_call_id_is_routed_by_session():
    extension = AskUserExtension()
    first = await Agent.create(AskModel(), config=AgentRunConfig("first"), extensions=[extension])
    second = await Agent.create(AskModel(), config=AgentRunConfig("second"), extensions=[extension])
    first_events, second_events = [], []
    first_ready, second_ready = asyncio.Event(), asyncio.Event()
    first_task = asyncio.create_task(_wait_for_ask(first_events, first_ready, first))
    second_task = asyncio.create_task(_wait_for_ask(second_events, second_ready, second))
    await asyncio.wait_for(asyncio.gather(first_ready.wait(), second_ready.wait()), timeout=1)

    assert second.emit_external_event(
        ExternalEvent(
            ASK_USER_RESPONSE_EVENT_NAME,
            {"session_id": "second", "tool_call_id": "question-1", "answer": "second"},
        )
    )
    await asyncio.wait_for(second_task, timeout=1)
    assert not first_task.done()
    assert first.emit_external_event(
        ExternalEvent(
            ASK_USER_RESPONSE_EVENT_NAME,
            {"session_id": "first", "tool_call_id": "question-1", "answer": "first"},
        )
    )
    await asyncio.wait_for(first_task, timeout=1)


async def test_multiple_questions_complete_end_to_end_in_model_order():
    class TwoQuestionModel:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(
                    tool_calls=(
                        ToolCall("first-question", "ask_user", {"question": "First?"}),
                        ToolCall("second-question", "ask_user", {"question": "Second?"}),
                    )
                )
                if len(self.requests) == 1
                else AssistantMessage(content="Both answers received")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = TwoQuestionModel()
    agent = await Agent.create(model, config=AgentRunConfig("two-questions"), extensions=[AskUserExtension()])
    events = []
    async for event in agent.stream("Ask both questions"):
        events.append(event)
        if isinstance(event, AskUserEvent):
            assert agent.emit_external_event(
                ExternalEvent(
                    ASK_USER_RESPONSE_EVENT_NAME,
                    {
                        "session_id": "two-questions",
                        "tool_call_id": event.tool_calls[0].id,
                        "answer": event.payload["question"],
                    },
                )
            )

    assert agent.state.phase == AgentPhase.COMPLETED
    assert [event.tool_calls[0].id for event in events if isinstance(event, AskUserEvent)] == [
        "first-question",
        "second-question",
    ]
    assert [
        event.type for event in events if event.type in (AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED)
    ] == [
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.TOOL_COMPLETED,
    ]
    tool_messages = [message for message in model.requests[1].messages if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["first-question", "second-question"]


async def test_ask_user_tool_schema_and_guidance_are_visible_to_model():
    class AnswerModel:
        def __init__(self):
            self.request = None

        async def stream(self, request):
            self.request = request
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    model = AnswerModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("schema"),
        extensions=[ToolGuidelinesExtension(), AskUserExtension()],
    )
    await agent.run("Hello")
    definition = next(tool for tool in model.request.tools if tool.name == "ask_user")
    assert definition.parameters["required"] == ["question"]
    options_schema = definition.parameters["properties"]["options"]["anyOf"][0]
    assert options_schema["maxItems"] == 20
    assert options_schema["items"]["minLength"] == 1
    guidance = "\n".join(message.content for message in model.request.messages if isinstance(message, SystemMessage))
    assert "## ask_user" in guidance
    assert "Use only when user input is required" in guidance
    assert "Ask exactly one question per call" in guidance
    assert "one ask_user call per question" in guidance


async def test_cancelling_wait_removes_pending_response():
    extension = AskUserExtension()
    agent = await Agent.create(AskModel(), config=AgentRunConfig("cancel-session"), extensions=[extension])
    events, ready = [], asyncio.Event()
    task = asyncio.create_task(_wait_for_ask(events, ready, agent))
    await asyncio.wait_for(ready.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.phase == AgentPhase.CANCELLED
    assert not agent.emit_external_event(
        ExternalEvent(
            ASK_USER_RESPONSE_EVENT_NAME,
            {"session_id": "cancel-session", "tool_call_id": "question-1", "answer": "late"},
        )
    )


async def test_cancel_event_wakes_the_waiting_agent_and_rejects_a_late_answer():
    extension = AskUserExtension()
    recorder = ContextRecorder()
    agent = await Agent.create(
        AskModel(),
        config=AgentRunConfig("cancel-event"),
        extensions=[recorder, extension],
    )
    events, ready = [], asyncio.Event()
    task = asyncio.create_task(_wait_for_ask(events, ready, agent))
    await asyncio.wait_for(ready.wait(), timeout=1)

    now = datetime.now(UTC)
    await extension.on_event(
        recorder.context,
        RunCancelledEvent(
            previous_phase=AgentPhase.READY,
            occurred_at=now,
            monotonic_ns=monotonic_ns(),
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.phase == AgentPhase.CANCELLED
    assert extension._pending == {}
    assert not agent.emit_external_event(
        ExternalEvent(
            ASK_USER_RESPONSE_EVENT_NAME,
            {"session_id": "cancel-event", "tool_call_id": "question-1", "answer": "late"},
        )
    )


async def test_error_hook_wakes_the_waiting_agent_with_the_original_error():
    extension = AskUserExtension()
    recorder = ContextRecorder()
    agent = await Agent.create(
        AskModel(),
        config=AgentRunConfig("error-event"),
        extensions=[recorder, extension],
    )
    events, ready = [], asyncio.Event()
    task = asyncio.create_task(_wait_for_ask(events, ready, agent))
    await asyncio.wait_for(ready.wait(), timeout=1)
    error = RuntimeError("external failure")

    await extension.on_error(recorder.context, error)

    with pytest.raises(RuntimeError, match="external failure") as captured:
        await task
    assert captured.value is error
    assert agent.state.phase == AgentPhase.FAILED
    assert extension._pending == {}


async def test_unrelated_internal_event_does_not_wake_the_waiting_agent():
    extension = AskUserExtension()
    recorder = ContextRecorder()
    agent = await Agent.create(
        AskModel(),
        config=AgentRunConfig("unrelated-event"),
        extensions=[recorder, extension],
    )
    events, ready = [], asyncio.Event()
    task = asyncio.create_task(_wait_for_ask(events, ready, agent))
    await asyncio.wait_for(ready.wait(), timeout=1)

    await extension.on_event(recorder.context, ExtensionEvent())
    await asyncio.sleep(0)

    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_closing_after_tool_started_discards_the_delivered_response():
    extension = AskUserExtension()
    agent = await Agent.create(AskModel(), config=AgentRunConfig("close-after-answer"), extensions=[extension])
    stream = agent.stream("Ask")

    while not isinstance(await anext(stream), AskUserEvent):
        pass
    assert agent.emit_external_event(
        ExternalEvent(
            ASK_USER_RESPONSE_EVENT_NAME,
            {"session_id": "close-after-answer", "tool_call_id": "question-1", "answer": "Markdown"},
        )
    )
    while (await anext(stream)).type != AgentEventType.TOOL_STARTED:
        pass

    await stream.aclose()

    assert agent.state.phase == AgentPhase.CANCELLED
    assert extension._accepted == {}


async def test_invalid_ask_arguments_do_not_pause_the_agent():
    class InvalidModel:
        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(tool_calls=(ToolCall("bad", "ask_user", {"question": ""}),))
                if len(self.requests) == 1
                else AssistantMessage(content="Recovered")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = InvalidModel()
    agent = await Agent.create(model, config=AgentRunConfig("invalid"), extensions=[AskUserExtension()])
    events = [event async for event in agent.stream("Ask")]
    assert all(not isinstance(event, AskUserEvent) for event in events)
    failed = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert not failed.success
    assert "validation error" in failed.content
