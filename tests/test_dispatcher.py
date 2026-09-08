"""Streaming callback dispatch, ordering, and failure propagation."""

import asyncio
from contextlib import aclosing
from uuid import UUID

import pytest

from zett_agent import (
    Agent,
    AgentClient,
    AgentConfig,
    AgentEvent,
    AgentEventDispatcher,
    AgentEventType,
    AgentPhase,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    ReasoningEffort,
    create_agent,
)


@pytest.mark.parametrize("event_type", list(AgentEventType))
async def test_every_event_has_an_optional_overridable_callback(event_type):
    event = AgentEvent(event_type, session_id="test", name="test.custom")
    await AgentEventDispatcher().dispatch(event)
    received = []

    async def callback(self, value):
        received.append(value)

    handler_type = type("Handler", (AgentEventDispatcher,), {f"on_{event_type.value}_event": callback})
    await handler_type().dispatch(event)
    assert len(received) == 1
    assert received[0] is event


async def test_custom_name_is_data_not_a_callback_name():
    received = []

    class Handler(AgentEventDispatcher):
        async def on_custom_event(self, event):
            received.append((event.name, event.payload))

    await Handler().dispatch(AgentEvent(AgentEventType.CUSTOM, "test", name="dispatch", payload={"answer": [1, 2]}))
    assert received == [("dispatch", {"answer": [1, 2]})]


async def test_callback_is_not_wrapped_and_runs_in_dispatch_task():
    tasks = []

    async def callback(self, event):
        tasks.append(asyncio.current_task())

    handler_type = type("Handler", (AgentEventDispatcher,), {"on_text_delta_event": callback})
    assert handler_type.on_text_delta_event is callback
    await handler_type().dispatch(AgentEvent(AgentEventType.TEXT_DELTA, session_id="test", delta="hello"))
    assert tasks == [asyncio.current_task()]


@pytest.mark.parametrize("exception", [RuntimeError("handler failed"), asyncio.CancelledError()])
async def test_callback_failures_propagate(exception):
    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            raise exception

    with pytest.raises(type(exception)) as captured:
        await Handler().dispatch(AgentEvent(AgentEventType.TEXT_DELTA, "test"))
    assert captured.value is exception


async def test_dispatch_with_real_agent_stream_preserves_order():
    class Model:
        async def stream(self, request):
            yield ModelEvent.text("Hello")
            yield ModelEvent.text(" world")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Hello world")))

    received = []

    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            await asyncio.sleep(0)
            received.append(event.delta)

        async def on_run_completed_event(self, event):
            received.append(event.message.content)

    agent = await Agent.create(Model(), config=AgentConfig(session_id="test"))
    handler = Handler()
    async with aclosing(agent.stream("Greet me")) as events:
        async for event in events:
            await handler.dispatch(event)
    assert received == ["Hello", " world", "Hello world"]


@pytest.mark.parametrize("collect", [True, False])
async def test_create_binds_handler_for_run_and_stream(collect):
    class Model:
        async def stream(self, request):
            yield ModelEvent.text("Hello")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Hello")))

    received = []

    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            received.append(event)

    agent = await create_agent(Model(), event_dispatcher=Handler())
    if collect:
        assert (await agent.run("Hello")).content == "Hello"
    else:
        async with aclosing(agent.stream("Hello")) as events:
            emitted = [event async for event in events]
        assert next(event for event in emitted if event.type == AgentEventType.TEXT_DELTA) is received[0]
    assert len(received) == 1
    assert UUID(received[0].session_id).version == 7
    await agent.run("Again")
    assert received[1].session_id == received[0].session_id
    other = await create_agent(Model(), event_dispatcher=Handler())
    await other.run("Different session")
    assert received[2].session_id != received[0].session_id
    await agent.run("Explicit session", config=AgentConfig(session_id="explicit"))
    assert received[3].session_id == "explicit"


@pytest.mark.parametrize("error", [RuntimeError("callback failed"), asyncio.CancelledError()])
async def test_bound_callback_failure_cleans_up_and_allows_next_run(error):
    class Model:
        async def stream(self, request):
            yield ModelEvent.text("Hello")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Hello")))

    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            raise error

    agent = await create_agent(Model(), event_dispatcher=Handler())
    with pytest.raises(type(error)) as captured:
        await agent.run("Hello")
    assert captured.value is error
    assert agent.agent.state.phase == AgentPhase.CANCELLED
    agent.event_dispatcher = None
    assert (await agent.run("Try again")).content == "Hello"


async def test_client_wraps_existing_agent_and_closes_interrupted_stream():
    class Model:
        async def stream(self, request):
            yield ModelEvent.text("Hello")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Hello")))

    runtime = await Agent.create(Model(), config=AgentConfig(session_id="existing"))
    client = AgentClient(runtime)
    assert client.agent is runtime
    async with aclosing(client.stream("Hello")) as events:
        async for event in events:
            if event.type == AgentEventType.TEXT_DELTA:
                break
    assert runtime.state.phase == AgentPhase.CANCELLED
    assert (await client.run("Again")).content == "Hello"


async def test_factory_preserves_explicit_configuration_and_model_errors():
    class Model:
        async def stream(self, request):
            raise ValueError("model failed")
            yield

    model = Model()
    client = await create_agent(
        model,
        config=AgentConfig(session_id="explicit"),
        system_prompt="Custom instructions",
        reasoning_effort=ReasoningEffort.HIGH,
        max_iterations=4,
        max_internal_messages=2,
        extensions=[],
    )
    assert client.agent.model is model
    assert client.agent.system_prompt == "Custom instructions"
    assert client.agent.reasoning_effort == ReasoningEffort.HIGH
    assert client.agent.max_iterations == 4
    assert client.agent.max_internal_messages == 2
    with pytest.raises(ValueError, match="model failed"):
        await client.run("Hello")
    assert client.agent.state.phase == AgentPhase.FAILED
