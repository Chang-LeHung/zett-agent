"""Fault injection and scheduling boundaries for the decomposed stream lifecycle."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentMessage,
    AgentPhase,
    AgentProtocolError,
    AgentState,
    AssistantMessage,
    ExternalEvent,
    InternalMessageEvent,
    InternalMessageExtension,
    ModelEvent,
    ModelResponse,
    SteeringExtension,
    SteeringMessageEvent,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool,
)


class ToolModel:
    async def stream(self, request):
        if isinstance(request.messages[-1], ToolMessage):
            result = AssistantMessage(content="done")
        else:
            result = AssistantMessage(tool_calls=(ToolCall("one", "echo", {}),))
        yield ModelEvent.completed(ModelResponse(result))


@tool(guidelines="Return a deterministic result for lifecycle tests.")
def echo() -> str:
    """Return a deterministic result."""
    return "ok"


HOOKS = [
    "on_tool",
    "on_message",
    "before_run",
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "after_run",
    "on_success",
]


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("cancel", [False, True])
async def test_every_lifecycle_hook_failure_releases_session_and_allows_retry(hook, cancel):
    failure = asyncio.CancelledError("injected") if cancel else RuntimeError("injected")
    errors = []

    class Fault(AgentExtension):
        enabled = True

        async def on_error(self, context, error):
            errors.append(error)

    async def inject(self, context, *args):
        if self.enabled:
            self.enabled = False
            raise failure

    setattr(Fault, hook, inject)
    agent = await Agent.create(ToolModel(), config=AgentConfig("fault"), tools=[echo], extensions=[Fault()])
    with pytest.raises(type(failure)) as caught:
        await asyncio.wait_for(agent.run("first"), 2)
    assert caught.value is failure
    assert agent.get_state("fault").phase is (AgentPhase.CANCELLED if cancel else AgentPhase.FAILED)
    assert errors == ([] if cancel else [failure])
    assert agent._active_configs == {}
    assert agent._internal_message_extension._inboxes == {}
    assert agent._steering_extension._inboxes == {}
    assert (await asyncio.wait_for(agent.run("retry"), 2)).content == "done"


@pytest.mark.parametrize("hook", ["before_model_events", "before_tool_events"])
@pytest.mark.parametrize("cancel", [True, False])
async def test_event_generator_failure_runs_finally_and_preserves_error(hook, cancel):
    finalized = []
    failure = asyncio.CancelledError("stream cancelled") if cancel else RuntimeError("stream failed")

    class Fault(AgentExtension):
        enabled = True

    async def events(self, context, *args):
        if self.enabled:
            self.enabled = False
            try:
                yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name="progress")
                raise failure
            finally:
                finalized.append(True)

    setattr(Fault, hook, events)
    agent = await Agent.create(ToolModel(), config=AgentConfig("events"), tools=[echo], extensions=[Fault()])
    with pytest.raises(type(failure)) as caught:
        await agent.run("fail")
    assert caught.value is failure
    assert finalized == [True]
    assert agent._active_configs == {}
    assert (await agent.run("retry")).content == "done"


@pytest.mark.parametrize(
    "boundary",
    [
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.RUN_COMPLETED,
    ],
)
async def test_closing_at_each_public_boundary_is_idempotent(boundary):
    agent = await Agent.create(ToolModel(), config=AgentConfig("close"), tools=[echo], extensions=[])
    stream = agent.stream("first")
    async for event in stream:
        if event.type is boundary:
            break
    else:
        pytest.fail(f"Missing boundary: {boundary}")
    await stream.aclose()
    await stream.aclose()
    expected = AgentPhase.COMPLETED if boundary is AgentEventType.RUN_COMPLETED else AgentPhase.CANCELLED
    assert agent.get_state("close").phase is expected
    assert agent._active_configs == {}
    assert (await agent.run("again")).content == "done"


@pytest.mark.parametrize(
    "extension_type,name,message,event_type",
    [
        (SteeringExtension, "steering_message", UserMessage(content="queued"), SteeringMessageEvent),
        (InternalMessageExtension, "internal_message", AgentMessage(content="queued"), InternalMessageEvent),
    ],
)
async def test_inbox_open_close_and_late_publication(extension_type, name, message, event_type):
    extension = extension_type()
    config = AgentConfig("inbox")
    context = AgentContext(config, AgentState(), {})
    external = ExternalEvent(name, {"content": "queued"})
    assert not extension.accept(None, external)
    assert not extension.accept(config, external)
    extension.open(context)
    with pytest.raises(AgentProtocolError, match="already active"):
        extension.open(context)
    assert not extension.accept(config, ExternalEvent(name, {"content": "   "}))
    await extension.on_event(context, event_type(message))
    extension.close(context)
    extension.close(context)
    assert not extension.accept(config, external)
    with pytest.raises(AgentProtocolError, match="closed"):
        await extension.on_event(context, event_type(message))


def test_steering_accept_and_final_selection_are_atomic_across_threads():
    # Exercise both lock acquisition orders without sleeps or network calls.
    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(100):
            extension = SteeringExtension()
            config = AgentConfig("race")
            context = AgentContext(config, AgentState(), {})
            extension.open(context)
            barrier = Barrier(2, timeout=2)

            def deliver(barrier=barrier, extension=extension, config=config):
                barrier.wait()
                return extension.accept(config, ExternalEvent("steering_message", {"content": "urgent"}))

            def select(barrier=barrier, extension=extension, context=context):
                barrier.wait()
                return extension.select_next(context, lambda: False)

            producer, consumer = executor.submit(deliver), executor.submit(select)
            accepted, selected = producer.result(timeout=3), consumer.result(timeout=3)
            assert accepted == (selected == UserMessage(content="urgent"))
            extension.close(context)


async def test_many_sessions_overlap_and_cleanup_without_cross_session_messages():
    count = 24
    ready, release = asyncio.Event(), asyncio.Event()
    entered = 0

    class ParallelModel:
        async def stream(self, request):
            nonlocal entered
            entered += 1
            if entered == count:
                ready.set()
            await release.wait()
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content=request.messages[-1].content)))

    agent = await Agent.create(ParallelModel(), config=AgentConfig("default"), system_prompt="")
    tasks = [asyncio.create_task(agent.run(str(i), config=AgentConfig(str(i)))) for i in range(count)]
    try:
        await asyncio.wait_for(ready.wait(), 3)
        assert len(agent._active_configs) == count
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 3)
        for i, result in enumerate(results):
            assert result.content == str(i)
            assert [m.content for m in agent.get_state(str(i)).messages] == [str(i)] * 2
        assert agent._active_configs == {}
        assert agent._steering_extension._inboxes == {}
        assert agent._internal_message_extension._inboxes == {}
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
