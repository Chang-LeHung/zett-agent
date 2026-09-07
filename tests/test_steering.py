"""Steering redirects at completed operation boundaries and preserves tool protocol."""

import asyncio

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentEventType,
    AgentExtension,
    AgentMessage,
    AgentPhase,
    AgentState,
    AssistantMessage,
    ExternalEvent,
    InternalMessageEvent,
    ModelEvent,
    ModelResponse,
    SQLiteSessionExtension,
    SteeringExtension,
    SteeringMessageEvent,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool,
)


class ScriptedModel:
    def __init__(self, *messages):
        self.messages = list(messages)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(self.messages.pop(0)))


@pytest.mark.parametrize("other_input", [True, False])
def test_select_next_returns_only_user_messages_and_closes_only_when_idle(other_input):
    extension = SteeringExtension()
    config = AgentConfig("selection")
    context = AgentContext(config, AgentState(), {})
    extension.open(context)
    reservations = []

    def reserve():
        reservations.append(True)
        return other_input

    try:
        event = ExternalEvent("steering_message", {"content": "urgent"})
        assert extension.accept(config, event)
        assert extension.select_next(context, reserve) == UserMessage(content="urgent")
        assert reservations == []
        assert extension.select_next(context, reserve) is None
        assert reservations == [True]
        assert extension.accept(config, event) is other_input
    finally:
        extension.close(context)


@pytest.mark.parametrize("renamed", [False, True])
def test_steering_is_owned_by_agent(renamed):
    extension = SteeringExtension()
    if renamed:
        extension.name = "custom"
    with pytest.raises(ValueError, match="built-in"):
        Agent(ScriptedModel(), extensions=[extension])


@pytest.mark.parametrize("boundary", ["model", "tool"])
async def test_steering_skips_remaining_tools_and_persists_matching_results(tmp_path, boundary):
    executed = []

    @tool(guidelines="Record execution.")
    def record(value: str) -> str:
        """Record a value."""
        executed.append(value)
        return value

    calls = (ToolCall("a", "record", {"value": "a"}), ToolCall("b", "record", {"value": "b"}))
    model = ScriptedModel(AssistantMessage(tool_calls=calls), AssistantMessage(content="redirected"))
    persistence = SQLiteSessionExtension(tmp_path / "session.db")
    agent = await Agent.create(
        model,
        config=AgentConfig("s", request_id="r"),
        tools=[record],
        extensions=[persistence],
        max_iterations=1,
    )
    try:
        events = []
        trigger = AgentEventType.MODEL_COMPLETED if boundary == "model" else AgentEventType.TOOL_COMPLETED
        sent = False
        async for event in agent.stream("original"):
            events.append(event)
            if event.type == trigger and not sent:
                sent = True
                assert await asyncio.to_thread(
                    agent.emit_external_event,
                    ExternalEvent("steering_message", {"session_id": "s", "request_id": "r", "content": "new task"}),
                )
        assert executed == ([] if boundary == "model" else ["a"])
        assert model.requests[-1].messages[-1] == UserMessage(content="new task")
        results = [message for message in model.requests[-1].messages if isinstance(message, ToolMessage)]
        assert [message.tool_call_id for message in results] == ["a", "b"]
        assert results[-1].success is False
        assert "Superseded" in results[-1].content
        skipped = [event for event in events if event.type == AgentEventType.TOOL_SKIPPED]
        assert len(skipped) == (2 if boundary == "model" else 1)
        assert all(event.phase == AgentPhase.READY for event in skipped)
        assert sum(event.type == AgentEventType.STEERING_STARTED for event in events) == 1
        assert sum(event.type == AgentEventType.STEERING_COMPLETED for event in events) == 1
        assert events[-1].type == AgentEventType.RUN_COMPLETED
        records = persistence.list_raw_messages("s")
        assert [record.message.role.value for record in records] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
            "assistant",
        ]
        restored = await persistence.storage.load("s")
        assert restored.messages[-2] == UserMessage(content="new task")
        assert not agent.emit_external_event(ExternalEvent("steering_message", {"session_id": "s", "content": "late"}))
    finally:
        persistence.close()


async def test_steering_precedes_internal_and_new_steering_interrupts_active_internal():
    class Inject(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "initial":
                await context.publish(InternalMessageEvent(AgentMessage(content="internal")))
                await context.publish(SteeringMessageEvent(UserMessage(content="steer first")))
            elif response.message.tool_calls:
                await context.publish(SteeringMessageEvent(UserMessage(content="steer again")))

    model = ScriptedModel(
        AssistantMessage(content="initial"),
        AssistantMessage(content="first steering done"),
        AssistantMessage(tool_calls=(ToolCall("skip", "unused"),)),
        AssistantMessage(content="done"),
    )
    agent = await Agent.create(model, config=AgentConfig("s"), extensions=[Inject()])
    events = [event async for event in agent.stream("original")]
    assert [request.messages[-1].content for request in model.requests] == [
        "original",
        "steer first",
        "internal",
        "steer again",
    ]
    interrupted = [event for event in events if event.type == AgentEventType.INTERNAL_MESSAGE_INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].internal_message.content == "internal"
    assert not any(event.type == AgentEventType.INTERNAL_MESSAGE_COMPLETED for event in events)


async def test_new_steering_interrupts_previous_steering_without_false_completion():
    class Inject(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "initial":
                await context.publish(SteeringMessageEvent(UserMessage(content="first")))
            elif response.message.tool_calls:
                await context.publish(SteeringMessageEvent(UserMessage(content="second")))

    model = ScriptedModel(
        AssistantMessage(content="initial"),
        AssistantMessage(tool_calls=(ToolCall("t", "unused"),)),
        AssistantMessage(content="done"),
    )
    agent = await Agent.create(model, config=AgentConfig("s"), extensions=[Inject()])
    events = [event async for event in agent.stream("original")]
    assert [
        event.steering_message.content for event in events if event.type == AgentEventType.STEERING_INTERRUPTED
    ] == ["first"]
    assert [event.steering_message.content for event in events if event.type == AgentEventType.STEERING_COMPLETED] == [
        "second"
    ]


async def test_steering_waits_for_running_tool_then_redirects():
    entered, release = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Wait for an external release.")
    async def wait() -> str:
        """Wait until released."""
        entered.set()
        await release.wait()
        return "released"

    model = ScriptedModel(AssistantMessage(tool_calls=(ToolCall("t", "wait"),)), AssistantMessage(content="done"))
    agent = await Agent.create(model, config=AgentConfig("s"), tools=[wait])
    task = asyncio.create_task(agent.run("original"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert agent.emit_external_event(ExternalEvent("steering_message", {"session_id": "s", "content": "new"}))
        assert not task.done()
        release.set()
        assert (await asyncio.wait_for(task, 2)).content == "done"
        assert model.requests[-1].messages[-1].content == "new"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "config,payload",
    [
        (AgentConfig("s"), {}),
        (AgentConfig("other"), {"content": "x"}),
        (AgentConfig("s"), {"content": ""}),
        (AgentConfig("s", request_id="other"), {"content": "x"}),
    ],
)
async def test_invalid_steering_is_rejected_and_close_cleans_queue(config, payload):
    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="fresh")), config=AgentConfig("s", request_id="r"), extensions=[]
    )
    stream = agent.stream("cancel")
    await anext(stream)
    assert not agent.emit_external_event(ExternalEvent("steering_message", payload), config=config)
    assert agent.emit_external_event(ExternalEvent("steering_message", {"session_id": "s", "content": "stale"}))
    await stream.aclose()
    assert not agent.emit_external_event(ExternalEvent("steering_message", {"session_id": "s", "content": "late"}))
    assert (await agent.run("fresh request")).content == "fresh"
    assert not any(message.content == "stale" for message in agent.state.messages)


def test_steering_requires_a_user_message():
    with pytest.raises(TypeError, match="UserMessage"):
        SteeringMessageEvent(AgentMessage(content="internal"))


async def test_steering_is_not_limited_by_internal_quota_and_resets_message_budget():
    class Inject(AgentExtension):
        async def after_model(self, context, response):
            if response.message.content == "initial":
                for number in range(10):
                    await context.publish(SteeringMessageEvent(UserMessage(content=str(number))))

    model = ScriptedModel(AssistantMessage(content="initial"), *[AssistantMessage(content="done") for _ in range(10)])
    agent = await Agent.create(
        model, config=AgentConfig("s"), extensions=[Inject()], max_internal_messages=0, max_iterations=1
    )
    assert (await agent.run("original")).content == "done"
    assert [request.messages[-1].content for request in model.requests] == ["original", *map(str, range(10))]


async def test_model_error_discards_pending_steering_and_allows_reuse():
    class FailingOnce:
        fail = True

        async def stream(self, request):
            if self.fail:
                self.fail = False
                yield ModelEvent.text("partial")
                raise ValueError("model failed")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="fresh")))

    agent = await Agent.create(FailingOnce(), config=AgentConfig("s"), extensions=[])
    with pytest.raises(ValueError, match="model failed"):
        async for event in agent.stream("old"):
            if event.type == AgentEventType.TEXT_DELTA:
                assert agent.emit_external_event(
                    ExternalEvent("steering_message", {"session_id": "s", "content": "stale"})
                )
    assert (await agent.run("new")).content == "fresh"
    assert not any(message.content == "stale" for message in agent.state.messages)
