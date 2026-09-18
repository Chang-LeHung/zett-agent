from dataclasses import FrozenInstanceError

import pytest

from zett_agent import (
    AgentExtension,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    CompactionEvent,
    InMemoryMessageAccumulator,
    MessageAppendedEvent,
    MessageTiming,
    ModelUsage,
    SystemMessage,
    UserMessage,
)
from zett_agent.extensions.compaction import CompactedMessage


async def test_context_appends_message_before_publishing_its_event():
    observed = []

    class Subscriber(AgentExtension):
        async def on_event(self, context, event):
            assert context.state.messages[-1] is event.message
            observed.append(event)

    context = AgentRunContext(
        AgentRunConfig("session"),
        AgentState(),
        {},
        (Subscriber(),),
    )
    message = UserMessage(content="New message")
    timing = MessageTiming.instant()

    await context.append_message(message, timing)

    assert context.state.messages == (message,)
    assert observed == [MessageAppendedEvent(message, timing)]


def test_message_event_rejects_usage_for_non_assistant_messages():
    with pytest.raises(ValueError, match="only to assistant"):
        MessageAppendedEvent(UserMessage(content="Invalid"), MessageTiming.instant(), ModelUsage(input_tokens=1))


async def test_publish_preserves_order_and_stops_on_handler_failure():
    calls = []
    event = CompactionEvent(1, 2, 3, 4, "Summary")

    class Subscriber(AgentExtension):
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        async def on_event(self, context, received):
            assert received is event
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("Subscriber failed")

    context = AgentRunContext(
        AgentRunConfig("session"),
        AgentState(),
        {},
        (Subscriber("first"), Subscriber("second", fail=True), Subscriber("third")),
    )
    with pytest.raises(RuntimeError, match="Subscriber failed"):
        await context.publish(event)
    assert calls == ["first", "second"]
    with pytest.raises(FrozenInstanceError):
        event.summary = "Changed"


async def test_publish_can_target_one_named_extension():
    calls = []
    event = CompactionEvent(1, 2, 3, 4, "Summary")

    class Subscriber(AgentExtension):
        def __init__(self, name):
            self.name = name

        async def on_event(self, context, received):
            assert received is event
            calls.append(self.name)

    context = AgentRunContext(
        AgentRunConfig("session"),
        AgentState(),
        {},
        (Subscriber("first"), Subscriber("second"), Subscriber("third")),
    )

    await context.publish(event, target="second")

    assert calls == ["second"]


@pytest.mark.parametrize("target", ["", "missing"])
async def test_publish_rejects_invalid_or_unknown_target(target):
    context = AgentRunContext(AgentRunConfig("session"), AgentState(), {}, (AgentExtension(),))

    with pytest.raises(ValueError, match="target"):
        await context.publish(CompactionEvent(1, 2, 3, 4, "Summary"), target=target)


async def test_publish_rejects_ambiguous_target_without_delivery():
    calls = []

    class Subscriber(AgentExtension):
        name = "duplicate"

        async def on_event(self, context, event):
            calls.append(event)

    context = AgentRunContext(
        AgentRunConfig("session"),
        AgentState(),
        {},
        (Subscriber(), Subscriber()),
    )

    with pytest.raises(ValueError, match="Ambiguous"):
        await context.publish(CompactionEvent(1, 2, 3, 4, "Summary"), target="duplicate")
    assert calls == []


async def test_memory_accumulator_ignores_system_events_and_tracks_compaction():
    accumulator = InMemoryMessageAccumulator()
    context = AgentRunContext(
        AgentRunConfig("session"),
        AgentState(_messages=[SystemMessage(content="Current"), UserMessage(content="Old")]),
        {},
        (accumulator,),
    )
    await accumulator.on_state(context)
    await accumulator.on_event(
        context,
        MessageAppendedEvent(SystemMessage(content="Transient"), MessageTiming.instant()),
    )

    checkpoint = CompactedMessage(content="Checkpoint")
    recent = UserMessage(content="Recent")
    context.replace_messages([SystemMessage(content="Current"), checkpoint, recent], emit_new=False)
    await accumulator.on_event(context, CompactionEvent(1, 1, 2, 2, checkpoint.content))

    assert accumulator.messages("session") == (checkpoint, recent)
