from dataclasses import FrozenInstanceError

import pytest

from kcs_agent import (
    AgentConfig,
    AgentContext,
    AgentExtension,
    AgentState,
    CompactionEvent,
    InMemoryMessageAccumulator,
    MessageAppendedEvent,
    MessageTiming,
    SystemMessage,
    UserMessage,
)
from kcs_agent.compaction import CompactedMessage


async def test_context_appends_message_before_publishing_its_event():
    observed = []

    class Subscriber(AgentExtension):
        async def on_event(self, context, event):
            assert context.state.messages[-1] is event.message
            observed.append(event)

    context = AgentContext(
        AgentConfig("session"),
        AgentState(),
        {},
        (Subscriber(),),
    )
    message = UserMessage(content="New message")
    timing = MessageTiming.instant()

    await context.append_message(message, timing)

    assert context.state.messages == [message]
    assert observed == [MessageAppendedEvent(message, timing)]


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

    context = AgentContext(
        AgentConfig("session"),
        AgentState(),
        {},
        (Subscriber("first"), Subscriber("second", fail=True), Subscriber("third")),
    )
    with pytest.raises(RuntimeError, match="Subscriber failed"):
        await context.publish(event)
    assert calls == ["first", "second"]
    with pytest.raises(FrozenInstanceError):
        event.summary = "Changed"


async def test_memory_accumulator_ignores_system_events_and_tracks_compaction():
    accumulator = InMemoryMessageAccumulator()
    context = AgentContext(
        AgentConfig("session"),
        AgentState(messages=[SystemMessage(content="Current"), UserMessage(content="Old")]),
        {},
        (accumulator,),
    )
    await accumulator.on_message(context)
    await accumulator.on_event(
        context,
        MessageAppendedEvent(SystemMessage(content="Transient"), MessageTiming.instant()),
    )

    checkpoint = CompactedMessage(content="Checkpoint")
    recent = UserMessage(content="Recent")
    context.state.messages[:] = [SystemMessage(content="Current"), checkpoint, recent]
    await accumulator.on_event(context, CompactionEvent(1, 1, 2, 2, checkpoint.content))

    assert accumulator.messages("session") == (checkpoint, recent)
