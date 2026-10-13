"""Verify internal messages survive raw-history and checkpoint restoration."""

import pytest

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.events import AgentEventType
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.compaction import CompactedMessage
from zett_agent.extensions.events import InternalMessageEvent
from zett_agent.extensions.sqlite import SQLiteSessionExtension
from zett_agent.messages import (
    AgentMessage,
    AssistantMessage,
    UserMessage,
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)
from zett_agent.storage import _message_adapter, decode_messages, encode_messages


def test_decoding_reuses_one_pydantic_adapter_per_message_kind():
    _message_adapter.cache_clear()
    messages = [UserMessage(content=f"item-{index}") for index in range(5)]
    messages.append(AssistantMessage(content="answer"))

    decoded = decode_messages(encode_messages(messages))

    assert decoded[0].content == "item-0" and decoded[-1].content == "answer"
    cache = _message_adapter.cache_info()
    assert (cache.misses, cache.hits) == (2, 4)


class AnswerModel:
    def __init__(self):
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="answer")))


async def test_internal_messages_are_restored_from_raw_history_and_snapshot_tail(tmp_path):
    class Inject(AgentExtension):
        calls = 0

        async def after_model(self, context, response):
            self.calls += 1
            if self.calls in (1, 2):
                await context.publish(InternalMessageEvent(AgentMessage(content=f"internal {self.calls}")))

    persistence = SQLiteSessionExtension(tmp_path / "history.db")
    model = AnswerModel()
    agent = await Agent.create(model, config=AgentRunConfig("priority"), extensions=[Inject(), persistence])
    try:
        events = []
        async for event in agent.stream("initial"):
            events.append(event)
        assert [request.messages[-1].content for request in model.requests] == [
            "initial",
            "internal 1",
            "internal 2",
        ]
        records = await persistence.list_raw_messages("priority")
        assert [record.message.role.value for record in records] == [
            "system",
            "user",
            "assistant",
            "agent",
            "assistant",
            "agent",
            "assistant",
        ]
        restored = decode_messages(encode_messages([record.message for record in records]))
        assert isinstance(restored[3], AgentMessage)
        assert restored[3].content == "internal 1"
        view = await persistence.storage.load("priority")
        assert isinstance(view.messages[2], AgentMessage)
        await persistence.storage.snapshot("priority", CompactedMessage(content="Earlier exchange"), 3, 0)
        view = await persistence.storage.load("priority")
        assert isinstance(view.messages[1], AgentMessage)
        assert view.messages[1].content == "internal 1"
        assert len(await persistence.list_raw_messages("priority")) == 7
        assert sum(event.type == AgentEventType.INTERNAL_MESSAGE_STARTED for event in events) == 2
        assert sum(event.type == AgentEventType.INTERNAL_MESSAGE_COMPLETED for event in events) == 2
    finally:
        await persistence.close()


def test_internal_event_rejects_user_messages():
    with pytest.raises(TypeError, match="AgentMessage"):
        InternalMessageEvent(UserMessage(content="user"))
