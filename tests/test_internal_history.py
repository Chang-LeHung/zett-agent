"""Verify internal messages survive raw-history and checkpoint restoration."""

import pytest

from zett_agent import (
    Agent,
    AgentEventType,
    AgentExtension,
    AgentMessage,
    AgentRunConfig,
    AssistantMessage,
    InternalMessageEvent,
    ModelEvent,
    ModelResponse,
    SQLiteSessionExtension,
    UserMessage,
)
from zett_agent.extensions.compaction import CompactedMessage
from zett_agent.storage import decode_messages, encode_messages


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
