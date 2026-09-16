"""Verify restoration, input transformation, and durable append boundaries."""

import pytest

from zett_agent import (
    Agent,
    AgentExtension,
    AgentProtocolError,
    AgentRunConfig,
    AssistantMessage,
    MessageAppendedEvent,
    ModelEvent,
    ModelResponse,
    SQLiteSessionExtension,
    UserMessage,
)


@pytest.mark.parametrize("priority", [0, 200])
async def test_restore_then_transform_then_persist_once_before_run(tmp_path, priority):
    storage = SQLiteSessionExtension(tmp_path / "history.sqlite3")
    order = []
    original = UserMessage(content="current", attributes={"source": "test"})
    prior = UserMessage(content="history")
    config = AgentRunConfig("setup")

    class Transform(AgentExtension):
        async def on_tool(self, context):
            order.append("tool")

        async def on_state(self, context):
            order.append("state")

        async def on_message(self, context):
            order.append("message")
            assert context.state.messages[-1] == prior
            assert context.input_message is original
            context.input_message = UserMessage(content="transformed", attributes={"raw_content": original.content})

        async def on_event(self, context, event):
            if isinstance(event, MessageAppendedEvent) and isinstance(event.message, UserMessage):
                order.append("append")
                assert event.message.content == "transformed"

        async def before_run(self, context):
            order.append("before_run")
            records = await storage.list_raw_messages(config.session_id)
            assert [record.message.content for record in records] == ["history", "transformed"]
            assert records[-1].message == context.state.messages[-1]

    class Model:
        async def stream(self, request):
            order.append("model")
            assert request.messages[-1].content == "transformed"
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="answer")))

    extension = Transform()
    extension.priority = priority
    try:
        await storage.storage.append(config.session_id, "previous", prior)
        agent = await Agent.create(Model(), config=config, extensions=[extension, storage])
        await agent.run(original)
        assert order == ["tool", "state", "message", "append", "before_run", "model"]
        restored = await storage.storage.load(config.session_id)
        assert [message.content for message in restored.messages] == ["history", "transformed", "answer"]
        assert restored.messages[1].attributes == {"raw_content": "current"}
        assert original.content == "current"
    finally:
        await storage.close()


async def test_invalid_transformed_input_is_not_persisted(tmp_path):
    storage = SQLiteSessionExtension(tmp_path / "invalid.sqlite3")

    class InvalidInput(AgentExtension):
        async def on_message(self, context):
            context.input_message = None

    class UnusedModel:
        async def stream(self, request):
            raise AssertionError("Invalid input must fail before the model")
            yield

    try:
        agent = await Agent.create(
            UnusedModel(), config=AgentRunConfig("invalid"), extensions=[storage, InvalidInput()]
        )
        with pytest.raises(AgentProtocolError, match="must produce a UserMessage"):
            await agent.run("input")
        assert await storage.list_raw_messages("invalid") == []
    finally:
        await storage.close()
