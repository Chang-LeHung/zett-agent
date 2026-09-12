"""Explicit request routing for externally delivered events."""

import pytest

from zett_agent import (
    Agent,
    AgentEventType,
    AgentExtension,
    AgentRunConfig,
    AssistantMessage,
    ExternalEvent,
    ModelEvent,
    ModelResponse,
)


class AnswerModel:
    async def stream(self, request):
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))


@pytest.mark.parametrize("explicit", [True, False])
async def test_external_event_config_routes_steering_without_payload_identity(explicit):
    agent = await Agent.create(AnswerModel(), config=AgentRunConfig("default"))
    config = AgentRunConfig("active", request_id="request")
    stream = agent.stream("initial", config=config)
    assert (await anext(stream)).type == AgentEventType.MODEL_STARTED
    event = ExternalEvent("steering_message", {"content": "redirect"})
    kwargs = {"config": config} if explicit else {}
    assert agent.emit_external_event(event, **kwargs) == ["SteeringExtension"]
    assert event.payload == {"content": "redirect"}
    remaining = [event async for event in stream]
    assert any(event.type == AgentEventType.STEERING_STARTED for event in remaining)
    assert all(event.session_id == "active" for event in remaining)


async def test_extensions_route_by_config_not_payload_and_reject_mismatched_request():
    agent = await Agent.create(AnswerModel(), config=AgentRunConfig("s"))
    stream = agent.stream("initial", config=AgentRunConfig("s", request_id="r"))
    await anext(stream)
    try:
        event = ExternalEvent("steering_message", {"session_id": "other", "request_id": "other", "content": "x"})
        assert agent.emit_external_event(event, config=AgentRunConfig("s")) == ["SteeringExtension"]
        assert event.payload["session_id"] == "other"
        assert not agent.emit_external_event(
            ExternalEvent("steering_message", {"content": "x"}), config=AgentRunConfig("s", request_id="other")
        )
    finally:
        await stream.aclose()


@pytest.mark.parametrize("initialized", [True, False])
async def test_broadcast_preserves_objects_and_returns_all_accepting_names_in_priority_order(initialized):
    received = []

    class Receiver(AgentExtension):
        def __init__(self, name, priority, accepts):
            self.name = name
            self.priority = priority
            self.accepts = accepts

        def accept(self, config, event):
            received.append((self.name, config, event))
            return self.accepts

    config = AgentRunConfig("session", request_id="request")
    agent = Agent(
        AnswerModel(),
        extensions=[Receiver("last", 300, True), Receiver("first", 1, True), Receiver("ignored", 20, False)],
    )
    if initialized:
        await agent.initialize(config=config)
    event = ExternalEvent("custom", {"session_id": "business-field"})
    assert agent.emit_external_event(event) == ["first", "last"]
    assert [name for name, _, _ in received] == ["first", "ignored", "last"]
    assert all(item is event for _, _, item in received)
    assert all(routing is (config if initialized else None) for _, routing, _ in received)
    received.clear()
    explicit = AgentRunConfig("explicit")
    assert agent.emit_external_event(event, config=explicit) == ["first", "last"]
    assert all(routing is explicit for _, routing, _ in received)
    assert event.payload == {"session_id": "business-field"}


def test_no_receivers_returns_empty_list():
    assert Agent(AnswerModel()).emit_external_event(ExternalEvent("unknown", {})) == []
