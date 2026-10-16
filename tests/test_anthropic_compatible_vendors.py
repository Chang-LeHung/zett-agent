"""The Anthropic-compatible base class, its hooks, and the providers built on them.

Every test drives the real SDK through ``httpx.MockTransport``, so what is asserted is the HTTP request the
service would receive and the events the runtime would see. The bodies are written from each service's
documentation, not recorded from a live service: they pin the behavior the hooks promise, and a service
that has changed since is for a recording to show.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from zett_agent.json_types import JsonValue
from zett_agent.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from zett_agent.model import (
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    ToolDefinition,
)
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.anthropic_compatible import AnthropicCompatibleProvider
from zett_agent.providers.base import ProviderAuthError, ProviderResponseError
from zett_agent.providers.deepseek import DeepSeekAnthropicProvider
from zett_agent.providers.glm import GLMAnthropicProvider
from zett_agent.providers.kimi import KimiAnthropicProvider
from zett_agent.providers.wire import JsonObject
from zett_agent.stop import StopInfo, StopReason

NO_RETRY = RetryOptions(max_retries=0)
HELLO = (UserMessage(content="Hello"),)


def sse(*events: Mapping[str, Any]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def answer(text: str = "ok", usage: Mapping[str, Any] | None = None, **delta_usage: Any) -> bytes:
    return sse(
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 3, **(usage or {})}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2, **delta_usage}},
        {"type": "message_stop"},
    )


THINKING_TOOL = sse(
    {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 3}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "see."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}},
    },
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "1}"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
)


class Wire:
    """A transport that replays scripted bodies and keeps every request it receives."""

    def __init__(self, *bodies: bytes) -> None:
        self.bodies = list(bodies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def sent(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


async def run(provider: AnthropicCompatibleProvider, *messages: Any, **request: Any) -> list[ModelEvent]:
    try:
        return [event async for event in provider.stream(ModelRequest(messages or HELLO, **request))]
    finally:
        await provider.aclose()


def final(events: list[ModelEvent]):
    assert events[-1].type is ModelEventType.RESPONSE and events[-1].response is not None
    return events[-1].response


# ------------------------------------------------------------------ the base class and its hooks


async def test_a_compatible_endpoint_works_with_nothing_but_a_base_url():
    wire = Wire(answer("hi"))
    provider = AnthropicCompatibleProvider(
        model="local", api_key="k", base_url="https://gateway.example", transport=wire.transport
    )

    response = final(await run(provider))

    request = wire.requests[0]
    assert str(request.url) == "https://gateway.example/v1/messages"
    assert request.headers["x-api-key"] == "k" and "authorization" not in request.headers
    body = wire.sent()
    assert "cache_control" not in json.dumps(body) and "tools" not in body  # nothing unasked-for
    assert body["thinking"]["type"] == "enabled"  # the Anthropic default for hooks not overridden
    assert response.message.content == "hi" and response.message.provider == "anthropic"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (3, 2)


class Quirky(AnthropicCompatibleProvider):
    """A service that overrides every hook once, to show each is called and each result is used."""

    provider_name = "quirky"

    def vendor_headers(self) -> dict[str, str]:
        return {"X-Quirk": "1", "X-Replaced": "vendor"}

    def request_fields(self, request: ModelRequest) -> JsonObject:
        return {"quirk": request.reasoning_effort.value}

    def mark_cache(self, body: JsonObject, client_tools: list[JsonObject]) -> None:
        body["system"] = "marked"

    def start_usage(self, usage: Mapping[str, JsonValue]) -> ModelUsage:
        return ModelUsage(input_tokens=70)

    def finish_usage(self, usage: ModelUsage, update: Mapping[str, JsonValue]) -> ModelUsage:
        return ModelUsage(input_tokens=usage.input_tokens, output_tokens=30)

    def message_stop(self, finish_reason, details, *, has_tool_calls):
        return StopInfo(StopReason.MAX_OUTPUT, raw="quirk")


async def test_every_hook_is_called_and_what_it_returns_is_used():
    wire = Wire(answer())
    provider = Quirky(
        model="m",
        api_key="k",
        base_url="https://q.example",
        transport=wire.transport,
        extra_headers={"X-Replaced": "app"},
        extra_body={"extra": True},
    )

    response = final(await run(provider))

    headers = wire.requests[0].headers
    assert headers["x-quirk"] == "1" and headers["x-replaced"] == "app"  # the application outranks the vendor
    body = wire.sent()
    assert body["quirk"] == "medium" and body["system"] == "marked" and body["extra"] is True
    assert "thinking" not in body  # request_fields replaced the default wholesale
    assert (response.usage.input_tokens, response.usage.output_tokens) == (70, 30)
    assert response.stop.reason is StopReason.MAX_OUTPUT


async def test_a_subclass_can_turn_an_http_failure_into_its_own_error():
    class Strict(AnthropicCompatibleProvider):
        def status_error(self, error):
            return ProviderAuthError("custom: " + str(error.status_code))

    provider = Strict(
        model="m",
        api_key="k",
        base_url="https://s.example",
        transport=httpx.MockTransport(lambda r: httpx.Response(418, json={"error": {"message": "teapot"}})),
        retry=NO_RETRY,
    )

    with pytest.raises(ProviderAuthError, match="custom: 418"):
        await run(provider)


async def test_bearer_services_get_both_credentials_and_others_only_the_key():
    wire = Wire(answer())
    await run(KimiAnthropicProvider("kimi-k3", "secret", transport=wire.transport))
    headers = wire.requests[0].headers
    assert headers["authorization"] == "Bearer secret" and headers["x-api-key"] == "secret"


async def test_anthropic_keeps_its_cache_markers_native_tool_search_and_signed_thinking():
    wire = Wire(THINKING_TOOL, answer())
    provider = AnthropicProvider("claude", "k", transport=wire.transport, retry=NO_RETRY)
    tool = ToolDefinition("add", "Add", {"type": "object"})

    first = final([event async for event in provider.stream(ModelRequest(HELLO, tools=(tool,)))])
    assert first.message.replay_blocks[0] == {
        "type": "thinking",
        "thinking": "Let me see.",
        "signature": "sig-1",
    }
    assert first.message.tool_calls[0].arguments == {"a": 1}
    assert first.message.reasoning == "Let me see."

    history = (*HELLO, first.message, ToolMessage(tool_call_id="toolu_1", name="add", content="2"))
    await run(provider, *history, tools=(tool,))

    body = wire.sent()
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][1]["content"][0]["signature"] == "sig-1"  # sent back unchanged
    assert provider.native_deferred_tools is True


async def test_replay_blocks_go_back_only_to_the_provider_that_wrote_them():
    wire = Wire(answer())
    foreign = AssistantMessage(
        content="text",
        provider="kimi-anthropic",
        model="m",
        replay_blocks=({"type": "thinking", "thinking": "x", "signature": "s"},),
    )
    provider = AnthropicCompatibleProvider(
        model="m", api_key="k", base_url="https://g.example", transport=wire.transport
    )

    await run(provider, *HELLO, foreign, UserMessage(content="more"))

    assert wire.sent()["messages"][1]["content"] == [{"type": "text", "text": "text"}]


async def test_an_error_event_in_the_middle_of_a_stream_keeps_what_was_written():
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    wire = Wire(
        sse(
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 3}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
            overloaded,
        )
    )
    provider = AnthropicCompatibleProvider(
        model="m", api_key="k", base_url="https://g.example", transport=wire.transport, retry=NO_RETRY
    )

    response = final(await run(provider))

    assert response.message.content == "partial"
    assert response.stop.reason is StopReason.INTERRUPTED
    assert response.stop.message == "Overloaded"


async def test_an_error_event_before_any_output_is_a_typed_error():
    wire = Wire(
        sse(
            {"type": "message_start", "message": {"id": "m", "usage": {}}},
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
    )
    provider = AnthropicCompatibleProvider(
        model="m", api_key="k", base_url="https://g.example", transport=wire.transport, retry=NO_RETRY
    )

    with pytest.raises(ProviderResponseError, match="Overloaded"):
        await run(provider)


async def test_system_prompts_are_joined_and_tools_are_left_out_when_there_are_none():
    wire = Wire(answer())
    provider = AnthropicCompatibleProvider(
        model="m", api_key="k", base_url="https://g.example", transport=wire.transport
    )

    await run(provider, SystemMessage(content="one"), SystemMessage(content="two"), *HELLO)

    body = wire.sent()
    assert body["system"] == "one\n\ntwo" and "tools" not in body


# --------------------------------------------------------------------------------------- DeepSeek


@pytest.mark.parametrize(
    ("model", "effort", "fields"),
    [
        ("deepseek-v4-pro", ReasoningEffort.OFF, {"thinking": {"type": "disabled"}}),
        (
            "deepseek-v4-pro",
            ReasoningEffort.MINIMAL,
            {"thinking": {"type": "enabled"}, "output_config": {"effort": "low"}},
        ),
        (
            "deepseek-v4-pro",
            ReasoningEffort.MEDIUM,
            {"thinking": {"type": "enabled"}, "output_config": {"effort": "high"}},
        ),
        (
            "deepseek-v4-flash",
            ReasoningEffort.ULTRA,
            {"thinking": {"type": "enabled"}, "output_config": {"effort": "max"}},
        ),
        ("deepseek-chat", ReasoningEffort.HIGH, {"thinking": {"type": "enabled"}}),  # no level before V4
    ],
)
def test_deepseek_switches_thinking_without_a_budget_and_gives_v4_its_level(model, effort, fields):
    provider = DeepSeekAnthropicProvider(model, "k")
    assert provider.request_fields(ModelRequest(HELLO, reasoning_effort=effort)) == fields


async def test_deepseek_sends_no_cache_markers_and_uses_the_x_api_key_header():
    wire = Wire(answer())
    await run(DeepSeekAnthropicProvider("deepseek-v4-pro", "k", transport=wire.transport))

    request = wire.requests[0]
    assert str(request.url) == "https://api.deepseek.com/anthropic/v1/messages"
    assert request.headers["x-api-key"] == "k" and "authorization" not in request.headers
    body = wire.sent()
    assert "cache_control" not in json.dumps(body) and "budget_tokens" not in json.dumps(body)
    assert body["max_tokens"] == 16_384  # not raised by a thinking budget the service ignores


# ----------------------------------------------------------------------------------------- Kimi


@pytest.mark.parametrize(
    ("model", "effort", "key", "fields"),
    [
        ("kimi-k3", ReasoningEffort.HIGH, None, {"output_config": {"effort": "high"}}),
        ("kimi-k3", ReasoningEffort.ULTRA, None, {"output_config": {"effort": "max"}}),
        (
            "kimi-k3",
            ReasoningEffort.MINIMAL,
            "chat-1",
            {"output_config": {"effort": "low"}, "metadata": {"user_id": "chat-1"}},
        ),
        ("kimi-k3", ReasoningEffort.OFF, None, {}),  # always reasons
        ("kimi-k2.6", ReasoningEffort.HIGH, None, {}),  # no documented Messages field: none sent
        ("kimi-k2.6", ReasoningEffort.HIGH, "chat-1", {"metadata": {"user_id": "chat-1"}}),
    ],
)
def test_kimi_sends_its_level_in_output_config_and_the_cache_key_as_the_user_id(model, effort, key, fields):
    provider = KimiAnthropicProvider(model, "k")
    request = ModelRequest(HELLO, reasoning_effort=effort, cache_key=key)
    assert provider.request_fields(request) == fields


async def test_kimi_sends_no_thinking_field_and_no_message_level_cache_marker():
    wire = Wire(answer())
    await run(KimiAnthropicProvider("kimi-k3", "k", transport=wire.transport), cache_key="s-1")

    body = wire.sent()
    assert str(wire.requests[0].url) == "https://api.moonshot.ai/anthropic/v1/messages"
    assert "thinking" not in body and "cache_control" not in json.dumps(body)
    assert body["metadata"] == {"user_id": "s-1"}


async def test_kimi_cache_key_can_be_withheld():
    wire = Wire(answer())
    await run(KimiAnthropicProvider("kimi-k3", "k", transport=wire.transport, send_cache_key=False), cache_key="s-1")
    assert "metadata" not in wire.sent()


async def test_kimi_reports_reasoning_tokens_from_either_usage_object():
    start_only = Wire(
        answer(usage={"output_tokens": 5, "output_tokens_details": {"thinking_tokens": 4}}, output_tokens=9)
    )
    response = final(await run(KimiAnthropicProvider("kimi-k3", "k", transport=start_only.transport)))
    assert response.usage.reasoning_tokens == 4 and response.usage.output_tokens == 9

    at_the_end = Wire(answer(output_tokens=9, output_tokens_details={"thinking_tokens": 7}))
    response = final(await run(KimiAnthropicProvider("kimi-k3", "k", transport=at_the_end.transport)))
    assert response.usage.reasoning_tokens == 7

    not_reported = Wire(answer())
    response = final(await run(KimiAnthropicProvider("kimi-k3", "k", transport=not_reported.transport)))
    assert response.usage.reasoning_tokens == 0


async def test_kimi_sends_signed_thinking_back_unchanged_to_kimi_only():
    wire = Wire(THINKING_TOOL, answer())
    provider = KimiAnthropicProvider("kimi-k3", "k", transport=wire.transport, retry=NO_RETRY)
    tool = ToolDefinition("add", "Add", {"type": "object"})

    first = final([event async for event in provider.stream(ModelRequest(HELLO, tools=(tool,)))])
    assert first.message.provider == "kimi-anthropic"

    history = (*HELLO, first.message, ToolMessage(tool_call_id="toolu_1", name="add", content="2"))
    await run(provider, *history, tools=(tool,))
    assert wire.sent()["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "Let me see.",
        "signature": "sig-1",
    }


# ------------------------------------------------------------------------------------------ GLM


@pytest.mark.parametrize(
    ("model", "effort", "thinking"),
    [
        ("glm-4.5", ReasoningEffort.HIGH, None),  # before thinking existed
        ("glm-4.6", ReasoningEffort.HIGH, "enabled"),
        ("glm-4.6", ReasoningEffort.OFF, "disabled"),
        ("glm-5.3", ReasoningEffort.MEDIUM, "enabled"),
        ("glm-5.3", ReasoningEffort.OFF, None),  # only ever thinks
    ],
)
def test_glm_asks_for_thinking_only_from_the_models_that_have_it(model, effort, thinking):
    fields = GLMAnthropicProvider(model, "k").request_fields(ModelRequest(HELLO, reasoning_effort=effort))
    assert (fields.get("thinking") or {}).get("type") == thinking  # type: ignore[union-attr]  # JsonValue narrowing


async def test_glm_authenticates_with_a_bearer_token_at_the_z_ai_endpoint():
    wire = Wire(answer())
    await run(GLMAnthropicProvider("glm-5.1", "secret", transport=wire.transport))
    request = wire.requests[0]
    assert str(request.url) == "https://api.z.ai/api/anthropic/v1/messages"
    assert request.headers["authorization"] == "Bearer secret"


async def test_the_forced_tool_choice_turns_thinking_off_for_the_standard_fields():
    provider = AnthropicCompatibleProvider(model="m", api_key="k", base_url="https://g.example")
    request = ModelRequest(HELLO, tool_choice="add")
    assert provider.request_fields(request) == {"thinking": {"type": "disabled"}}
    await provider.aclose()


def test_a_tool_call_in_history_renders_as_a_tool_use_block():
    provider = AnthropicCompatibleProvider(model="m", api_key="k", base_url="https://g.example")
    messages, system = provider._messages(
        ModelRequest((AssistantMessage(content="", tool_calls=(ToolCall("c1", "add", {"a": 1}),)),))
    )
    assert messages == [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 1}}]}
    ]
    assert system == []
