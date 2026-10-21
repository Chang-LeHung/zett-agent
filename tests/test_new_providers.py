"""xAI, MiniMax, Mistral, Groq, Cerebras and the plain hosting services.

Every test drives the real SDK through ``httpx.MockTransport``. The bodies are written from each service's
documentation, not recorded from a live service: they pin what the adapters promise to send, and a service
that has changed since is for a recording to show.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from zett_agent.extensions.server_tools.xai import XAIServerToolExtension
from zett_agent.messages import AssistantMessage, UserMessage
from zett_agent.model import ModelEvent, ModelEventType, ModelRequest, ReasoningEffort
from zett_agent.providers import (
    CerebrasProvider,
    DeepInfraProvider,
    FireworksProvider,
    GroqProvider,
    MiniMaxAnthropicProvider,
    MiniMaxProvider,
    MistralProvider,
    OpenAICompatibleProvider,
    ProviderAuthError,
    SiliconFlowProvider,
    TogetherProvider,
    XAIProvider,
)

E = ReasoningEffort
HELLO = (UserMessage(content="Hello"),)


def sse(*events: Mapping[str, Any]) -> bytes:
    """A Chat Completions or Responses SSE body."""
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"


def messages_sse(*events: Mapping[str, Any]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def answer(text: str = "ok") -> bytes:
    """A one-text-block Messages stream."""
    return messages_sse(
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )


def chunk(delta: Mapping[str, Any] | None = None, finish: str | None = None) -> dict[str, Any]:
    choice = {"index": 0, "delta": dict(delta or {}), "finish_reason": finish}
    return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [choice]}


STOPPED = chunk(finish="stop")


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


async def run(provider: Any, *messages: Any, **request: Any) -> list[ModelEvent]:
    try:
        return [event async for event in provider.stream(ModelRequest(messages or HELLO, **request))]
    finally:
        await provider.aclose()


def final(events: list[ModelEvent]):
    assert events[-1].type is ModelEventType.RESPONSE and events[-1].response is not None
    return events[-1].response


def fields(provider: Any, effort: ReasoningEffort) -> dict[str, Any]:
    return dict(provider.request_fields(ModelRequest(HELLO, reasoning_effort=effort)))


# ------------------------------------------------------------------------------- the shared hooks


def test_a_generic_provider_still_requires_a_base_url():
    with pytest.raises(ValueError, match="base_url is required"):
        OpenAICompatibleProvider(model="m", api_key="k")


@pytest.mark.parametrize(
    ("effort", "expected"),
    [(E.OFF, "low"), (E.MINIMAL, "low"), (E.LOW, "low"), (E.MEDIUM, "medium"), (E.HIGH, "high"), (E.ULTRA, "high")],
)
def test_low_medium_high_groups_the_levels_a_service_does_not_name(effort, expected):
    from zett_agent.providers.openai_compatible import low_medium_high

    assert low_medium_high(effort) == expected


def test_a_tool_call_without_text_is_sent_with_null_content():
    from zett_agent.messages import ToolCall
    from zett_agent.providers.openai_compatible import chat_message

    call = ToolCall(id="c1", name="add", arguments={"left": 1})
    assert chat_message(AssistantMessage(content="", tool_calls=(call,)))["content"] is None
    # Text beside the call is kept, and a plain answer keeps its empty string.
    assert chat_message(AssistantMessage(content="Adding.", tool_calls=(call,)))["content"] == "Adding."
    assert chat_message(AssistantMessage(content=""))["content"] == ""


# ------------------------------------------------------------------------------------------- xAI


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("grok-4.7", E.OFF, {"reasoning_effort": "low"}),  # reasoning cannot be turned off
        ("grok-4.7", E.MINIMAL, {"reasoning_effort": "low"}),
        ("grok-4.6", E.MEDIUM, {"reasoning_effort": "medium"}),
        ("grok-4.5", E.HIGH, {"reasoning_effort": "high"}),
        ("grok-4.7", E.XHIGH, {"reasoning_effort": "xhigh"}),
        ("grok-4.7", E.ULTRA, {"reasoning_effort": "xhigh"}),
        ("grok-4", E.HIGH, {}),  # before the level existed
        ("grok-3-mini", E.HIGH, {}),
        ("custom-endpoint", E.HIGH, {}),
    ],
)
def test_xai_gives_only_the_models_that_take_one_a_level(model, effort, expected):
    assert fields(XAIProvider(model=model, api_key="k"), effort) == expected


async def test_xai_talks_to_its_own_endpoint_with_a_level_and_a_cache_key():
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    response = final(await run(XAIProvider(model="grok-4.7", api_key="k", transport=wire.transport), cache_key="run-1"))

    assert str(wire.requests[0].url) == "https://api.x.ai/v1/chat/completions"
    assert wire.requests[0].headers["authorization"] == "Bearer k"
    assert wire.sent()["reasoning_effort"] == "medium" and wire.sent()["prompt_cache_key"] == "run-1"
    assert response.message.content == "hi" and response.message.provider == "xai"


async def test_xai_on_responses_sends_effort_and_server_tools_without_a_summary():
    wire = Wire(
        sse(
            {"type": "response.output_text.delta", "output_index": 0, "delta": "ok"},
            {
                "type": "response.completed",
                "response": {"id": "r", "status": "completed", "output": [], "usage": {}},
            },
        )
    )
    provider = XAIProvider(model="grok-4.7", api_key="k", response=True, transport=wire.transport)
    tools = XAIServerToolExtension().server_tools
    await run(provider, server_tools=tools, reasoning_effort=E.XHIGH)

    body = wire.sent()
    assert str(wire.requests[0].url) == "https://api.x.ai/v1/responses"
    assert body["reasoning"] == {"effort": "xhigh"}  # no "summary": xAI does not document it
    assert body["include"] == ["reasoning.encrypted_content"]
    assert [tool["type"] for tool in body["tools"]] == ["web_search", "x_search"]


async def test_xai_before_4_5_sends_no_reasoning_object_on_responses():
    wire = Wire(
        sse({"type": "response.completed", "response": {"id": "r", "status": "completed", "output": [], "usage": {}}})
    )
    provider = XAIProvider(model="grok-4", api_key="k", response=True, transport=wire.transport)
    await run(provider)

    assert wire.sent().get("reasoning") is None


def test_xai_server_tools_default_to_both_searches_and_can_be_narrowed():
    assert [tool.type for tool in XAIServerToolExtension().server_tools] == ["web_search", "x_search"]
    narrowed = XAIServerToolExtension(["web_search"])
    assert [tool.type for tool in narrowed.server_tools] == ["web_search"]


# --------------------------------------------------------------------------------------- MiniMax


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("MiniMax-M3.1-Flash", E.MEDIUM, {"reasoning_split": True, "reasoning_effort": "medium"}),
        ("MiniMax-M3.1-Flash", E.ULTRA, {"reasoning_split": True, "reasoning_effort": "max"}),
        ("MiniMax-M3.1-Flash", E.OFF, {"reasoning_split": True}),  # cannot be disabled: a 400
        ("MiniMax-M3", E.HIGH, {"reasoning_split": True, "thinking": {"type": "adaptive"}}),
        ("MiniMax-M3", E.OFF, {"reasoning_split": True, "thinking": {"type": "disabled"}}),
        ("MiniMax-M2.7", E.HIGH, {"reasoning_split": True}),
        ("MiniMax-M2.7", E.OFF, {"reasoning_split": True}),
        ("something-else", E.HIGH, {"reasoning_split": True}),
    ],
)
def test_minimax_chat_spells_reasoning_the_way_each_generation_takes_it(model, effort, expected):
    assert fields(MiniMaxProvider(model=model, api_key="k"), effort) == expected


async def test_minimax_chat_reads_the_separate_reasoning_and_sends_it_back():
    wire = Wire(
        sse(chunk({"reasoning_content": "think"}), chunk({"content": "hi"}), STOPPED),
        sse(chunk({"content": "again"}), STOPPED),
    )
    provider = MiniMaxProvider(model="MiniMax-M3.1-Flash", api_key="k", transport=wire.transport)
    first = final([event async for event in provider.stream(ModelRequest(HELLO))])

    assert str(wire.requests[0].url) == "https://api.minimax.io/v1/chat/completions"
    assert first.message.reasoning == "think" and first.message.provider == "minimax"
    await run(provider, *HELLO, first.message, UserMessage(content="more"))
    assistant = [m for m in wire.sent()["messages"] if m["role"] == "assistant"][0]
    assert assistant["reasoning_content"] == "think"


@pytest.mark.parametrize(
    ("model", "effort", "kwargs", "expected"),
    [
        (
            "MiniMax-M3.1-Flash",
            E.XHIGH,
            {},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
        ),
        ("MiniMax-M3.1-Flash", E.OFF, {}, {}),
        ("MiniMax-M3", E.HIGH, {}, {"thinking": {"type": "adaptive"}}),
        ("MiniMax-M3", E.OFF, {}, {"thinking": {"type": "disabled"}}),
        ("MiniMax-M3", E.HIGH, {"tool_choice": "extract"}, {"thinking": {"type": "disabled"}}),
        ("MiniMax-M3.1-Flash", E.HIGH, {"tool_choice": "extract"}, {}),
        ("MiniMax-M2.7", E.HIGH, {}, {}),
    ],
)
def test_minimax_messages_sends_an_effort_or_a_switch_and_never_a_budget(model, effort, kwargs, expected):
    provider = MiniMaxAnthropicProvider(model=model, api_key="k")
    request = ModelRequest(HELLO, reasoning_effort=effort, **kwargs)
    assert provider.request_fields(request) == expected


async def test_minimax_messages_uses_its_endpoint_and_does_not_raise_max_tokens():
    wire = Wire(answer("hi"))
    provider = MiniMaxAnthropicProvider(model="MiniMax-M3.1-Flash", api_key="k", transport=wire.transport)
    try:
        events = [event async for event in provider.stream(ModelRequest(HELLO))]
    finally:
        await provider.aclose()

    assert str(wire.requests[0].url) == "https://api.minimax.io/anthropic/v1/messages"
    body = wire.sent()
    assert body["max_tokens"] == provider.max_tokens and "budget_tokens" not in json.dumps(body)
    assert body["output_config"] == {"effort": "medium"}
    assert final(events).message.provider == "minimax-anthropic"


# --------------------------------------------------------------------------------------- Mistral


async def test_mistral_sends_no_field_its_schema_does_not_list():
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    response = final(await run(MistralProvider(model="mistral-medium-latest", api_key="k", transport=wire.transport)))

    body = wire.sent()
    assert str(wire.requests[0].url) == "https://api.mistral.ai/v1/chat/completions"
    assert "stream_options" not in body and "reasoning_effort" not in body
    assert response.message.provider == "mistral"


async def test_mistral_takes_a_level_from_extra_body():
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    provider = MistralProvider(
        model="mistral-medium-latest",
        api_key="k",
        transport=wire.transport,
        extra_body={"reasoning_effort": "high"},
    )
    await run(provider)
    assert wire.sent()["reasoning_effort"] == "high"


# ----------------------------------------------------------------------------- Groq and Cerebras


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("openai/gpt-oss-120b", E.OFF, {"reasoning_effort": "low"}),
        ("openai/gpt-oss-20b", E.MEDIUM, {"reasoning_effort": "medium"}),
        ("openai/gpt-oss-20b", E.MAX, {"reasoning_effort": "high"}),
        ("llama-3.3-70b-versatile", E.HIGH, {}),
    ],
)
def test_groq_gives_gpt_oss_a_level_and_other_models_nothing(model, effort, expected):
    assert fields(GroqProvider(model=model, api_key="k"), effort) == expected


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-oss-120b", E.OFF, {"reasoning_effort": "low"}),  # cannot be switched off
        ("gpt-oss-120b", E.HIGH, {"reasoning_effort": "high"}),
        ("qwen-3.8-27b", E.OFF, {"reasoning_effort": "none"}),
        ("qwen-3.8-27b", E.MEDIUM, {"reasoning_effort": "medium"}),
        ("gemma-4-31b", E.OFF, {"reasoning_effort": "none"}),
        ("kimi-k2.7-code", E.HIGH, {}),
        ("ep-dedicated-123", E.HIGH, {}),
    ],
)
def test_cerebras_gives_each_family_the_levels_it_names(model, effort, expected):
    assert fields(CerebrasProvider(model=model, api_key="k"), effort) == expected


@pytest.mark.parametrize(
    ("provider_class", "url", "key"),
    [
        (GroqProvider, "https://api.groq.com/openai/v1/chat/completions", "reasoning"),
        (CerebrasProvider, "https://api.cerebras.ai/v1/chat/completions", "reasoning"),
    ],
)
async def test_groq_and_cerebras_return_reasoning_under_the_name_they_read_it_by(provider_class, url, key):
    wire = Wire(
        sse(chunk({"reasoning": "think"}), chunk({"content": "hi"}), STOPPED),
        sse(chunk({"content": "again"}), STOPPED),
    )
    provider = provider_class(model="gpt-oss-120b", api_key="k", transport=wire.transport)
    first = final([event async for event in provider.stream(ModelRequest(HELLO))])

    assert str(wire.requests[0].url) == url
    assert first.message.reasoning == "think"
    await run(provider, *HELLO, first.message, UserMessage(content="more"))
    assistant = [m for m in wire.sent()["messages"] if m["role"] == "assistant"][0]
    assert assistant[key] == "think" and "reasoning_content" not in assistant


async def test_groq_does_not_send_a_cache_key_unless_asked_but_cerebras_does():
    for provider_class, expected in ((GroqProvider, False), (CerebrasProvider, True)):
        wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
        await run(provider_class(model="gpt-oss-120b", api_key="k", transport=wire.transport), cache_key="run-1")
        assert ("prompt_cache_key" in wire.sent()) is expected


async def test_a_replayed_message_from_another_provider_is_not_given_foreign_reasoning():
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    foreign = AssistantMessage(content="x", reasoning="secret", provider="deepseek")
    await run(GroqProvider(model="gpt-oss-120b", api_key="k", transport=wire.transport), *HELLO, foreign)
    assistant = [m for m in wire.sent()["messages"] if m["role"] == "assistant"][0]
    assert "reasoning" not in assistant


# ------------------------------------------------------------------------------------ the hosts


@pytest.mark.parametrize(
    ("provider_class", "name", "url", "caches"),
    [
        (TogetherProvider, "together", "https://api.together.xyz/v1/chat/completions", False),
        (FireworksProvider, "fireworks", "https://api.fireworks.ai/inference/v1/chat/completions", True),
        (SiliconFlowProvider, "siliconflow", "https://api.siliconflow.cn/v1/chat/completions", False),
        (DeepInfraProvider, "deepinfra", "https://api.deepinfra.com/v1/openai/chat/completions", False),
    ],
)
async def test_a_plain_host_sends_no_reasoning_field_and_only_what_it_documents(provider_class, name, url, caches):
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    response = final(await run(provider_class(model="m", api_key="k", transport=wire.transport), cache_key="run-1"))

    body = wire.sent()
    assert str(wire.requests[0].url) == url
    assert "reasoning_effort" not in body and ("prompt_cache_key" in body) is caches
    assert response.message.provider == name


async def test_a_plain_host_takes_a_model_specific_option_from_extra_body_and_a_custom_root():
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    provider = SiliconFlowProvider(
        model="m",
        api_key="k",
        base_url="https://api.siliconflow.com/v1",
        transport=wire.transport,
        extra_body={"enable_thinking": True},
    )
    await run(provider)
    assert str(wire.requests[0].url) == "https://api.siliconflow.com/v1/chat/completions"
    assert wire.sent()["enable_thinking"] is True


def test_every_new_provider_rejects_a_missing_key():
    for provider_class in (
        XAIProvider,
        MiniMaxProvider,
        MistralProvider,
        GroqProvider,
        CerebrasProvider,
        TogetherProvider,
    ):
        with pytest.raises(ValueError, match="api_key is required"):
            provider_class(model="m", api_key="")


async def test_an_auth_failure_is_reported_as_one_by_a_new_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = XAIProvider(model="grok-4.7", api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderAuthError):
        await run(provider)


async def test_deepseek_messages_declares_a_web_search_the_endpoint_runs_itself():
    from zett_agent.extensions.server_tools import DeepSeekMessagesServerToolExtension
    from zett_agent.model import ServerToolDefinition
    from zett_agent.providers import DeepSeekAnthropicProvider

    default = DeepSeekMessagesServerToolExtension().server_tools
    capped = DeepSeekMessagesServerToolExtension([ServerToolDefinition("web_search_20250305", {"max_uses": 3})])
    wire = Wire(answer("hi"))
    provider = DeepSeekAnthropicProvider("deepseek-flash", "k", wire.transport)
    await run(provider, server_tools=default)
    assert wire.sent()["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]
    assert capped.server_tools[0].configuration == {"max_uses": 3, "name": "web_search"}
