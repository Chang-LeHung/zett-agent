"""The OpenAI-compatible base class, its hooks, and the providers built on them.

Every test drives the real SDK through ``httpx.MockTransport``, so what is asserted is the HTTP request the
service would receive and the events the runtime would see. The bodies are written from each service's
documentation, not recorded from a live service: they pin the behavior the hooks promise, and a service that
has changed since is for a recording to show.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from zett_agent.json_types import JsonValue
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from zett_agent.model import ModelEvent, ModelEventType, ModelRequest, ReasoningEffort, RetryOptions
from zett_agent.providers.base import ProviderAuthError
from zett_agent.providers.glm import GLMProvider, glm_version
from zett_agent.providers.kimi import KimiProvider
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.providers.openai_compatible import ChatDelta, OpenAICompatibleProvider, chat_message
from zett_agent.providers.openrouter import OpenRouterProvider
from zett_agent.providers.wire import JsonObject
from zett_agent.stop import StopReason

NO_RETRY = RetryOptions(max_retries=0)
HELLO = (UserMessage(content="Hello"),)


def sse(*events: Mapping[str, Any], comments: bool = False) -> bytes:
    """An SSE body of ``events``, optionally with keep-alive comment lines between them."""
    lines = []
    for event in events:
        if comments:
            lines.append(": OPENROUTER PROCESSING\n\n")
        lines.append(f"data: {json.dumps(event)}\n\n")
    return "".join(lines).encode() + b"data: [DONE]\n\n"


def chunk(delta: Mapping[str, Any] | None = None, finish: str | None = None, **extra: Any) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": dict(delta or {}), "finish_reason": finish}
    choice.update({key: value for key, value in extra.items() if key == "native_finish_reason"})
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [choice],
        **{key: value for key, value in extra.items() if key != "native_finish_reason"},
    }


class Wire:
    """A transport that replays ``body`` and keeps every request it receives."""

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


async def run(provider: OpenAICompatibleProvider, *messages: Any, **request: Any) -> list[ModelEvent]:
    try:
        return [event async for event in provider.stream(ModelRequest(messages or HELLO, **request))]
    finally:
        await provider.aclose()


def final(events: list[ModelEvent]):
    assert events[-1].type is ModelEventType.RESPONSE and events[-1].response is not None
    return events[-1].response


STOPPED = chunk(finish="stop")


# ------------------------------------------------------------------ the base class and its hooks


async def test_a_compatible_endpoint_works_with_nothing_but_a_base_url():
    wire = Wire(sse(chunk({"content": "hi"}), STOPPED))
    provider = OpenAICompatibleProvider(
        model="local", api_key="k", base_url="https://gateway.example/v1", transport=wire.transport
    )

    response = final(await run(provider))

    assert str(wire.requests[0].url) == "https://gateway.example/v1/chat/completions"
    assert wire.sent()["reasoning_effort"] == "medium"
    assert response.message.content == "hi" and response.message.provider == "openai"


class Quirky(OpenAICompatibleProvider):
    """A service that overrides every hook once, to show each is called and each result is used."""

    provider_name = "quirky"
    reasoning_keys = ("thought",)

    def vendor_headers(self) -> dict[str, str]:
        return {"X-Quirk": "1", "X-Replaced": "vendor"}

    def request_fields(self, request: ModelRequest) -> JsonObject:
        return {"quirk": request.reasoning_effort.value}

    def replay_reasoning(self, message: AssistantMessage, payload: dict[str, JsonValue]) -> None:
        if message.reasoning:
            payload["thought"] = message.reasoning.upper()

    def read_delta(self, delta: Mapping[str, JsonValue]) -> ChatDelta:
        read = super().read_delta(delta)
        marker = delta.get("marker")
        return ChatDelta(read.text, read.reasoning, read.refusal, read.tool_calls, ({"m": marker},) if marker else ())

    def merge_replay(self, blocks: list[JsonObject], fragment: JsonObject) -> None:
        blocks.insert(0, fragment)

    def read_usage(self, usage: Mapping[str, JsonValue]):
        return super().read_usage({"prompt_tokens": 7, "completion_tokens": 3})

    def chat_stop(self, finish_reason: str | None, **facts: Any):
        return super().chat_stop("length", **facts)


async def test_every_hook_is_called_and_what_it_returns_is_used():
    wire = Wire(
        sse(
            chunk({"thought": "mm", "content": "ok", "marker": 1}),
            chunk({"marker": 2}, finish="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}),
        )
    )
    provider = Quirky(
        model="m",
        api_key="k",
        base_url="https://q.example/v1",
        transport=wire.transport,
        extra_headers={"X-Replaced": "app"},
    )

    response = final(await run(provider))

    request = wire.requests[0]
    assert request.headers["x-quirk"] == "1"
    assert request.headers["x-replaced"] == "app"  # the application's header outranks the vendor's
    assert wire.sent()["quirk"] == "medium" and "reasoning_effort" not in wire.sent()
    assert response.message.reasoning == "mm"  # read through reasoning_keys
    assert response.message.replay_blocks == ({"m": 2}, {"m": 1})  # merge_replay put the later one first
    assert (response.usage.input_tokens, response.usage.output_tokens) == (7, 3)
    assert response.stop.reason is StopReason.MAX_OUTPUT  # chat_stop was asked for "length"


async def test_a_subclass_can_turn_an_http_failure_into_its_own_error():
    class Strict(OpenAICompatibleProvider):
        def status_error(self, error):
            return ProviderAuthError("custom: " + str(error.status_code))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(418, json={"error": {"message": "teapot"}})

    provider = Strict(
        model="m", api_key="k", base_url="https://s.example/v1", transport=httpx.MockTransport(handler), retry=NO_RETRY
    )

    with pytest.raises(ProviderAuthError, match="custom: 418"):
        await run(provider)


def test_the_standard_message_rendering_is_the_one_every_service_starts_from():
    assert chat_message(UserMessage(content="x")) == {"role": "user", "content": "x"}
    call = ToolCall("c1", "add", {"a": 1})
    rendered = chat_message(AssistantMessage(content="", tool_calls=(call,)))
    assert rendered["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}
    ]
    assert chat_message(ToolMessage(tool_call_id="c1", name="add", content="3")) == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "3",
    }


def test_openai_keeps_its_parallel_tool_preference_and_effort_in_the_hook():
    provider = OpenAIProvider("gpt", "k")
    fields = provider.request_fields(ModelRequest(HELLO, parallel_tool_call=False))
    assert fields == {"reasoning_effort": "medium", "parallel_tool_calls": False}
    assert provider.request_fields(ModelRequest(HELLO, reasoning_effort=ReasoningEffort.OFF)) == {
        "parallel_tool_calls": True
    }


# ------------------------------------------------------------------------------------ OpenRouter


def openrouter(wire: Wire, **options: Any) -> OpenRouterProvider:
    return OpenRouterProvider("anthropic/claude-sonnet-4.5", "k", transport=wire.transport, retry=NO_RETRY, **options)


async def test_openrouter_identifies_the_app_and_says_nothing_when_not_asked():
    wire = Wire(sse(STOPPED))
    await run(openrouter(wire, site_url="https://example.com", app_name="My agent"))
    headers = wire.requests[0].headers
    assert headers["http-referer"] == "https://example.com"
    assert headers["x-openrouter-title"] == "My agent"
    assert str(wire.requests[0].url) == "https://openrouter.ai/api/v1/chat/completions"

    quiet = Wire(sse(STOPPED))
    await run(openrouter(quiet))
    assert "http-referer" not in quiet.requests[0].headers and "x-openrouter-title" not in quiet.requests[0].headers


@pytest.mark.parametrize(
    ("effort", "sent"),
    [
        (ReasoningEffort.OFF, "none"),
        (ReasoningEffort.MINIMAL, "minimal"),
        (ReasoningEffort.MEDIUM, "medium"),
        (ReasoningEffort.XHIGH, "xhigh"),
        (ReasoningEffort.MAX, "max"),
        (ReasoningEffort.ULTRA, "max"),
    ],
)
async def test_openrouter_sends_reasoning_as_an_object_and_never_the_openai_field(effort, sent):
    wire = Wire(sse(STOPPED))
    await run(openrouter(wire), reasoning_effort=effort, cache_key="chat-1", parallel_tool_call=False)

    body = wire.sent()
    assert body["reasoning"] == {"effort": sent}
    assert "reasoning_effort" not in body
    assert body["parallel_tool_calls"] is False
    assert body["prompt_cache_key"] == "chat-1"  # OpenRouter uses it as the sticky routing key


async def test_openrouter_reads_reasoning_from_its_own_field_and_ignores_keepalive_comments():
    wire = Wire(
        sse(
            chunk({"reasoning": "Let me "}),
            chunk({"reasoning": "think."}),
            chunk({"content": "Done"}),
            chunk(finish="stop", native_finish_reason="end_turn"),
            comments=True,
        )
    )

    events = await run(openrouter(wire))

    assert [event.delta for event in events if event.type is ModelEventType.REASONING_DELTA] == ["Let me ", "think."]
    response = final(events)
    assert response.message.reasoning == "Let me think." and response.message.content == "Done"
    assert response.stop.detail == {"native_finish_reason": "end_turn"}  # a name the runtime does not know is kept


DETAIL_STREAM = sse(
    chunk(
        {"reasoning": "Plan.", "reasoning_details": [{"type": "reasoning.text", "text": "Pla", "index": 0, "id": "r1"}]}
    ),
    chunk(
        {
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "n.",
                    "index": 0,
                    "signature": "sig-1",
                    "format": "anthropic-claude-v1",
                }
            ]
        }
    ),
    chunk({"reasoning_details": [{"type": "reasoning.encrypted", "data": "AAA", "index": 1}]}),
    chunk({"reasoning_details": [{"type": "reasoning.encrypted", "data": "BBB"}]}),
    chunk(
        {
            "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a":1}'}}
            ]
        },
        finish="tool_calls",
    ),
)
MERGED = (
    {
        "type": "reasoning.text",
        "text": "Plan.",
        "index": 0,
        "id": "r1",
        "signature": "sig-1",
        "format": "anthropic-claude-v1",
    },
    {"type": "reasoning.encrypted", "data": "AAA", "index": 1},
    {"type": "reasoning.encrypted", "data": "BBB"},
)


async def test_openrouter_joins_the_pieces_of_a_reasoning_detail_and_sends_them_back_unchanged():
    wire = Wire(DETAIL_STREAM, sse(chunk({"content": "3"}), STOPPED))
    provider = openrouter(wire)

    first = final([event async for event in provider.stream(ModelRequest(HELLO))])
    assert first.message.replay_blocks == MERGED
    assert first.message.tool_calls[0].name == "add"

    history = (*HELLO, first.message, ToolMessage(tool_call_id="call_1", name="add", content="2"))
    await run(provider, *history)

    assistant = wire.sent()["messages"][1]
    assert assistant["reasoning_details"] == list(MERGED)
    assert "reasoning" not in assistant and "reasoning_content" not in assistant


async def test_openrouter_sends_plain_reasoning_text_when_there_are_no_details_and_skips_other_providers():
    wire = Wire(sse(chunk({"content": "ok"}), STOPPED))
    own = AssistantMessage(content="a", reasoning="because", provider="openrouter", model="m")
    foreign = AssistantMessage(content="b", reasoning="not ours", provider="deepseek", model="m")

    await run(openrouter(wire), *HELLO, own, UserMessage(content="more"), foreign, UserMessage(content="again"))

    messages = wire.sent()["messages"]
    assert messages[1]["reasoning"] == "because"
    assert "reasoning" not in messages[3] and "reasoning_content" not in messages[3]


async def test_openrouter_reports_its_usage_including_cache_writes_and_reasoning():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 10},
        "completion_tokens_details": {"reasoning_tokens": 25},
        "cost": 0.0012,
    }
    wire = Wire(sse(chunk({"content": "x"}), chunk(finish="stop"), chunk(finish="stop", usage=usage)))

    response = final(await run(openrouter(wire)))

    assert (response.usage.input_tokens, response.usage.output_tokens) == (100, 40)
    assert (response.usage.cache_read_tokens, response.usage.cache_write_tokens) == (60, 10)
    assert response.usage.reasoning_tokens == 25


async def test_openrouter_error_in_the_middle_of_a_stream_keeps_what_was_written():
    error_chunk = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "error": {"code": 502, "message": "Upstream went away"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    wire = Wire(sse(chunk({"content": "partial"}), error_chunk))

    response = final(await run(openrouter(wire)))

    assert response.message.content == "partial"
    assert response.stop.reason is StopReason.INTERRUPTED
    assert response.stop.detail["error"]["message"] == "Upstream went away"


# ------------------------------------------------------------------------------------------- Kimi


def kimi(wire: Wire, model: str, **options: Any) -> KimiProvider:
    return KimiProvider(model, "k", transport=wire.transport, retry=NO_RETRY, **options)


@pytest.mark.parametrize(
    ("effort", "sent"),
    [
        (ReasoningEffort.MINIMAL, "low"),
        (ReasoningEffort.LOW, "low"),
        (ReasoningEffort.MEDIUM, "high"),
        (ReasoningEffort.HIGH, "high"),
        (ReasoningEffort.XHIGH, "max"),
        (ReasoningEffort.ULTRA, "max"),
    ],
)
async def test_kimi_k3_takes_the_three_level_effort_and_has_no_thinking_switch(effort, sent):
    wire = Wire(sse(STOPPED))
    await run(kimi(wire, "kimi-k3"), reasoning_effort=effort)
    body = wire.sent()
    assert body["reasoning_effort"] == sent and "thinking" not in body
    assert str(wire.requests[0].url) == "https://api.moonshot.ai/v1/chat/completions"


@pytest.mark.parametrize(
    ("model", "effort", "fields"),
    [
        ("kimi-k3", ReasoningEffort.OFF, {}),  # always reasons: nothing to turn off
        ("kimi-k2.6", ReasoningEffort.MEDIUM, {"thinking": {"type": "enabled", "keep": "all"}}),
        ("kimi-k2.6", ReasoningEffort.OFF, {"thinking": {"type": "disabled"}}),
        ("kimi-k2.7-code", ReasoningEffort.HIGH, {"thinking": {"type": "enabled", "keep": "all"}}),
        ("kimi-k2.7-code", ReasoningEffort.OFF, {}),  # always on: a disable would be an error
        ("moonshot-v1-8k", ReasoningEffort.HIGH, {}),  # a model that documents no field gets none
    ],
)
def test_kimi_spells_reasoning_the_way_each_model_takes_it(model, effort, fields):
    assert KimiProvider(model, "k").request_fields(ModelRequest(HELLO, reasoning_effort=effort)) == fields


async def test_kimi_streams_reasoning_content_and_sends_it_back():
    wire = Wire(
        sse(chunk({"reasoning_content": "hm"}), chunk({"content": "ok"}), chunk(finish="stop")),
        sse(chunk({"content": "more"}), STOPPED),
    )
    provider = kimi(wire, "kimi-k2.6")

    first = final([event async for event in provider.stream(ModelRequest(HELLO))])
    assert first.message.reasoning == "hm" and first.message.provider == "kimi"

    await run(provider, *HELLO, first.message, UserMessage(content="more"))
    assert wire.sent()["messages"][1]["reasoning_content"] == "hm"


async def test_kimi_content_filter_is_a_refusal_not_a_crash():
    body = {
        "error": {"type": "content_filter", "message": "The request was rejected because it was considered high risk"}
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body)

    provider = KimiProvider("kimi-k3", "k", transport=httpx.MockTransport(handler), retry=NO_RETRY)

    response = final(await run(provider))

    assert response.stop.reason is StopReason.REFUSAL


# -------------------------------------------------------------------------------------------- GLM


def glm(wire: Wire, model: str, **options: Any) -> GLMProvider:
    return GLMProvider(model, "k", transport=wire.transport, retry=NO_RETRY, **options)


@pytest.mark.parametrize(
    ("name", "version"),
    [("glm-4.6", (4, 6)), ("GLM-5", (5, 0)), ("glm-5.3-flash", (5, 3)), ("glm-4.5v", (4, 5)), ("chatglm", None)],
)
def test_glm_reads_the_version_from_a_model_name(name, version):
    assert glm_version(name) == version


@pytest.mark.parametrize(
    ("model", "effort", "fields"),
    [
        ("glm-4.5", ReasoningEffort.HIGH, {}),  # before the thinking parameter existed
        ("glm-4.6", ReasoningEffort.HIGH, {"thinking": {"type": "enabled", "clear_thinking": False}}),
        ("glm-4.6", ReasoningEffort.OFF, {"thinking": {"type": "disabled"}}),
        (
            "glm-5.2",
            ReasoningEffort.MINIMAL,
            {"thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "low"},
        ),
        (
            "glm-5.2",
            ReasoningEffort.MEDIUM,
            {"thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "high"},
        ),
        (
            "glm-5.3",
            ReasoningEffort.MAX,
            {"thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "max"},
        ),
        ("glm-5.3", ReasoningEffort.OFF, {}),  # only ever thinks
    ],
)
def test_glm_sends_thinking_and_effort_only_to_the_models_that_take_them(model, effort, fields):
    assert GLMProvider(model, "k").request_fields(ModelRequest(HELLO, reasoning_effort=effort)) == fields


def test_glm_options_change_what_is_sent():
    request = ModelRequest(HELLO)
    fields = GLMProvider("glm-4.7", "k", clear_thinking=True, tool_stream=True).request_fields(request)
    assert fields == {"tool_stream": True, "thinking": {"type": "enabled", "clear_thinking": True}}
    assert "tool_stream" not in GLMProvider("glm-4.7", "k").request_fields(request)


async def test_glm_streams_reasoning_content_and_sends_it_back_for_preserved_thinking():
    wire = Wire(
        sse(chunk({"reasoning_content": "step"}), chunk({"content": "ok"}), chunk(finish="stop")),
        sse(chunk({"content": "more"}), STOPPED),
    )
    provider = glm(wire, "glm-5.1")

    first = final([event async for event in provider.stream(ModelRequest(HELLO))])
    assert first.message.reasoning == "step" and first.message.provider == "glm"

    await run(provider, *HELLO, first.message, UserMessage(content="more"))
    body = wire.sent()
    assert body["messages"][1]["reasoning_content"] == "step"
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert str(wire.requests[0].url) == "https://api.z.ai/api/paas/v4/chat/completions"


async def test_glm_streams_tool_call_arguments_in_pieces_when_tool_stream_is_on():
    pieces = [
        chunk(
            {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "add", "arguments": ""}}]}
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, finish="tool_calls"),
    ]
    wire = Wire(sse(*pieces))

    events = await run(glm(wire, "glm-4.7", tool_stream=True))

    assert wire.sent()["tool_stream"] is True
    pieces = [event.tool_call_delta.arguments_delta for event in events if event.tool_call_delta is not None]
    assert pieces == ["", '{"a":', "1}"]
    assert final(events).message.tool_calls[0].arguments == {"a": 1}
