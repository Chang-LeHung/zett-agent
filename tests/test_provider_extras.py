"""Extra headers, query parameters, and body fields on every provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from zett_agent.messages import UserMessage
from zett_agent.model import ModelRequest, ReasoningEffort
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.base import validate_extra_body, validate_extra_headers, validate_extra_query
from zett_agent.providers.deepseek import DeepSeekProvider
from zett_agent.providers.google import GoogleProvider
from zett_agent.providers.ollama import OllamaProvider
from zett_agent.providers.openai import OpenAIProvider

REQUEST = ModelRequest(messages=(UserMessage(content="hi"),), reasoning_effort=ReasoningEffort.OFF)


def sse(events: list[dict[str, Any]], *, done: bool = True, named: bool = False) -> bytes:
    lines = "".join(
        (f"event: {event['type']}\n" if named else "") + f"data: {json.dumps(event)}\n\n" for event in events
    )
    return (lines + ("data: [DONE]\n\n" if done else "")).encode()


class Capture:
    """Record the single request a provider sends and answer with ``body``."""

    def __init__(self, body: bytes, *, content_type: str = "text/event-stream") -> None:
        self.request: httpx.Request | None = None
        self._body = body
        self._type = content_type

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(200, content=self._body, headers={"content-type": self._type})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def json(self) -> dict[str, Any]:
        assert self.request is not None
        return json.loads(self.request.content)


CHAT = sse([{"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"finish_reason": "stop"}]}])
RESPONSES = sse(
    [
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "r",
                "object": "response",
                "created_at": 0,
                "model": "m",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
            },
        }
    ],
    done=False,
)
ANTHROPIC = sse(
    [
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ],
    done=False,
    named=True,
)
GOOGLE = sse(
    [{"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}],
    done=False,
)
OLLAMA = (
    json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True, "done_reason": "stop"}) + "\n"
).encode()


async def drain(provider, request: ModelRequest = REQUEST) -> None:
    try:
        _ = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()


OPTIONS = {
    "extra_headers": {"X-Tenant": "acme", "User-Agent": "gateway-client/1"},
    "extra_query": {"api-version": "2025-01-01"},
}


async def test_openai_chat_sends_headers_query_and_body_fields():
    capture = Capture(CHAT)
    provider = OpenAIProvider("gpt", "key", capture.transport, extra_body={"service_tier": "flex"}, **OPTIONS)

    await drain(provider)

    request = capture.request
    assert request.headers["x-tenant"] == "acme"
    assert request.headers["user-agent"] == "gateway-client/1"
    assert request.url.params["api-version"] == "2025-01-01"
    assert request.headers["authorization"] == "Bearer key"
    assert capture.json["service_tier"] == "flex"


async def test_openai_responses_sends_body_fields_too():
    capture = Capture(RESPONSES)
    provider = OpenAIProvider("gpt", "key", capture.transport, response=True, extra_body={"service_tier": "flex"})

    await drain(provider)

    assert capture.json["service_tier"] == "flex"
    assert capture.json["model"] == "gpt"


async def test_deepseek_body_fields_override_the_adapters_own_extra_fields():
    capture = Capture(CHAT)
    provider = DeepSeekProvider("deepseek-chat", "key", capture.transport, extra_body={"thinking": {"type": "enabled"}})

    await drain(provider)

    # The adapter would send {"type": "disabled"} for reasoning OFF; the caller wins.
    assert capture.json["thinking"] == {"type": "enabled"}


async def test_deepseek_keeps_its_own_fields_when_none_are_overridden():
    capture = Capture(CHAT)

    await drain(DeepSeekProvider("deepseek-chat", "key", capture.transport, extra_headers={"X-A": "1"}))

    assert capture.json["thinking"] == {"type": "disabled"}
    assert capture.request.headers["x-a"] == "1"


async def test_anthropic_sends_headers_query_and_body_fields():
    capture = Capture(ANTHROPIC)
    provider = AnthropicProvider(
        "claude", "key", transport=capture.transport, extra_body={"metadata": {"user_id": "u-1"}}, **OPTIONS
    )

    await drain(provider)

    request = capture.request
    assert request.headers["x-tenant"] == "acme"
    assert request.url.params["api-version"] == "2025-01-01"
    assert request.headers["x-api-key"] == "key"
    assert capture.json["metadata"] == {"user_id": "u-1"}


async def test_anthropic_beta_flags_can_be_sent_as_a_header():
    capture = Capture(ANTHROPIC)

    await drain(AnthropicProvider("claude", "key", transport=capture.transport, extra_headers={"anthropic-beta": "x"}))

    assert capture.request.headers["anthropic-beta"] == "x"


async def test_google_sends_headers_body_fields_and_uses_the_base_url():
    capture = Capture(GOOGLE)
    provider = GoogleProvider(
        "gemini-2.5-flash",
        "key",
        capture.transport,
        base_url="https://gateway.example/proxy",
        extra_headers={"X-Tenant": "acme"},
        extra_body={"labels": {"team": "agents"}},
    )

    await drain(provider)

    assert capture.request.url.host == "gateway.example"
    assert capture.request.url.path.startswith("/proxy/")
    assert capture.request.headers["x-tenant"] == "acme"
    assert capture.request.headers["x-goog-api-key"] == "key"
    assert capture.json["labels"] == {"team": "agents"}


async def test_ollama_sends_headers_and_query_parameters():
    capture = Capture(OLLAMA, content_type="application/x-ndjson")
    provider = OllamaProvider(
        "qwen3",
        capture.transport,
        extra_headers={"Authorization": "Bearer proxy"},
        **{"extra_query": {"tenant": "acme"}},
    )

    await drain(provider)

    assert capture.request.headers["authorization"] == "Bearer proxy"
    assert capture.request.url.params["tenant"] == "acme"


@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: OpenAIProvider("m", "k", **kw),
        lambda **kw: DeepSeekProvider("m", "k", **kw),
        lambda **kw: AnthropicProvider("m", "k", **kw),
    ],
)
def test_bad_options_are_rejected_when_the_provider_is_built(build):
    with pytest.raises(ValueError, match="extra_headers"):
        build(extra_headers={"X": 1})
    with pytest.raises(ValueError, match="extra_headers"):
        build(extra_headers={"": "x"})
    with pytest.raises(ValueError, match="extra_query"):
        build(extra_query={"a": 1})
    with pytest.raises(ValueError, match="JSON"):
        build(extra_body={"a": object()})


def test_validators_copy_their_input_and_treat_none_as_empty():
    headers = {"X": "1"}
    copied = validate_extra_headers(headers)
    headers["Y"] = "2"
    assert copied == {"X": "1"}
    assert validate_extra_headers(None) == validate_extra_query(None) == validate_extra_body(None) == {}
    with pytest.raises(ValueError, match="mapping"):
        validate_extra_headers([("X", "1")])
    with pytest.raises(ValueError, match="mapping"):
        validate_extra_query("a=b")
    with pytest.raises(ValueError, match="mapping"):
        validate_extra_body([])
    with pytest.raises(ValueError, match="names"):
        validate_extra_body({"": 1})
    with pytest.raises(ValueError, match="names"):
        validate_extra_query({"": "1"})


def test_google_and_ollama_validate_their_options_too():
    with pytest.raises(ValueError, match="extra_headers"):
        GoogleProvider("m", "k", extra_headers={"X": 1})
    with pytest.raises(ValueError, match="JSON"):
        GoogleProvider("m", "k", extra_body={"a": object()})
    with pytest.raises(ValueError, match="extra_query"):
        OllamaProvider("m", extra_query={"a": 1})
