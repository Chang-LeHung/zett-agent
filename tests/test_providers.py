"""Tests for provider adapters built on top of official SDKs.

Each test injects an ``httpx.MockTransport`` into the SDK's HTTP client so
the provider makes no real network calls.  The mock handler replays a canned
SSE or NDJSON body and captures the request URL, headers, and JSON payload
for assertions.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from zett_agent import (
    AgentMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
    ModelEventType,
    ModelUsage,
    ReasoningEffort,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    UserMessage,
)
from zett_agent.model import ModelRequest
from zett_agent.providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderAuthError,
    ProviderResponseError,
    _message_to_openai_payload,
    _normalize_image_source,
    _usage_from_mapping,
)


def _sse_body(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode("utf-8") + b"data: [DONE]\n\n"


def _sse_body_no_done(events: list[dict[str, Any]]) -> bytes:
    """SSE body without the OpenAI-style ``[DONE]`` sentinel (Google, etc.)."""
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode("utf-8")


def _anthropic_sse_body(events: list[dict[str, Any]]) -> bytes:
    """Anthropic SSE uses ``event:`` lines alongside ``data:`` payloads."""
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode("utf-8")


def _ndjson_body(objects: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(obj) + "\n" for obj in objects).encode("utf-8")


def _request_with_text_and_tools() -> ModelRequest:
    return ModelRequest(
        messages=(
            SystemMessage(content="Be concise."),
            UserMessage(content="What is 2+2?"),
            AgentMessage(content="Verify the arithmetic."),
        ),
        tools=(
            ToolDefinition(
                name="add",
                description="Add two integers",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
            ),
        ),
        reasoning_effort=ReasoningEffort.MEDIUM,
    )


def test_normalize_image_source_for_url() -> None:
    payload = _normalize_image_source(ImageUrlSource(url="https://example.com/cat.png"))
    assert payload == {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/cat.png",
            "detail": "auto",
        },
    }


def test_normalize_image_source_for_bytes() -> None:
    payload = _normalize_image_source(ImageBytesSource(data=b"\x89PNG", media_type="image/png"))
    assert payload["type"] == "image_url"
    assert payload["image_url"]["url"].startswith("data:image/png;base64")


async def _collect(stream):
    return [event async for event in stream]


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _environment_proxy_urls(client: httpx.AsyncClient) -> dict[str, str]:
    """Inspect HTTPX transports to verify terminal proxy variables were applied."""
    proxies = {}
    for pattern, transport in client._mounts.items():
        pool = getattr(transport, "_pool", None)
        proxy_url = getattr(pool, "_proxy_url", None)
        if proxy_url is not None:
            proxies[pattern.pattern] = bytes(proxy_url).decode("ascii").removesuffix("/")
    return proxies


@pytest.mark.parametrize("provider_name", ["openai", "deepseek", "anthropic", "google", "ollama"])
async def test_provider_uses_terminal_http_and_https_proxy_environment(monkeypatch, provider_name: str) -> None:
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:18080")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:18443")

    match provider_name:
        case "openai":
            provider = OpenAIProvider(model="gpt-4o", api_key="test")
            client = provider._http_client
        case "deepseek":
            provider = DeepSeekProvider(model="deepseek-chat", api_key="test")
            client = provider._http_client
        case "anthropic":
            provider = AnthropicProvider(model="claude", api_key="test")
            client = provider._http_client
        case "google":
            provider = GoogleProvider(model="gemini", api_key="test")
            client = provider._http_client
        case "ollama":
            provider = OllamaProvider(model="llama")
            client = provider._client._client
        case _:
            raise AssertionError(f"Unhandled provider: {provider_name}")

    try:
        assert _environment_proxy_urls(client) == {
            "https://": "http://127.0.0.1:18443",
            "http://": "http://127.0.0.1:18080",
        }
    finally:
        await provider.aclose()


def test_message_to_openai_payload_supports_text_and_image_parts() -> None:
    message = UserMessage(
        content=[TextContent(text="hello"), ImageContent(source=ImageUrlSource(url="https://example.com/a.png"))]
    )
    payload = _message_to_openai_payload(message)
    assert payload == {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "auto"}},
        ],
    }


def test_message_to_openai_payload_supports_image_bytes_parts() -> None:
    message = UserMessage(content=[ImageContent(source=ImageBytesSource(data=b"\x01\x02", media_type="image/jpeg"))])
    payload = _message_to_openai_payload(message)
    assert payload["role"] == "user"
    block = payload["content"][0]
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/jpeg;base64")


def test_usage_from_mapping_normalizes_openai_token_details() -> None:
    usage = _usage_from_mapping(
        {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )

    assert usage.input_tokens == 120
    assert usage.output_tokens == 45
    assert usage.cache_read_tokens == 80
    assert usage.cache_write_tokens == 0
    assert usage.reasoning_tokens == 30
    assert usage.total_tokens == 165
    assert usage.cache_hit_rate == pytest.approx(80 / 120)


def test_model_usage_rejects_negative_counters_and_handles_empty_input() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ModelUsage(input_tokens=-1)
    with pytest.raises(ValueError, match="cache_read_tokens cannot exceed"):
        ModelUsage(input_tokens=1, cache_read_tokens=2)
    with pytest.raises(ValueError, match="cache_write_tokens cannot exceed"):
        ModelUsage(input_tokens=1, cache_write_tokens=2)
    with pytest.raises(ValueError, match="reasoning_tokens cannot exceed"):
        ModelUsage(output_tokens=1, reasoning_tokens=2)
    assert ModelUsage().cache_hit_rate is None


def test_usage_from_mapping_normalizes_deepseek_cache_counters() -> None:
    usage = _usage_from_mapping(
        {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )

    assert usage.input_tokens == 120
    assert usage.output_tokens == 45
    assert usage.cache_read_tokens == 80
    assert usage.cache_write_tokens == 0
    assert usage.reasoning_tokens == 30
    assert usage.total_tokens == 165


# ----------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------


async def test_openai_provider_streams_text_and_usage() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        body = _sse_body(
            [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 3},
                        "completion_tokens_details": {"reasoning_tokens": 1},
                    },
                },
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="sk-test",
        transport=_mock_transport(handler),
    )
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o"
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert captured["body"]["messages"][0] == {"role": "system", "content": "Be concise."}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "What is 2+2?"}
    assert captured["body"]["messages"][2] == {"role": "user", "content": "Verify the arithmetic."}
    assert captured["body"]["tools"][0]["function"]["name"] == "add"
    assert captured["body"]["reasoning_effort"] == "medium"

    deltas = [event.delta for event in events if event.type == ModelEventType.TEXT_DELTA]
    assert deltas == ["Hello", " world"]
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.content == "Hello world"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 2
    assert response.usage.cache_read_tokens == 3
    assert response.usage.reasoning_tokens == 1
    assert response.message.provider == "openai"


async def test_openai_provider_streams_request_with_multimodal_user_content() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_body([{"choices": [{"finish_reason": "stop"}]}]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", transport=_mock_transport(handler))
    request = ModelRequest(
        messages=(
            UserMessage(
                content=[
                    TextContent("Write a caption"),
                    ImageContent(source=ImageUrlSource(url="https://example.com/pic.png")),
                    ImageContent(source=ImageBytesSource(data=b"\x89PNG", media_type="image/png")),
                ]
            ),
        ),
        reasoning_effort=ReasoningEffort.OFF,
    )
    await _collect(provider.stream(request))

    message = captured["body"]["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0] == {"type": "text", "text": "Write a caption"}
    assert message["content"][1]["type"] == "image_url"
    assert message["content"][1]["image_url"]["url"] == "https://example.com/pic.png"
    assert message["content"][2]["type"] == "image_url"
    assert message["content"][2]["image_url"]["url"].startswith("data:image/png;base64")


async def test_openai_provider_streams_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "add", "arguments": '{"a":'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' 2, "b": 2}'}}]}}]},
                {"choices": [{"finish_reason": "tool_calls"}]},
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", transport=_mock_transport(handler))
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    deltas = [event.tool_call_delta for event in events if event.type == ModelEventType.TOOL_CALL_DELTA]
    assert deltas[0].index == 0 and deltas[0].name_delta == "add"
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.tool_calls == (ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 2}),)
    assert response.finish_reason == "tool_calls"


async def test_openai_provider_raises_on_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            content=json.dumps({"error": {"message": "Invalid API key", "type": "invalid_request_error"}}).encode(),
            headers={"content-type": "application/json"},
        )

    provider = OpenAIProvider(model="gpt-4o", api_key="bad", transport=_mock_transport(handler))
    with pytest.raises(ProviderAuthError):
        await _collect(provider.stream(_request_with_text_and_tools()))


async def test_openai_provider_rejects_invalid_tool_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "add", "arguments": "not-json"},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"finish_reason": "tool_calls"}]},
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", transport=_mock_transport(handler))
    with pytest.raises(ProviderResponseError):
        await _collect(provider.stream(_request_with_text_and_tools()))


# ----------------------------------------------------------------------
# DeepSeek
# ----------------------------------------------------------------------


async def test_deepseek_provider_streams_reasoning_content() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        body = _sse_body(
            [
                {"choices": [{"delta": {"reasoning_content": "thinking hard"}}]},
                {"choices": [{"delta": {"content": "4"}}]},
                {"choices": [{"finish_reason": "stop"}]},
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = DeepSeekProvider(model="deepseek-reasoner", api_key="sk-test", transport=_mock_transport(handler))
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    assert captured["body"]["model"] == "deepseek-reasoner"
    assert captured["body"]["messages"][2] == {"role": "user", "content": "Verify the arithmetic."}
    assert "reasoning_effort" not in captured["body"]
    assert captured["body"]["thinking"] == {"type": "enabled"}
    reasoning = [event.delta for event in events if event.type == ModelEventType.REASONING_DELTA]
    assert reasoning == ["thinking hard"]
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.reasoning == "thinking hard"
    assert response.message.content == "4"
    assert response.message.provider == "deepseek"


async def test_deepseek_provider_disables_thinking_for_off_effort() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_body([{"choices": [{"finish_reason": "stop"}]}]),
            headers={"content-type": "text/event-stream"},
        )

    provider = DeepSeekProvider(model="deepseek-chat", api_key="sk-test", transport=_mock_transport(handler))
    request = ModelRequest(
        messages=(UserMessage(content="hi"),),
        reasoning_effort=ReasoningEffort.OFF,
    )
    await _collect(provider.stream(request))
    assert captured["body"]["thinking"] == {"type": "disabled"}


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------


async def test_anthropic_provider_streams_text_and_tool_use() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        body = _anthropic_sse_body(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-3-5-sonnet",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 2,
                            "cache_read_input_tokens": 6,
                            "cache_creation_input_tokens": 2,
                            "output_tokens": 0,
                        },
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me add"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " those."}},
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "add"},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a":'},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ' 2, "b": 2}'},
                },
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 18}},
                {"type": "message_stop"},
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = AnthropicProvider(
        model="claude-3-5-sonnet",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        transport=_mock_transport(handler),
    )
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["system"] == "Be concise."
    assert captured["body"]["messages"][1] == {
        "role": "user",
        "content": [{"type": "text", "text": "Verify the arithmetic."}],
    }
    assert captured["body"]["messages"][0]["role"] == "user"
    assert captured["body"]["tools"][0]["name"] == "add"
    assert "input_schema" in captured["body"]["tools"][0]
    assert captured["body"]["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    text = [event.delta for event in events if event.type == ModelEventType.TEXT_DELTA]
    assert text == ["Let me add", " those."]
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.content == "Let me add those."
    assert response.message.tool_calls == (ToolCall(id="toolu_1", name="add", arguments={"a": 2, "b": 2}),)
    assert response.finish_reason == "tool_use"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 18
    assert response.usage.cache_read_tokens == 6
    assert response.usage.cache_write_tokens == 2
    assert response.message.provider == "anthropic"


async def test_anthropic_provider_streams_thinking_delta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _anthropic_sse_body(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-3-5-sonnet",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "reasoning"},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "answer"}},
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
                {"type": "message_stop"},
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = AnthropicProvider(
        model="claude-3-5-sonnet",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        transport=_mock_transport(handler),
    )
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    reasoning = [event.delta for event in events if event.type == ModelEventType.REASONING_DELTA]
    assert reasoning == ["reasoning"]
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.reasoning == "reasoning"
    assert response.message.content == "answer"


async def test_anthropic_provider_streams_request_with_tool_and_multimodal_blocks() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_anthropic_sse_body(
                [
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": "claude-3-5-sonnet",
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
                    {"type": "message_stop"},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    request = ModelRequest(
        messages=(
            SystemMessage(content="Use tools precisely."),
            UserMessage(
                content=[
                    TextContent("Please add"),
                    ImageContent(source=ImageUrlSource(url="https://example.com/chart.png")),
                ]
            ),
            AssistantMessage(content="", tool_calls=(ToolCall(id="toolu_1", name="add", arguments={"a": 1, "b": 2}),)),
            ToolMessage(content='{"ok":true}', tool_call_id="toolu_1", name="add"),
        ),
        reasoning_effort=ReasoningEffort.OFF,
    )
    provider = AnthropicProvider(model="claude-3-5-sonnet", api_key="sk-ant", transport=_mock_transport(handler))
    await _collect(provider.stream(request))

    user_message = captured["body"]["messages"][0]
    assistant_message = captured["body"]["messages"][1]
    tool_message = captured["body"]["messages"][2]
    assert user_message["role"] == "user"
    assert user_message["content"] == [
        {"type": "text", "text": "Please add"},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/chart.png"}},
    ]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1, "b": 2}},
    ]
    assert tool_message["role"] == "user"
    assert tool_message["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"ok":true}', "is_error": False}
    ]


# ----------------------------------------------------------------------
# Google
# ----------------------------------------------------------------------


async def test_google_provider_streams_text_and_function_call() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        body = _sse_body_no_done(
            [
                {"candidates": [{"content": {"role": "model", "parts": [{"text": "Let me"}]}, "finishReason": None}]},
                {
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": " compute."}]}, "finishReason": None}
                    ]
                },
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"functionCall": {"name": "add", "args": {"a": 2, "b": 2}}}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 4,
                        "thoughtsTokenCount": 12,
                        "cachedContentTokenCount": 3,
                    },
                },
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = GoogleProvider(model="gemini-1.5-pro", api_key="google-key", transport=_mock_transport(handler))
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    assert captured["headers"]["x-goog-api-key"] == "google-key"
    system_instruction = captured["body"]["systemInstruction"]
    assert system_instruction["parts"] == [{"text": "Be concise."}]
    assert captured["body"]["contents"][0]["role"] == "user"
    assert captured["body"]["contents"][0]["parts"] == [{"text": "What is 2+2?"}]
    assert captured["body"]["contents"][1] == {"role": "user", "parts": [{"text": "Verify the arithmetic."}]}
    assert captured["body"]["tools"][0]["functionDeclarations"][0]["name"] == "add"
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {"thinking_budget": 8192, "include_thoughts": True}

    text = [event.delta for event in events if event.type == ModelEventType.TEXT_DELTA]
    assert text == ["Let me", " compute."]
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.content == "Let me compute."
    assert response.message.tool_calls == (ToolCall(id="call_0", name="add", arguments={"a": 2, "b": 2}),)
    assert response.finish_reason == "STOP"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 16
    assert response.usage.reasoning_tokens == 12
    assert response.usage.cache_read_tokens == 3
    assert response.message.provider == "google"


# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------


async def test_ollama_provider_streams_text_and_tool_calls() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        body = _ndjson_body(
            [
                {"message": {"role": "assistant", "content": "Let me add"}, "done": False},
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "add", "arguments": {"a": 2, "b": 2}},
                            }
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 6,
                    "eval_count": 9,
                },
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    provider = OllamaProvider(model="llama3.1", transport=_mock_transport(handler))
    events = await _collect(provider.stream(_request_with_text_and_tools()))

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "llama3.1"
    assert captured["body"]["messages"][2] == {"role": "user", "content": "Verify the arithmetic."}
    assert captured["body"]["messages"][0] == {"role": "system", "content": "Be concise."}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "What is 2+2?"}
    assert captured["body"]["tools"][0]["function"]["name"] == "add"

    text = [event.delta for event in events if event.type == ModelEventType.TEXT_DELTA]
    assert text == ["Let me add"]
    response = [event for event in events if event.type == ModelEventType.RESPONSE][0].response
    assert response.message.content == "Let me add"
    assert response.message.tool_calls == (ToolCall(id="call_0", name="add", arguments={"a": 2, "b": 2}),)
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 6
    assert response.usage.output_tokens == 9
    assert response.usage.cache_read_tokens == 0
    assert response.usage.cache_write_tokens == 0
    assert response.message.provider == "ollama"
