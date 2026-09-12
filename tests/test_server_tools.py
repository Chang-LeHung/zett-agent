"""Provider request tests for tools executed outside the local Agent loop."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from zett_agent import ModelEventType, ModelRequest, ServerToolDefinition, ToolDefinition, UserMessage
from zett_agent.providers import (
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderResponseError,
)


def _openai_sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"


def _anthropic_sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def test_server_tool_definition_preserves_opaque_configuration() -> None:
    tool = ServerToolDefinition(
        type="openrouter:web_fetch",
        configuration={"parameters": {"max_uses": 3}},
    )

    assert tool.type == "openrouter:web_fetch"
    assert tool.configuration == {"parameters": {"max_uses": 3}}


@pytest.mark.parametrize(
    ("tool_type", "configuration", "message"),
    [
        ("", {}, "cannot be empty"),
        ("web_search", {"type": "other"}, "cannot override type"),
    ],
)
def test_server_tool_definition_rejects_ambiguous_wire_shapes(
    tool_type: str,
    configuration: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ServerToolDefinition(type=tool_type, configuration=configuration)


async def test_openai_compatible_provider_combines_local_and_server_tools() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_openai_sse([{"choices": [{"finish_reason": "stop"}]}]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("model", "key", transport=httpx.MockTransport(handler))
    request = ModelRequest(
        messages=(UserMessage(content="Find the current documentation"),),
        tools=(ToolDefinition("save", "Save a result", {"type": "object"}),),
        server_tools=(
            ServerToolDefinition(
                type="openrouter:web_search",
                configuration={"parameters": {"max_results": 4}},
            ),
        ),
    )
    try:
        events = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()

    assert captured["body"]["tools"] == [
        {
            "type": "function",
            "function": {"name": "save", "description": "Save a result", "parameters": {"type": "object"}},
        },
        {"type": "openrouter:web_search", "parameters": {"max_results": 4}},
    ]
    assert [event.type for event in events] == [ModelEventType.RESPONSE]
    assert events[0].response is not None
    assert events[0].response.message.tool_calls == ()


async def test_anthropic_provider_sends_versioned_server_tool_unchanged() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_anthropic_sse(
                [
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": "claude",
                            "stop_reason": None,
                            "usage": {"input_tokens": 2, "output_tokens": 0},
                        },
                    },
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    provider = AnthropicProvider("claude", "key", transport=httpx.MockTransport(handler))
    request = ModelRequest(
        messages=(UserMessage(content="Search"),),
        server_tools=(
            ServerToolDefinition(
                type="web_search_20260318",
                configuration={"name": "web_search", "max_uses": 2},
            ),
        ),
    )
    try:
        events = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()

    assert captured["body"]["tools"] == [{"type": "web_search_20260318", "name": "web_search", "max_uses": 2}]
    assert events[-1].response is not None
    assert events[-1].response.message.content == "done"


def test_google_provider_maps_native_server_tool_to_sdk_type() -> None:
    from google.genai import types

    request = ModelRequest(
        messages=(),
        server_tools=(ServerToolDefinition(type="google_search"),),
    )

    tools = GoogleProvider._google_tools(types, request)

    assert tools is not None
    assert [tool.model_dump(exclude_none=True) for tool in tools] == [{"google_search": {}}]


def test_google_provider_rejects_unknown_server_tool_before_request() -> None:
    from google.genai import types

    request = ModelRequest(messages=(), server_tools=(ServerToolDefinition(type="not_a_google_tool"),))

    with pytest.raises(ProviderResponseError, match="Unsupported Google server tool type"):
        GoogleProvider._google_tools(types, request)


async def test_ollama_provider_rejects_server_tools_without_network_io() -> None:
    provider = OllamaProvider("model")
    request = ModelRequest(messages=(), server_tools=(ServerToolDefinition(type="web_search"),))
    try:
        with pytest.raises(ProviderResponseError, match="does not support provider-hosted"):
            _ = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()
