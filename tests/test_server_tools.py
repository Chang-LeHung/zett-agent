"""Provider request tests for tools executed outside the local Agent loop."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from zett_agent import (
    Agent,
    AgentEventType,
    AgentProtocolError,
    AgentRunConfig,
    AssistantMessage,
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelResponse,
    ServerToolCall,
    ServerToolDefinition,
    ServerToolInputDelta,
    ServerToolResult,
    ToolDefinition,
    UserMessage,
    tool,
)
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


def test_server_tool_call_distinguishes_streaming_input_from_complete_empty_input() -> None:
    pending = ServerToolCall("hosted-1", "web_fetch")
    complete_empty = ServerToolCall("hosted-2", "web_search", {})

    assert pending.input is None
    assert complete_empty.input == {}


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


async def test_anthropic_provider_streams_server_tool_lifecycle_without_local_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_anthropic_sse(
                [
                    {
                        "type": "message_start",
                        "message": {
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                        },
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_fetch",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"url":"https://example.com"}',
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "web_fetch_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": {
                                "type": "web_fetch_result",
                                "url": "https://example.com",
                                "content": {
                                    "type": "document",
                                    "source": {"type": "text", "media_type": "text/plain", "data": "Example"},
                                },
                            },
                        },
                    },
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
                    {"type": "message_stop"},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    provider = AnthropicProvider("claude", "key", transport=httpx.MockTransport(handler))
    try:
        events = [event async for event in provider.stream(ModelRequest(messages=(UserMessage(content="Fetch"),)))]
    finally:
        await provider.aclose()

    assert [event.type for event in events] == [
        ModelEventType.SERVER_TOOL_STARTED,
        ModelEventType.SERVER_TOOL_INPUT_DELTA,
        ModelEventType.SERVER_TOOL_COMPLETED,
        ModelEventType.RESPONSE,
    ]
    assert events[0].server_tool_call == ServerToolCall("srvtoolu_1", "web_fetch")
    assert events[1].server_tool_input_delta == ServerToolInputDelta("srvtoolu_1", '{"url":"https://example.com"}')
    assert events[2].server_tool_result == ServerToolResult(
        "srvtoolu_1",
        "web_fetch",
        {
            "type": "web_fetch_result",
            "url": "https://example.com",
            "content": {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "Example"},
            },
        },
    )
    assert events[-1].response is not None
    assert events[-1].response.message.tool_calls == ()


async def test_anthropic_provider_exposes_server_tool_result_error_as_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_anthropic_sse(
                [
                    {"type": "message_start", "message": {"usage": {}}},
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_failed",
                            "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
                    {"type": "message_stop"},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    provider = AnthropicProvider("claude", "key", transport=httpx.MockTransport(handler))
    try:
        events = [event async for event in provider.stream(ModelRequest(messages=()))]
    finally:
        await provider.aclose()

    failed = next(event for event in events if event.type == ModelEventType.SERVER_TOOL_FAILED)
    assert failed.server_tool_result == ServerToolResult(
        "srvtoolu_failed",
        "web_search",
        {"type": "web_search_tool_result_error", "error_code": "unavailable"},
        "unavailable",
    )


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


def test_google_provider_maps_completed_hosted_tool_metadata_once() -> None:
    from google.genai import types

    from zett_agent.providers.google import _google_server_tool_events

    candidate = types.Candidate(
        grounding_metadata=types.GroundingMetadata(web_search_queries=["zettelkasten"]),
        url_context_metadata=types.UrlContextMetadata(
            url_metadata=[
                types.UrlMetadata(
                    retrieved_url="https://example.com",
                    url_retrieval_status=types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                )
            ]
        ),
    )
    seen: set[str] = set()

    events, sequence = _google_server_tool_events(candidate, sequence=0, seen=seen)
    duplicate_events, duplicate_sequence = _google_server_tool_events(candidate, sequence=sequence, seen=seen)

    assert [event.type for event in events] == [
        ModelEventType.SERVER_TOOL_STARTED,
        ModelEventType.SERVER_TOOL_COMPLETED,
        ModelEventType.SERVER_TOOL_STARTED,
        ModelEventType.SERVER_TOOL_COMPLETED,
    ]
    assert events[0].server_tool_call is not None
    assert events[0].server_tool_call.name == "google_search"
    assert events[2].server_tool_call is not None
    assert events[2].server_tool_call.input == {"url": "https://example.com"}
    assert sequence == 2
    assert duplicate_events == []
    assert duplicate_sequence == sequence


async def test_agent_forwards_server_tools_without_running_same_named_local_tool() -> None:
    local_calls = 0

    @tool
    def web_fetch(url: str) -> str:
        """Fetch a URL locally.

        Args:
            url: Absolute URL to fetch.

        Guidelines:
            - Use only for an explicit local fetch.
        """
        nonlocal local_calls
        local_calls += 1
        return url

    class Model:
        async def stream(self, request: ModelRequest):
            call = ServerToolCall("hosted-1", "web_fetch", {"url": "https://example.com"})
            yield ModelEvent.server_tool_started(call)
            yield ModelEvent.server_tool_completed(ServerToolResult(call.id, call.name, {"status": 200}))
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Fetched")))

    agent = await Agent.create(
        Model(),
        config=AgentRunConfig(session_id="server-tool-session"),
        tools=(web_fetch,),
    )
    events = [event async for event in agent.stream("Fetch it")]

    assert [event.type for event in events if event.type.value.startswith("server_tool_")] == [
        AgentEventType.SERVER_TOOL_STARTED,
        AgentEventType.SERVER_TOOL_COMPLETED,
    ]
    assert local_calls == 0
    assert not any(event.type in {AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED} for event in events)


@pytest.mark.parametrize(
    ("model_events", "message"),
    [
        (
            [ModelEvent.server_tool_input(ServerToolInputDelta("hosted-1", "{}"))],
            "input requires its matching start",
        ),
        (
            [ModelEvent.server_tool_completed(ServerToolResult("hosted-1", "web_fetch"))],
            "result requires its matching start",
        ),
        (
            [
                ModelEvent.server_tool_started(ServerToolCall("hosted-1", "web_fetch")),
                ModelEvent.server_tool_started(ServerToolCall("hosted-1", "web_fetch")),
            ],
            "started more than once",
        ),
        (
            [
                ModelEvent.server_tool_started(ServerToolCall("hosted-1", "web_fetch")),
                ModelEvent.completed(ModelResponse(AssistantMessage(content="premature"))),
            ],
            "before server tools completed",
        ),
    ],
)
async def test_agent_rejects_invalid_server_tool_lifecycle(
    model_events: list[ModelEvent],
    message: str,
) -> None:
    class Model:
        async def stream(self, request: ModelRequest):
            for event in model_events:
                yield event
            if not model_events or model_events[-1].type != ModelEventType.RESPONSE:
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    agent = await Agent.create(Model(), config=AgentRunConfig(session_id="invalid-server-tool"))

    with pytest.raises(AgentProtocolError, match=message):
        _ = [event async for event in agent.stream("Fetch it")]


async def test_ollama_provider_rejects_server_tools_without_network_io() -> None:
    provider = OllamaProvider("model")
    request = ModelRequest(messages=(), server_tools=(ServerToolDefinition(type="web_search"),))
    try:
        with pytest.raises(ProviderResponseError, match="does not support provider-hosted"):
            _ = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()
