"""Responses API coverage shared by the OpenAI and DeepSeek adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai.types.responses import FunctionToolParam

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageDetail,
    ImageUrlSource,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from zett_agent.model import (
    ModelEventType,
    ModelRequest,
    ReasoningEffort,
    ServerToolDefinition,
    ToolDefinition,
)
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.deepseek import DeepSeekProvider
from zett_agent.providers.google import GoogleProvider
from zett_agent.providers.ollama import OllamaProvider
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.providers.responses import (
    _object_mapping,
    responses_input,
    responses_reasoning,
    responses_tools,
    stream_responses,
)
from zett_agent.tools.base import tool


def _response(
    output: list[dict[str, Any]],
    *,
    model: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "created_at": 0,
        "model": model,
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": usage,
    }


def _responses_sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def _completed_event(output: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "sequence_number": 99,
        "response": _response(
            output,
            model=model,
            usage={
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        ),
    }


async def _collect(stream):
    return [event async for event in stream]


def _responses_capture(model: str = "gpt-5") -> tuple[dict[str, Any], httpx.MockTransport]:
    """Capture the JSON payload of one Responses API request."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        body = _responses_sse([_completed_event([], model=model)])
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return captured, httpx.MockTransport(handler)


async def test_openai_responses_declares_the_prompt_cache_key() -> None:
    captured, transport = _responses_capture()
    provider = OpenAIProvider("gpt-5", "key", transport=transport, response=True)
    request = ModelRequest(messages=(UserMessage(content="Hello"),), cache_key="chat-42")

    await _collect(provider.stream(request))

    assert captured["body"]["prompt_cache_key"] == "chat-42"


async def test_openai_responses_can_suppress_the_prompt_cache_key() -> None:
    captured, transport = _responses_capture()
    provider = OpenAIProvider("gpt-5", "key", transport=transport, response=True, send_prompt_cache_key=False)
    request = ModelRequest(messages=(UserMessage(content="Hello"),), cache_key="chat-42")

    await _collect(provider.stream(request))

    assert "prompt_cache_key" not in captured["body"]


def test_responses_tools_adds_tool_search_for_deferred_functions() -> None:
    rendered = responses_tools(
        (
            ToolDefinition("immediate", "Always visible", {"type": "object"}),
            ToolDefinition("later", "Deferred", {"type": "object"}, deferred=True),
        ),
        (ServerToolDefinition("web_search"),),
    )

    assert rendered == [
        {
            "type": "function",
            "name": "immediate",
            "description": "Always visible",
            "parameters": {"type": "object"},
            "strict": None,
        },
        {
            "type": "function",
            "name": "later",
            "description": "Deferred",
            "parameters": {"type": "object"},
            "strict": None,
            "defer_loading": True,
        },
        {"type": "web_search"},
        {"type": "tool_search"},
    ]


async def test_openai_responses_streams_local_and_server_tools() -> None:
    captured: dict[str, Any] = {}
    function = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "save",
        "arguments": '{"title":"Result"}',
        "status": "completed",
    }
    search_started = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "in_progress",
        "action": {"type": "search", "queries": ["zettelkasten"]},
    }
    search_done = {**search_started, "status": "completed"}
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Saved", "annotations": [], "logprobs": []}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        events = [
            {"type": "response.output_item.added", "output_index": 0, "sequence_number": 1, "item": search_started},
            {"type": "response.output_item.done", "output_index": 0, "sequence_number": 2, "item": search_done},
            {"type": "response.output_item.added", "output_index": 1, "sequence_number": 3, "item": function},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 1,
                "sequence_number": 4,
                "delta": '{"title":"Result"}',
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 2,
                "content_index": 0,
                "sequence_number": 5,
                "delta": "Saved",
                "logprobs": [],
            },
            _completed_event([search_done, function, message], model="gpt-5"),
        ]
        return httpx.Response(200, content=_responses_sse(events), headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("gpt-5", "key", transport=httpx.MockTransport(handler), response=True)
    request = ModelRequest(
        messages=(SystemMessage(content="Be concise"), UserMessage(content="Search and save")),
        tools=(ToolDefinition("save", "Save result", {"type": "object"}),),
        server_tools=(ServerToolDefinition("web_search", {"search_context_size": "low"}),),
        reasoning_effort=ReasoningEffort.HIGH,
        parallel_tool_call=False,
    )
    try:
        events = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["body"]["store"] is False
    assert captured["body"]["parallel_tool_calls"] is False
    assert captured["body"]["reasoning"] == {"effort": "high", "summary": "auto"}
    assert captured["body"]["tools"] == [
        {
            "type": "function",
            "name": "save",
            "description": "Save result",
            "parameters": {"type": "object"},
            "strict": None,
        },
        {"type": "web_search", "search_context_size": "low"},
    ]
    assert [event.type for event in events] == [
        ModelEventType.SERVER_TOOL_STARTED,
        ModelEventType.SERVER_TOOL_COMPLETED,
        ModelEventType.TOOL_CALL_DELTA,
        ModelEventType.TOOL_CALL_DELTA,
        ModelEventType.TEXT_DELTA,
        ModelEventType.RESPONSE,
    ]
    assert events[0].server_tool_call is not None
    assert events[0].server_tool_call.name == "web_search"
    response = events[-1].response
    assert response is not None
    assert response.message.content == "Saved"
    assert response.message.tool_calls == (ToolCall("call_1", "save", {"title": "Result"}),)
    assert response.message.replay_blocks == (search_done, function, message)
    assert response.usage.input_tokens == 20
    assert response.usage.cache_read_tokens == 5
    assert response.usage.cache_write_tokens == 2
    assert response.usage.reasoning_tokens == 3


async def test_responses_usage_reports_unknown_when_cache_details_are_absent() -> None:
    """A gateway that relays usage without details must not look like a full miss."""

    def handler(request: httpx.Request) -> httpx.Response:
        completed = {
            "type": "response.completed",
            "sequence_number": 99,
            "response": _response(
                [],
                model="deepseek-flash",
                usage={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            ),
        }
        return httpx.Response(200, content=_responses_sse([completed]), headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    try:
        events = await _collect(provider.stream(ModelRequest(messages=(UserMessage(content="Hello"),))))
    finally:
        await provider.aclose()

    usage = events[-1].response.usage
    assert usage.input_tokens == 20
    assert usage.cache_read_tokens == 0
    assert usage.cache_reported is False
    assert usage.cache_hit_rate is None


async def test_deepseek_responses_uses_stateless_endpoint_and_replays_output() -> None:
    captured: dict[str, Any] = {}
    prior = {"id": "reason_1", "type": "reasoning", "content": [], "summary": [], "status": "completed"}
    message = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Done", "annotations": [], "logprobs": []}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event([message], model="deepseek-v4-flash")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = DeepSeekProvider(
        "deepseek-v4-flash",
        "key",
        transport=httpx.MockTransport(handler),
        response=True,
    )
    request = ModelRequest(
        messages=(
            AssistantMessage(content="Earlier", provider="deepseek", model="deepseek-v4-flash", replay_blocks=(prior,)),
            UserMessage(content="Continue"),
        )
    )
    try:
        events = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()

    assert captured["url"] == "https://api.deepseek.com/responses"
    assert captured["body"]["input"] == [prior, {"role": "user", "content": "Continue"}]
    assert events[-1].response is not None
    assert events[-1].response.message.provider == "deepseek"


async def test_responses_history_uses_function_call_output_items() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event([], model="gpt-5")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("gpt-5", "key", transport=httpx.MockTransport(handler), response=True)
    request = ModelRequest(
        messages=(
            AssistantMessage(tool_calls=(ToolCall("call_1", "read", {"path": "README.md"}),)),
            ToolMessage(tool_call_id="call_1", name="read", content="contents"),
        ),
        reasoning_effort=ReasoningEffort.OFF,
    )
    try:
        await anext(provider.stream(request))
    finally:
        await provider.aclose()

    assert "reasoning" not in captured["body"]
    assert captured["body"]["input"] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path": "README.md"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "contents"},
    ]


async def test_responses_can_force_a_registered_server_tool() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event([], model="gpt-5")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("gpt-5", "key", transport=httpx.MockTransport(handler), response=True)
    request = ModelRequest(
        messages=(UserMessage(content="Search"),),
        server_tools=(ServerToolDefinition("web_search"),),
        tool_choice="web_search",
    )
    try:
        events = [event async for event in provider.stream(request)]
    finally:
        await provider.aclose()

    assert events[-1].response is not None
    assert captured["body"]["tool_choice"] == {"type": "web_search"}


async def test_agent_completes_a_responses_function_round_trip() -> None:
    requests: list[dict[str, Any]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: First integer.
            b: Second integer.

        Guidelines:
            - Use for integer addition.
        """
        return a + b

    function = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "add",
        "arguments": '{"a":2,"b":3}',
        "status": "completed",
    }
    answer = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "5", "annotations": [], "logprobs": []}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        output = [function] if len(requests) == 1 else [answer]
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event(output, model="gpt-5")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("gpt-5", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(provider, config=AgentRunConfig("responses-round-trip"), tools=(add,))
    try:
        result = await agent.run("Add 2 and 3")
    finally:
        await provider.aclose()

    assert result.content == "5"
    assert len(requests) == 2
    assert requests[1]["input"][-2:] == [
        function,
        {"type": "function_call_output", "call_id": "call_1", "output": "5"},
    ]


async def test_agent_completes_a_deferred_tool_search_round_trip() -> None:
    calls: list[str] = []

    @tool(deferred=True, guidelines="Use for warehouse stock questions.")
    def inventory(warehouse: str) -> str:
        """Return the stock level for one warehouse.

        Args:
            warehouse: Warehouse code to look up.
        """
        calls.append(warehouse)
        return f"{warehouse}:42"

    search = {
        "id": "ts_1",
        "call_id": "call_search",
        "type": "tool_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["inventory stock"]},
    }
    function = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "inventory",
        "arguments": '{"warehouse":"shanghai"}',
        "status": "completed",
    }
    answer = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Shanghai has 42 in stock.", "annotations": [], "logprobs": []}],
    }
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        output = [search, function] if len(requests) == 1 else [answer]
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event(output, model="deepseek-flash")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(provider, config=AgentRunConfig("deferred-tools"), tools=(inventory,))
    try:
        result = await agent.run("How much stock is in the shanghai warehouse?")
    finally:
        await provider.aclose()

    assert result.content == "Shanghai has 42 in stock."
    assert calls == ["shanghai"]
    advertised = {tool_schema["name"]: tool_schema for tool_schema in requests[0]["tools"] if "name" in tool_schema}
    assert advertised["inventory"]["defer_loading"] is True
    assert {"type": "tool_search"} in requests[0]["tools"]
    assert requests[1]["input"][-2:] == [
        function,
        {"type": "function_call_output", "call_id": "call_1", "output": "shanghai:42"},
    ]


def test_responses_tools_declare_client_side_search() -> None:
    rendered = responses_tools(
        (
            ToolDefinition("find_tools", "Find tools", {"type": "object"}, local_tool_search=True),
            ToolDefinition("later", "Deferred", {"type": "object"}, deferred=True),
        ),
        (),
    )

    assert [tool.get("name") for tool in rendered] == ["later", None]
    assert rendered[0]["defer_loading"] is True
    declared = rendered[-1]
    assert declared["type"] == "tool_search"
    assert declared["execution"] == "client"
    assert declared["parameters"] == {"type": "object"}


def test_responses_tools_reject_mixed_tool_search_and_duplicates() -> None:
    search = ToolDefinition("find_tools", "Find tools", {"type": "object"}, local_tool_search=True)
    with pytest.raises(ValueError, match="provider tool_search"):
        responses_tools((search,), (ServerToolDefinition("tool_search"),))
    with pytest.raises(ValueError, match="Only one local tool search"):
        responses_tools(
            (search, ToolDefinition("other", "Other search", {"type": "object"}, local_tool_search=True)),
            (),
        )


async def test_agent_answers_client_tool_search_locally() -> None:
    searched: list[list[str]] = []
    calls: list[str] = []

    @tool(local_tool_search=True, guidelines="Use to find optional tools.")
    def find_tools(queries: list[str]) -> list[FunctionToolParam]:
        """Return tool definitions matching the requested queries.

        Args:
            queries: What the model wants to do.
        """
        searched.append(list(queries))
        return [
            FunctionToolParam(
                type="function",
                name="warehouse_stock",
                description="Return stock for one warehouse",
                parameters={"type": "object", "properties": {"warehouse": {"type": "string"}}},
                strict=None,
            )
        ]

    @tool(deferred=True, guidelines="Use for stock questions.")
    def warehouse_stock(warehouse: str) -> str:
        """Return stock for one warehouse.

        Args:
            warehouse: Warehouse code to look up.
        """
        calls.append(warehouse)
        return f"{warehouse}:42"

    search = {
        "id": "ts_1",
        "call_id": "call_search",
        "type": "tool_search_call",
        "status": "completed",
        "execution": "client",
        "arguments": {"queries": ["warehouse stock"]},
    }
    function = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "warehouse_stock",
        "arguments": '{"warehouse":"shanghai"}',
        "status": "completed",
    }
    answer = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Shanghai has 42.", "annotations": [], "logprobs": []}],
    }
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        output = [[search], [function], [answer]][len(requests) - 1]
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event(output, model="deepseek-flash")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(
        provider,
        config=AgentRunConfig("client-tool-search"),
        tools=(find_tools, warehouse_stock),
    )
    try:
        result = await agent.run("How much stock is in the shanghai warehouse?")
    finally:
        await provider.aclose()

    assert result.content == "Shanghai has 42."
    assert searched == [["warehouse stock"]]
    assert calls == ["shanghai"]
    declared = requests[0]["tools"][-1]
    assert (declared["type"], declared["execution"]) == ("tool_search", "client")
    assert all(tool.get("name") != "find_tools" for tool in requests[0]["tools"])
    searched = requests[1]["input"][-1]
    assert searched["type"] == "tool_search_output"
    assert searched["call_id"] == "call_search"
    assert [tool["name"] for tool in searched["tools"]] == ["warehouse_stock"]
    assert requests[2]["input"][-2:] == [
        function,
        {"type": "function_call_output", "call_id": "call_1", "output": "shanghai:42"},
    ]


async def test_agent_rejects_invalid_local_tool_search_results() -> None:
    @tool(local_tool_search=True, guidelines="Use to find optional tools.")
    def find_tools(queries: list[str]) -> list[FunctionToolParam]:
        """Return tool definitions matching the requested queries.

        Args:
            queries: What the model wants to do.
        """
        return [{"name": "warehouse_stock", "parameters": {"type": "object"}}]  # type: ignore[list-item]

    search = {
        "id": "ts_1",
        "call_id": "call_search",
        "type": "tool_search_call",
        "status": "completed",
        "execution": "client",
        "arguments": {"queries": ["warehouse stock"]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event([search], model="deepseek-flash")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(provider, config=AgentRunConfig("invalid-search"), tools=(find_tools,))
    try:
        with pytest.raises(ValueError, match="invalid FunctionToolParam"):
            await agent.run("Find tools for the warehouse")
    finally:
        await provider.aclose()


async def test_agent_rejects_a_non_json_local_tool_search_result() -> None:
    @tool(local_tool_search=True, guidelines="Use to find optional tools.")
    def find_tools(queries: list[str]) -> str:
        """Return plain text instead of tool definitions.

        Args:
            queries: What the model wants to do.
        """
        return "no tools"

    search = {
        "id": "ts_1",
        "call_id": "call_search",
        "type": "tool_search_call",
        "status": "completed",
        "execution": "client",
        "arguments": {"queries": ["warehouse stock"]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_responses_sse([_completed_event([search], model="deepseek-flash")]),
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(provider, config=AgentRunConfig("non-json-search"), tools=(find_tools,))
    try:
        with pytest.raises(ValueError, match="must return JSON tool definitions"):
            await agent.run("Find tools for the warehouse")
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: AnthropicProvider("claude", "key", response=True), "AnthropicProvider"),
        (lambda: GoogleProvider("gemini", "key", response=True), "GoogleProvider"),
        (lambda: OllamaProvider("llama", response=True), "OllamaProvider"),
    ],
)
def test_native_non_responses_providers_reject_response_mode(factory: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_response_mode_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="response must be a boolean"):
        OpenAIProvider("gpt-5", "key", response=1)  # type: ignore[arg-type]


def test_responses_input_maps_fallback_history_and_ordered_images() -> None:
    items = responses_input(
        (
            AgentMessage(content="Internal instruction"),
            UserMessage(
                content=[
                    ImageContent(ImageUrlSource("https://example.com/image.png"), ImageDetail.LOW),
                    ImageContent(ImageBytesSource(b"png", "image/png"), ImageDetail.HIGH),
                ]
            ),
            AssistantMessage(content="Result", provider="other", model="other"),
        ),
        provider="openai",
        model="gpt-5",
    )

    assert items[0] == {"role": "user", "content": "Internal instruction"}
    assert items[1]["content"][0] == {
        "type": "input_image",
        "image_url": "https://example.com/image.png",
        "detail": "low",
    }
    assert items[1]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert items[2] == {"role": "assistant", "content": "Result"}
    assert responses_reasoning(ReasoningEffort.XHIGH) == {"effort": "high", "summary": "auto"}


async def _objects(*events: Any):
    for event in events:
        yield event


async def test_responses_terminal_output_is_used_when_no_text_deltas_arrive() -> None:
    output = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "Think"}],
            "summary": [{"type": "summary_text", "text": "Do not duplicate"}],
        },
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Answer"}, {"type": "refusal", "refusal": ""}],
        },
    ]
    terminal = SimpleNamespace(type="response.incomplete", response=_response(output, model="gpt-5"))

    events = [event async for event in stream_responses(_objects(terminal), provider="openai", model="gpt-5")]

    assert events[-1].response is not None
    assert events[-1].response.message.content == "Answer"
    assert events[-1].response.message.reasoning == "Think"


async def test_responses_emits_start_before_an_unannounced_failed_server_tool_result() -> None:
    failed_item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "failed",
        "action": {"type": "search", "query": "news"},
        "error": {"code": "search_unavailable"},
    }
    done = SimpleNamespace(type="response.output_item.done", item=failed_item, output_index=0)
    terminal = SimpleNamespace(type="response.completed", response=_response([failed_item], model="gpt-5"))

    events = [event async for event in stream_responses(_objects(done, terminal), provider="openai", model="gpt-5")]

    assert [event.type for event in events] == [
        ModelEventType.SERVER_TOOL_STARTED,
        ModelEventType.SERVER_TOOL_FAILED,
        ModelEventType.RESPONSE,
    ]
    assert events[1].server_tool_result is not None
    assert events[1].server_tool_result.error_code == "search_unavailable"


async def test_responses_rejects_failed_or_unterminated_streams() -> None:
    failed = SimpleNamespace(
        type="response.failed",
        response={"error": {"message": "provider failed"}},
    )
    with pytest.raises(RuntimeError, match="provider failed"):
        _ = [event async for event in stream_responses(_objects(failed), provider="openai", model="gpt-5")]
    with pytest.raises(RuntimeError, match="without a terminal response"):
        _ = [event async for event in stream_responses(_objects(), provider="openai", model="gpt-5")]


def test_responses_mapping_rejects_unknown_sdk_objects() -> None:
    assert _object_mapping(None) == {}
    with pytest.raises(TypeError, match="did not normalize"):
        _object_mapping(object())
