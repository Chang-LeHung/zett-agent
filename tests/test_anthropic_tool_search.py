"""Client-side tool search on the Anthropic Messages API.

Anthropic declares deferred tools with ``defer_loading`` and loads one when a
tool result refers to it with a ``tool_reference`` block. The extension ranks
tools locally with BM25; the provider only translates the protocol.
"""

from __future__ import annotations

import json
from typing import Any, get_type_hints

import httpx
import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.tool_guidelines import ToolGuidancePlacement, ToolGuidelinesExtension
from zett_agent.extensions.tool_search import ToolSearchExtension
from zett_agent.messages import AssistantMessage, ToolMessage
from zett_agent.model import (
    AgentModel,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    NativeDeferredToolsModel,
    RetryOptions,
    ToolDefinition,
)
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.anthropic_compatible import messages_tools, search_result_blocks, to_content_blocks
from zett_agent.providers.base import ProviderResponseError
from zett_agent.providers.deepseek import DeepSeekProvider
from zett_agent.providers.google import GoogleProvider
from zett_agent.providers.ollama import OllamaProvider
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.tools.base import tool


@tool(deferred=True, guidelines="Use for warehouse stock questions.")
async def warehouse_stock(sku: str) -> str:
    """Look up the stock level of one SKU in the warehouse.

    Args:
        sku: Stock keeping unit.
    """
    return f"{sku}: 7 units"


@tool(deferred=True, guidelines="Use for billing questions.")
async def billing_lookup(invoice: str) -> str:
    """Look up one invoice.

    Args:
        invoice: Invoice number.
    """
    return f"{invoice}: paid"


@tool(guidelines="Use to repeat text.")
async def echo(text: str) -> str:
    """Echo text back.

    Args:
        text: Text to return.
    """
    return text


def sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> bytes:
    return sse(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(arguments)},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
    )


def text_answer(text: str) -> bytes:
    return sse(
        [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
    )


class FakeAnthropic:
    """Answer each request with the next scripted body and keep every request."""

    def __init__(self, *answers: bytes) -> None:
        self.answers = list(answers)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, content=self.answers.pop(0), headers={"content-type": "text/event-stream"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def search_definition() -> ToolDefinition:
    return ToolDefinition("tool_search", "Find tools", {"type": "object"}, local_tool_search=True)


async def run_search_conversation(extra_extensions=()):
    server = FakeAnthropic(
        tool_use("toolu_1", "tool_search", {"queries": ["warehouse stock"]}),
        tool_use("toolu_2", "warehouse_stock", {"sku": "A1"}),
        text_answer("There are 7 units."),
    )
    provider = AnthropicProvider("claude", "key", transport=server.transport)
    agent = await Agent.create(
        provider,
        config=AgentRunConfig("search"),
        tools=[echo, warehouse_stock, billing_lookup],
        extensions=[ToolSearchExtension(), *extra_extensions],
    )
    try:
        result = await agent.run("How many A1 units do we have?")
    finally:
        await provider.aclose()
    return result, server


async def test_a_search_loads_a_deferred_tool_by_reference_and_the_tool_then_runs():
    result, server = await run_search_conversation()

    assert result.content == "There are 7 units."
    assert len(server.bodies) == 3

    first_tools = {item["name"]: item for item in server.bodies[0]["tools"]}
    assert set(first_tools) == {"echo", "warehouse_stock", "billing_lookup", "tool_search"}
    # Deferred tools stay out of the model's context until referenced.
    assert first_tools["warehouse_stock"]["defer_loading"] is True
    assert first_tools["billing_lookup"]["defer_loading"] is True
    assert "defer_loading" not in first_tools["echo"] and "defer_loading" not in first_tools["tool_search"]
    # The search tool is an ordinary function on this protocol.
    assert "input_schema" in first_tools["tool_search"]
    assert "queries" in first_tools["tool_search"]["input_schema"]["properties"]

    reply = server.bodies[1]["messages"][-1]
    assert reply["role"] == "user"
    block = reply["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_1"
    assert block["content"] == [{"type": "tool_reference", "tool_name": "warehouse_stock"}]

    final = server.bodies[2]["messages"][-1]["content"][0]
    assert final["tool_use_id"] == "toolu_2" and "A1: 7 units" in json.dumps(final["content"])


async def test_deferred_tool_guidelines_travel_with_the_definition():
    _, server = await run_search_conversation()

    declared = {item["name"]: item for item in server.bodies[0]["tools"]}
    assert declared["warehouse_stock"]["description"].endswith("Use for warehouse stock questions.")
    assert declared["warehouse_stock"]["description"].startswith("Look up the stock level of one SKU")
    # A tool that is not deferred keeps its plain description.
    assert declared["echo"]["description"] == "Echo text back."


async def test_the_cache_breakpoint_sits_on_the_last_tool_that_is_loaded_up_front():
    _, server = await run_search_conversation()

    marked = [item["name"] for item in server.bodies[0]["tools"] if "cache_control" in item]
    assert marked == ["tool_search"]


async def test_tool_search_composes_with_description_placement():
    _, server = await run_search_conversation([ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION)])

    declared = {item["name"]: item for item in server.bodies[0]["tools"]}
    assert "Snippet:" not in declared["warehouse_stock"]["description"]  # deferred tools carry guidelines only
    assert declared["warehouse_stock"]["description"].count("Use for warehouse stock questions.") == 1
    assert "Use to discover deferred tools" in declared["tool_search"]["description"]


def test_deferred_and_search_definitions_render_as_anthropic_tools():
    rendered = messages_tools(
        (ToolDefinition("later", "Later", {"type": "object"}, deferred=True), search_definition())
    )

    assert rendered[0]["defer_loading"] is True and "defer_loading" not in rendered[1]
    assert rendered[1]["name"] == "tool_search"
    with pytest.raises(ValueError, match="Only one local tool search"):
        messages_tools((search_definition(), search_definition()))


def test_deferred_tools_are_declared_normally_when_nothing_can_load_them():
    """The JIT harness defers its inspection tools without any search tool present."""
    from zett_agent.model import ServerToolDefinition

    later = ToolDefinition("later", "Later", {"type": "object"}, deferred=True)

    assert "defer_loading" not in messages_tools((later,))[0]
    hosted = (ServerToolDefinition(type="tool_search_tool_bm25_20251119", configuration={"name": "tool_search"}),)
    assert messages_tools((later,), hosted)[0]["defer_loading"] is True
    unrelated = (ServerToolDefinition(type="web_search_20250305", configuration={"name": "web_search"}),)
    assert "defer_loading" not in messages_tools((later,), unrelated)[0]


async def test_a_deferred_tool_without_tool_search_still_reaches_the_model_on_anthropic():
    server = FakeAnthropic(text_answer("ok"))
    provider = AnthropicProvider("claude", "key", transport=server.transport)
    agent = await Agent.create(provider, config=AgentRunConfig("jit"), tools=[echo, warehouse_stock])
    try:
        await agent.run("hi")
    finally:
        await provider.aclose()

    declared = {item["name"]: item for item in server.bodies[0]["tools"]}
    assert set(declared) == {"echo", "warehouse_stock"}
    assert not any("defer_loading" in item for item in declared.values())


def test_cache_marker_skips_deferred_tools_and_tolerates_having_only_deferred_ones():
    body: dict[str, Any] = {"messages": []}
    tools = [{"name": "a"}, {"name": "b", "defer_loading": True}]

    provider = object.__new__(AnthropicProvider)
    provider.prompt_cache = True
    provider.mark_cache(body, tools)
    assert tools[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in tools[1]

    only_deferred = [{"name": "b", "defer_loading": True}]
    provider.mark_cache({"messages": []}, only_deferred)
    assert "cache_control" not in only_deferred[0]


def test_search_results_become_tool_references_in_order():
    message = ToolMessage(
        tool_call_id="t1",
        name="tool_search",
        content=json.dumps([{"name": "b", "type": "function"}, {"name": "a", "type": "function"}]),
    )

    assert search_result_blocks(message) == [
        {"type": "tool_reference", "tool_name": "b"},
        {"type": "tool_reference", "tool_name": "a"},
    ]


def test_an_empty_search_is_reported_as_text_because_empty_references_are_invalid():
    message = ToolMessage(tool_call_id="t1", name="tool_search", content="[]")

    assert search_result_blocks(message) == [{"type": "text", "text": "No matching tools found."}]


@pytest.mark.parametrize("content", ["not json", "{}", "42", '[{"type": "function"}]', '["a"]'])
def test_a_malformed_search_result_is_a_provider_error(content):
    with pytest.raises(ProviderResponseError):
        search_result_blocks(ToolMessage(tool_call_id="t1", name="tool_search", content=content))


def test_only_the_search_tools_own_result_is_converted_and_failures_stay_errors():
    ordinary = ToolMessage(tool_call_id="t1", name="echo", content='[{"name": "x"}]')
    failed = ToolMessage(tool_call_id="t2", name="tool_search", content="boom", success=False)

    assert to_content_blocks(ordinary, search_tool="tool_search")[0]["content"] == '[{"name": "x"}]'
    block = to_content_blocks(failed, search_tool="tool_search")[0]
    assert block["content"] == "boom" and block["is_error"] is True


async def test_a_provider_without_native_deferral_still_hides_deferred_tools():
    seen: list[ModelRequest] = []

    class Plain:
        async def stream(self, request):
            seen.append(request)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    agent = await Agent.create(
        Plain(), config=AgentRunConfig("plain"), tools=[echo, warehouse_stock], extensions=[ToolSearchExtension()]
    )
    await agent.run("hi")

    assert {item.name for item in seen[0].tools} == {"echo", "tool_search"}


def test_a_minimal_adapter_is_an_agent_model_and_does_not_defer_natively():
    """The flag is optional: ``retry`` and ``stream`` are all an adapter needs, and it then answers False."""

    class Minimal:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    assert isinstance(Minimal(), AgentModel)
    assert not isinstance(Minimal(), NativeDeferredToolsModel)
    assert "native_deferred_tools" not in get_type_hints(AgentModel)
    assert get_type_hints(NativeDeferredToolsModel)["native_deferred_tools"] is bool
    assert getattr(Minimal(), "native_deferred_tools", False) is False


def test_every_provider_declares_native_deferral_explicitly():
    """The built-in providers answer the flag with a bool, so each is a NativeDeferredToolsModel."""

    assert get_type_hints(AnthropicProvider)["native_deferred_tools"] is bool
    assert all(
        isinstance(provider(model="m", api_key="k"), NativeDeferredToolsModel)
        for provider in (AnthropicProvider, OpenAIProvider, DeepSeekProvider, GoogleProvider)
    )
    declared = {
        provider.__name__: provider.native_deferred_tools
        for provider in (AnthropicProvider, OpenAIProvider, DeepSeekProvider, GoogleProvider, OllamaProvider)
    }
    assert declared == {
        "AnthropicProvider": True,
        "OpenAIProvider": False,
        "DeepSeekProvider": False,
        "GoogleProvider": False,
        "OllamaProvider": False,
    }


async def test_a_custom_adapter_that_sets_the_flag_gets_deferred_definitions():
    seen: list[ModelRequest] = []

    class Native:
        native_deferred_tools = True

        async def stream(self, request):
            seen.append(request)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    agent = await Agent.create(
        Native(), config=AgentRunConfig("native"), tools=[echo, warehouse_stock], extensions=[ToolSearchExtension()]
    )
    await agent.run("hi")

    declared = {item.name: item for item in seen[0].tools}
    assert set(declared) == {"echo", "warehouse_stock", "tool_search"}
    assert declared["warehouse_stock"].deferred is True
    assert "Use for warehouse stock questions." in declared["warehouse_stock"].description
