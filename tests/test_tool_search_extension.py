"""Client-side tool search answered from the registered tools."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from zett_agent import (
    TOOL_SEARCH_TOOL_NAME,
    Agent,
    AgentExtension,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AgentTool,
    AssistantMessage,
    BM25Search,
    ModelEvent,
    ModelResponse,
    OpenAIProvider,
    SystemMessage,
    ToolGuidelinesExtension,
    ToolSearchExtension,
    tool,
)


@tool(deferred=True, guidelines="Use for stock questions.")
def warehouse_stock(warehouse: str) -> str:
    """Return the stock level for one warehouse.

    Args:
        warehouse: Warehouse code to look up.
    """
    return f"{warehouse}:42"


@tool(deferred=True, guidelines="Use for billing questions.")
def billing_lookup(account: str) -> str:
    """Return the outstanding balance for one account.

    Args:
        account: Account identifier.
    """
    return f"{account}:0"


@tool(guidelines="Use for shipping questions.")
def shipping_quote(destination: str) -> str:
    """Estimate the shipping cost and delivery time for one destination.

    Args:
        destination: Delivery city.
    """
    return f"{destination}:3d"


@tool(guidelines="用于库存查询。")
def stock_query(warehouse: str) -> str:
    """查询指定仓库的库存数量。

    Args:
        warehouse: 仓库代码。
    """
    return f"{warehouse}:7"


class QuietModel:
    """Answer every request without calling tools."""

    async def stream(self, request):
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))


def _sse(output: list[dict[str, Any]]) -> bytes:
    event = {
        "type": "response.completed",
        "sequence_number": 9,
        "response": {
            "id": "resp_1",
            "created_at": 0,
            "model": "deepseek-flash",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    }
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


async def test_extension_registers_the_search_tool_alongside_the_others() -> None:
    observed: list[list[str]] = []

    class Recorder(AgentExtension):
        async def before_run(self, context):
            observed.append(sorted(context.tools))

    agent = await Agent.create(
        QuietModel(),
        config=AgentRunConfig("catalog"),
        tools=(warehouse_stock, billing_lookup),
        extensions=[Recorder(), ToolSearchExtension()],
    )
    await agent.run("hello")

    assert observed == [["billing_lookup", "tool_search", "warehouse_stock"]]


def search_context(*tools: AgentTool) -> AgentRunContext:
    return AgentRunContext(AgentRunConfig("search-test"), AgentState(), {item.name: item for item in tools})


async def test_extension_registers_the_decorated_tool_it_builds(monkeypatch) -> None:
    extension = ToolSearchExtension(min_score=0.7)
    context = search_context(warehouse_stock)
    configured = extension._build_search_tool(context)
    monkeypatch.setattr(extension, "_build_search_tool", lambda context: configured)
    await extension.on_tool(context)
    registered = context.tools[TOOL_SEARCH_TOOL_NAME]

    # Registration copies metadata, but must preserve the decorator's handler.
    assert registered.handler is configured.handler
    assert registered.local_tool_search
    assert "BM25" in registered.description
    assert any("BM25" in guideline for guideline in registered.guidelines)
    assert registered.snippet == 'tool_search(queries=["warehouse stock inventory"])'
    schema = registered.parameters
    assert schema["properties"]["queries"]["type"] == "array"
    assert schema["properties"]["queries"]["items"] == {"type": "string"}
    assert schema["properties"]["score"]["default"] == 0.7
    assert schema["required"] == ["queries"]
    assert schema["additionalProperties"] is False
    definitions = await registered({"queries": ["warehouse stock"]})
    assert [definition["name"] for definition in definitions] == ["warehouse_stock"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"queries": "warehouse"},
        {"queries": [1]},
        {"queries": ["warehouse"], "unexpected": True},
        {"queries": ["warehouse"], "score": "invalid"},
    ],
)
async def test_registered_search_uses_decorator_argument_validation(arguments) -> None:
    from pydantic import ValidationError

    extension = ToolSearchExtension()
    context = search_context(warehouse_stock)
    await extension.on_tool(context)
    with pytest.raises(ValidationError):
        await context.tools[TOOL_SEARCH_TOOL_NAME](arguments)


async def test_search_defaults_and_discovery_state_are_request_local() -> None:
    extension = ToolSearchExtension(min_score=100.0)
    first = search_context(warehouse_stock)
    second = search_context(warehouse_stock)
    await extension.on_tool(first)
    await extension.on_tool(second)
    search = first.tools[TOOL_SEARCH_TOOL_NAME]
    assert await search({"queries": ["warehouse stock"]}) == []
    # Explicit zero overrides the configured threshold rather than being "unset".
    definitions = await search({"queries": ["warehouse stock"], "score": 0.0})
    assert [definition["name"] for definition in definitions] == ["warehouse_stock"]
    assert await search({"queries": ["warehouse stock"], "score": 0.0}) == []
    other = await second.tools[TOOL_SEARCH_TOOL_NAME]({"queries": ["warehouse stock"], "score": 0.0})
    assert other == definitions


async def test_search_reads_tools_registered_after_its_own_setup() -> None:
    extension = ToolSearchExtension()
    context = search_context()
    await extension.on_tool(context)
    search = context.tools[TOOL_SEARCH_TOOL_NAME]
    assert await search({"queries": ["warehouse stock"]}) == []
    context.register_tool(warehouse_stock)
    definitions = await search({"queries": ["warehouse stock"]})
    assert [definition["name"] for definition in definitions] == ["warehouse_stock"]


async def test_tool_guidelines_extension_renders_the_search_tool() -> None:
    observed: list[str] = []

    class Recorder(AgentExtension):
        async def before_model(self, context, request):
            observed.append(
                "\n".join(message.content for message in context.state.messages if isinstance(message, SystemMessage))
            )

    agent = await Agent.create(
        QuietModel(),
        config=AgentRunConfig("guidance"),
        extensions=[Recorder(), ToolSearchExtension(), ToolGuidelinesExtension()],
    )
    await agent.run("hello")

    assert "# Tool guidelines" in observed[0]
    assert "## tool_search" in observed[0]
    assert "BM25" in observed[0]
    assert "keyword" in observed[0]
    assert 'tool_search(queries=["warehouse stock inventory"])' in observed[0]


async def test_extension_adds_no_guidance_of_its_own() -> None:
    observed: list[str] = []

    class Recorder(AgentExtension):
        async def before_model(self, context, request):
            observed.append(
                "\n".join(message.content for message in context.state.messages if isinstance(message, SystemMessage))
            )

    agent = await Agent.create(
        QuietModel(),
        config=AgentRunConfig("guidance-once"),
        extensions=[Recorder(), ToolSearchExtension()],
    )
    await agent.run("hello")

    assert observed[0] == "You are a helpful assistant."


def test_bm25_ranks_the_closest_tools_and_drops_unrelated_ones() -> None:
    catalog = [warehouse_stock, billing_lookup, shipping_quote]
    index = BM25Search(catalog)

    assert [candidate.name for candidate in index.search(["warehouse inventory levels"])] == ["warehouse_stock"]
    assert [candidate.name for candidate in index.search(["shipping delivery time"])] == ["shipping_quote"]
    assert index.search(["unrelated topic"]) == []
    assert index.search([]) == []
    assert BM25Search(catalog, limit=1).search(["warehouse stock billing"]) == [warehouse_stock]


def test_bm25_matches_cjk_descriptions_by_character() -> None:
    index = BM25Search([stock_query, billing_lookup])
    assert [candidate.name for candidate in index.search(["库存"])] == ["stock_query"]


def test_bm25_matches_guidelines_too() -> None:
    @tool(guidelines="Use when the caller asks about pallet counts.")
    def inventory_level(warehouse: str) -> str:
        """Return a number for one site.

        Args:
            warehouse: Site code.
        """
        return f"{warehouse}:1"

    index = BM25Search([inventory_level, billing_lookup])

    assert [candidate.name for candidate in index.search(["pallet counts"])] == ["inventory_level"]


def test_bm25_matches_parameter_descriptions_and_snippets_too() -> None:
    @tool(guidelines="Use for site data.")
    def site_report(site: str) -> str:
        """Return a report for one site.

        Args:
            site: Two-letter depot code, for example "sh".

        Snippet:
            site_report(site="sh")
        """
        return site

    index = BM25Search([site_report, billing_lookup])

    assert [candidate.name for candidate in index.search(["depot code"])] == ["site_report"]
    assert [candidate.name for candidate in index.search(["site_report site sh"])] == ["site_report"]


def test_bm25_keeps_catalog_order_for_equal_scores() -> None:
    assert BM25Search([warehouse_stock, warehouse_stock]).search(["warehouse"]) == [warehouse_stock, warehouse_stock]


def test_bm25_respects_a_score_threshold() -> None:
    catalog = [warehouse_stock, billing_lookup]
    index = BM25Search(catalog)

    assert index.search(["warehouse stock"]) == [warehouse_stock]
    assert BM25Search(catalog, min_score=10.0).search(["warehouse stock"]) == []
    assert index.search(["warehouse stock"], min_score=10.0) == []
    assert index.search(["warehouse stock"], min_score=0.5) == [warehouse_stock]


async def test_extension_honors_a_score_threshold_from_the_request() -> None:
    search = {
        "id": "ts_1",
        "call_id": "call_search",
        "type": "tool_search_call",
        "status": "completed",
        "execution": "client",
        "arguments": {"queries": ["warehouse stock"], "score": 10.0},
    }
    answer = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Nothing matched.", "annotations": [], "logprobs": []}],
    }
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        output = [search] if len(requests) == 1 else [answer]
        return httpx.Response(200, content=_sse(output), headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(
        provider,
        config=AgentRunConfig("threshold"),
        tools=(warehouse_stock,),
        extensions=[ToolSearchExtension()],
    )
    try:
        result = await agent.run("Search with a high threshold")
    finally:
        await provider.aclose()

    assert result.content == "Nothing matched."
    assert requests[1]["input"][-1]["tools"] == []


async def test_agent_discovers_and_runs_a_catalog_tool() -> None:
    executed: list[str] = []

    @tool(deferred=True, guidelines="Use for stock questions.")
    def stock(warehouse: str) -> str:
        """Return the stock level for one warehouse.

        Args:
            warehouse: Warehouse code to look up.
        """
        executed.append(warehouse)
        return f"{warehouse}:42"

    @tool(guidelines="Use for always-visible questions.")
    def visible_note(topic: str) -> str:
        """Return a note that is always declared.

        Args:
            topic: Note topic.
        """
        return topic

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
        "name": "stock",
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
        return httpx.Response(200, content=_sse(output), headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(
        provider,
        config=AgentRunConfig("discovery"),
        tools=(stock, visible_note),
        extensions=[ToolSearchExtension()],
    )
    try:
        result = await agent.run("How much stock is in the shanghai warehouse?")
    finally:
        await provider.aclose()

    assert result.content == "Shanghai has 42."
    assert executed == ["shanghai"]
    # Deferred tools stay hidden; the visible tool keeps its ordinary schema.
    declared = requests[0]["tools"]
    assert [rule.get("name") or rule["type"] for rule in declared] == ["visible_note", "tool_search"]
    search_rule = declared[-1]
    assert search_rule["execution"] == "client"
    assert set(search_rule["parameters"]["properties"]) == {"queries", "score"}
    assert search_rule["parameters"]["required"] == ["queries"]
    assert search_rule["parameters"]["additionalProperties"] is False
    assert search_rule["parameters"]["properties"]["queries"]["items"] == {"type": "string"}
    assert "BM25" in search_rule["description"]
    # A discovered tool is callable but still never declared as a function, so
    # the request cannot repeat the definition the search already returned.
    assert [rule.get("name") or rule["type"] for rule in requests[1]["tools"]] == ["visible_note", "tool_search"]
    searched = requests[1]["input"][-1]
    assert searched["type"] == "tool_search_output"
    assert [candidate["name"] for candidate in searched["tools"]] == ["stock"]
    # A deferred tool's guidelines can only reach the model through this result.
    assert searched["tools"][0]["description"].endswith("Guidelines:\n- Use for stock questions.")
    assert requests[2]["input"][-2:] == [
        function,
        {"type": "function_call_output", "call_id": "call_1", "output": "shanghai:42"},
    ]


async def test_extension_never_repeats_a_discovered_definition() -> None:
    first = {
        "id": "ts_1",
        "call_id": "call_search_1",
        "type": "tool_search_call",
        "status": "completed",
        "execution": "client",
        "arguments": {"queries": ["warehouse stock"]},
    }
    second = {**first, "id": "ts_2", "call_id": "call_search_2"}
    answer = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Nothing new.", "annotations": [], "logprobs": []}],
    }
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        output = [[first], [second], [answer]][len(requests) - 1]
        return httpx.Response(200, content=_sse(output), headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("deepseek-flash", "key", transport=httpx.MockTransport(handler), response=True)
    agent = await Agent.create(
        provider,
        config=AgentRunConfig("repeat-search"),
        tools=(warehouse_stock,),
        extensions=[ToolSearchExtension()],
    )
    try:
        result = await agent.run("Search twice for the same tool")
    finally:
        await provider.aclose()

    assert result.content == "Nothing new."
    first_output = requests[1]["input"][-1]
    second_output = requests[2]["input"][-1]
    assert [candidate["name"] for candidate in first_output["tools"]] == ["warehouse_stock"]
    assert second_output == {
        "type": "tool_search_output",
        "call_id": "call_search_2",
        "execution": "client",
        "status": "completed",
        "tools": [],
    }
