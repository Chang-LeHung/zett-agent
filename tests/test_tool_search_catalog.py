"""The searchable-tool list: where it goes, and that it is independent of ToolGuidelinesExtension."""

from dataclasses import replace

import pytest

from test_anthropic_tool_search import FakeAnthropic, echo, text_answer
from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.tool_guidelines import ToolGuidancePlacement, ToolGuidelinesExtension
from zett_agent.extensions.tool_search import (
    TOOL_SEARCH_TOOL_NAME,
    ToolCatalogPlacement,
    ToolSearchExtension,
    render_tool_catalog,
)
from zett_agent.messages import AssistantMessage, SystemMessage
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.tools.base import render_tool_description, tool


@tool(deferred=True, guidelines=["Use for stock.", "Check the SKU format first."], snippet='stock_lookup(sku="A1")')
async def stock_lookup(sku: str) -> str:
    """Look up stock.

    Args:
        sku: Stock keeping unit.
    """
    return sku


@tool(deferred=True, guidelines="Use for invoices.")
async def invoice_lookup(invoice: str) -> str:
    """Look up an invoice.

    Args:
        invoice: Invoice number.
    """
    return invoice


STOCK = '- stock_lookup: Look up stock.\n  snippet: stock_lookup(sku="A1")\n  guidelines:\n  - Use for stock.\n  - Check the SKU format first.'
INVOICE = "- invoice_lookup: Look up an invoice.\n  guidelines:\n  - Use for invoices."
SEARCH, SYSTEM, NONE = (
    ToolCatalogPlacement.SEARCH_DESCRIPTION,
    ToolCatalogPlacement.SYSTEM_PROMPT,
    ToolCatalogPlacement.NONE,
)
IN_SYSTEM, IN_DESCRIPTION = ToolGuidancePlacement.SYSTEM_PROMPT, ToolGuidancePlacement.DESCRIPTION


async def anthropic_request(extensions, tools=(echo, stock_lookup, invoice_lookup)):
    """Return the search description, the system text, and every tool description sent to Anthropic."""
    server = FakeAnthropic(text_answer("ok"))
    provider = AnthropicProvider("claude", "key", transport=server.transport)
    agent = await Agent.create(provider, config=AgentRunConfig("catalog"), tools=list(tools), extensions=extensions)
    try:
        await agent.run("hi")
    finally:
        await provider.aclose()
    body = server.bodies[0]
    descriptions = {item["name"]: item["description"] for item in body["tools"]}
    system = "".join(block["text"] for block in body.get("system") or [])
    return descriptions.get(TOOL_SEARCH_TOOL_NAME, ""), system, descriptions


async def test_the_search_description_lists_every_deferred_tool_by_default():
    search, system, descriptions = await anthropic_request([ToolSearchExtension()])

    assert search.startswith("Search registered tools by BM25 keyword relevance.\n\nSearchable tools (not loaded yet")
    assert f"{STOCK}\n{INVOICE}" in search
    assert "Searchable tools" not in system
    # Only deferred tools are listed; an ordinary tool is just a tool.
    assert "- echo:" not in search and descriptions["echo"] == "Echo text back."


async def test_the_system_prompt_placement_leaves_the_search_description_alone():
    search, system, _ = await anthropic_request([ToolSearchExtension(catalog=SYSTEM)])

    assert search == "Search registered tools by BM25 keyword relevance."
    assert "# Searchable tools\nThese tools are not loaded yet. Call tool_search with keywords" in system
    assert f"{STOCK}\n{INVOICE}" in system


async def test_no_placement_lists_nothing():
    search, system, _ = await anthropic_request([ToolSearchExtension(catalog=NONE)])

    assert search == "Search registered tools by BM25 keyword relevance."
    assert "Searchable tools" not in system and "stock_lookup" not in system


@pytest.mark.parametrize(
    ("options", "has_snippet", "has_guidelines"),
    [
        ({}, True, True),
        ({"catalog_snippets": False}, False, True),
        ({"catalog_guidelines": False}, True, False),
        ({"catalog_snippets": False, "catalog_guidelines": False}, False, False),
    ],
)
@pytest.mark.parametrize("placement", [SEARCH, SYSTEM])
async def test_each_part_of_an_entry_can_be_switched_off(placement, options, has_snippet, has_guidelines):
    search, system, _ = await anthropic_request([ToolSearchExtension(catalog=placement, **options)])
    text = search if placement is SEARCH else system

    assert "- stock_lookup: Look up stock." in text and "- invoice_lookup: Look up an invoice." in text
    assert ("snippet:" in text) is has_snippet
    assert ("guidelines:" in text) is has_guidelines


@pytest.mark.parametrize("search_first", [True, False])
@pytest.mark.parametrize("guidance_placement", [IN_SYSTEM, IN_DESCRIPTION])
@pytest.mark.parametrize("catalog_placement", [SEARCH, SYSTEM, NONE])
async def test_the_catalog_and_tool_guidelines_combine_in_every_placement_and_order(
    catalog_placement, guidance_placement, search_first
):
    search_extension = ToolSearchExtension(catalog=catalog_placement)
    guidelines_extension = ToolGuidelinesExtension(placement=guidance_placement)
    extensions = [search_extension, guidelines_extension] if search_first else [guidelines_extension, search_extension]

    search, system, descriptions = await anthropic_request(extensions)

    # The catalog lands exactly where it was asked to, whatever the other extension does.
    assert (STOCK in search) is (catalog_placement is SEARCH)
    assert (STOCK in system) is (catalog_placement is SYSTEM)
    # ToolGuidelinesExtension still delivers the search tool's own guidance, appended after its text.
    own_guidance = "Use to discover deferred tools that are not listed in the tool definitions."
    assert (own_guidance in search) is (guidance_placement is IN_DESCRIPTION)
    assert (own_guidance in system) is (guidance_placement is IN_SYSTEM)
    assert descriptions["echo"].count("Use to repeat text.") == int(guidance_placement is IN_DESCRIPTION)
    # One extension never discards what the other wrote: both sit in the same description.
    if catalog_placement is SEARCH and guidance_placement is IN_DESCRIPTION:
        assert (
            search.count("Searchable tools (not loaded yet") == 1
            and search.count("Use to discover deferred tools that are not listed in the tool definitions.") == 1
        )
        # The search tool's guidance must not continue the catalog's last bullet,
        # where a model would read it as part of that deferred tool's rule.
        assert "  - Use for invoices. tool_search(" not in search
        assert "  - Use for invoices. Use to discover" not in search
    # Tool guidelines still skips deferred tools, so their rules appear once, in the catalog.
    if catalog_placement is not NONE:
        assert (search + system).count("- Check the SKU format first.") == 1


async def test_the_catalog_does_not_need_tool_guidelines_and_works_alone():
    search, system, _ = await anthropic_request([ToolSearchExtension(catalog=SYSTEM)])

    assert STOCK in system and "# Tool guidelines" not in system


async def test_catalog_switches_are_independent_of_the_tool_guidelines_switches():
    search, system, descriptions = await anthropic_request(
        [
            ToolGuidelinesExtension(placement=IN_DESCRIPTION, guidelines=False),
            ToolSearchExtension(catalog_guidelines=True, catalog_snippets=False),
        ]
    )

    assert "Check the SKU format first." in search and "snippet:" not in search
    assert "Use to repeat text." not in descriptions["echo"]


async def test_the_returns_switch_reaches_a_deferred_tools_description():
    """ToolGuidelinesExtension skips deferred tools, so tool search is their only channel."""
    _, _, with_returns = await anthropic_request(
        [ToolGuidelinesExtension(placement=IN_DESCRIPTION, returns=True), ToolSearchExtension()]
    )
    _, _, plain = await anthropic_request([ToolGuidelinesExtension(placement=IN_DESCRIPTION), ToolSearchExtension()])

    assert 'returns {"type":"string"}' in with_returns["stock_lookup"]
    assert "returns" not in plain["stock_lookup"]


async def test_nothing_is_added_when_no_tool_is_deferred():
    search, system, _ = await anthropic_request([ToolSearchExtension()], tools=(echo,))
    assert search == "Search registered tools by BM25 keyword relevance."
    assert "Searchable tools" not in system

    _, system, _ = await anthropic_request([ToolSearchExtension(catalog=SYSTEM)], tools=(echo,))
    assert "Searchable tools" not in system


async def test_the_system_catalog_appears_once_per_request_across_requests():
    seen: list[list[SystemMessage]] = []

    class Model:
        async def stream(self, request):
            seen.append([m for m in request.messages if isinstance(m, SystemMessage)])
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    agent = await Agent.create(
        Model(),
        config=AgentRunConfig("twice"),
        tools=[echo, stock_lookup],
        extensions=[ToolSearchExtension(catalog=SYSTEM)],
    )
    await agent.run("one")
    await agent.run("two")

    for messages in seen:
        assert sum("# Searchable tools" in m.content for m in messages) == 1


async def test_a_provider_that_hides_deferred_tools_still_gets_the_catalog_in_the_search_tool():
    """The Responses path never sends the deferred definitions, so the list is the only hint."""
    seen = []

    class Plain:
        async def stream(self, request):
            seen.append(request)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    agent = await Agent.create(
        Plain(), config=AgentRunConfig("plain"), tools=[echo, stock_lookup], extensions=[ToolSearchExtension()]
    )
    await agent.run("hi")

    definitions = {definition.name: definition for definition in seen[0].tools}
    assert set(definitions) == {"echo", TOOL_SEARCH_TOOL_NAME}
    assert STOCK in definitions[TOOL_SEARCH_TOOL_NAME].description
    assert definitions[TOOL_SEARCH_TOOL_NAME].local_tool_search is True


async def test_the_catalog_never_alters_the_registered_tools():
    before = stock_lookup.description
    agent = await Agent.create(
        AnthropicProvider("claude", "key", transport=FakeAnthropic(text_answer("a"), text_answer("b")).transport),
        config=AgentRunConfig("static"),
        tools=[echo, stock_lookup],
        extensions=[ToolSearchExtension()],
    )
    await agent.run("one")
    await agent.run("two")

    assert stock_lookup.description == before and agent.tools["stock_lookup"].description == before
    assert TOOL_SEARCH_TOOL_NAME not in agent.tools  # the search tool is request-scoped


def test_render_tool_catalog_formats_multiline_descriptions_and_handles_no_tools():
    @tool(deferred=True, guidelines="Rule.")
    async def long_one() -> str:
        """First line."""
        return ""

    long_one = replace(long_one, description="First line.\nSecond line.")

    assert render_tool_catalog([]) == ""
    assert render_tool_catalog([long_one], snippets=False, guidelines=False).splitlines() == [
        "- long_one: First line.",
        "  Second line.",
    ]
    assert render_tool_catalog([long_one], snippets=False).endswith("  guidelines:\n  - Rule.")


def test_render_tool_description_builds_on_a_supplied_base():
    text = render_tool_description(stock_lookup, base="Prefix.")

    assert text == 'Prefix. stock_lookup(sku="A1") Use for stock. Check the SKU format first.'


def test_the_placement_is_validated():
    assert ToolSearchExtension(catalog="system_prompt").catalog is SYSTEM
    with pytest.raises(ValueError):
        ToolSearchExtension(catalog="elsewhere")
