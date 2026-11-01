"""Deferred tools and ToolGuidelinesExtension share one setting for what a description carries.

A deferred tool is skipped by ToolGuidelinesExtension, so its description (on the
request for Anthropic, in the search result for Responses) is the only place the
model can read its guidance. These tests pin that the extension's switches decide
what that description holds, in either registration order.
"""

import json

import pytest

from test_anthropic_tool_search import FakeAnthropic, echo, text_answer
from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.tool_guidelines import ToolGuidancePlacement, ToolGuidelinesExtension
from zett_agent.extensions.tool_search import ToolSearchExtension
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.tools.base import tool


@tool(deferred=True, guidelines="Use for stock questions.", snippet='stock_lookup(sku="A1")')
async def stock_lookup(sku: str) -> str:
    """Look up stock.

    Args:
        sku: Stock keeping unit.
    """
    return sku


SNIPPET = 'stock_lookup(sku="A1")'
GUIDELINES = "Use for stock questions."

CASES = [
    # (guidelines extension options, expected parts of the deferred description)
    (None, False, True),  # no extension: guidelines only, the long-standing behavior
    ({}, True, True),
    ({"placement": ToolGuidancePlacement.DESCRIPTION}, True, True),
    ({"guidelines": False}, True, False),
    ({"snippets": False}, False, True),
    ({"placement": ToolGuidancePlacement.DESCRIPTION, "guidelines": False}, True, False),
    ({"placement": ToolGuidancePlacement.DESCRIPTION, "snippets": False}, False, True),
    ({"snippets": False, "guidelines": False}, False, False),
]


def expected(snippet: bool, guidelines: bool) -> str:
    return " ".join(["Look up stock.", *([SNIPPET] if snippet else []), *([GUIDELINES] if guidelines else [])])


@pytest.mark.parametrize("guidelines_first", [True, False])
@pytest.mark.parametrize(("options", "snippet", "guidelines"), CASES)
async def test_anthropic_deferred_description_follows_the_guidelines_extension(
    options, snippet, guidelines, guidelines_first
):
    server = FakeAnthropic(text_answer("ok"))
    provider = AnthropicProvider("claude", "key", transport=server.transport)
    extensions = [ToolSearchExtension()]
    if options is not None:
        extension = ToolGuidelinesExtension(**options)
        extensions.insert(0 if guidelines_first else len(extensions), extension)
    agent = await Agent.create(
        provider, config=AgentRunConfig("combo"), tools=[echo, stock_lookup], extensions=extensions
    )
    try:
        await agent.run("hi")
    finally:
        await provider.aclose()

    declared = {item["name"]: item for item in server.bodies[0]["tools"]}
    assert declared["stock_lookup"]["description"] == expected(snippet, guidelines)
    assert declared["stock_lookup"]["defer_loading"] is True
    # The guidance appears once, whichever extension rewrote the request first.
    assert declared["stock_lookup"]["description"].count("Use for stock questions.") == int(guidelines)
    # Deferred guidance never leaks into the system prompt, where the tool is not callable yet.
    system = server.bodies[0].get("system") or []
    assert "stock_lookup" not in "".join(block["text"] for block in system)


@pytest.mark.parametrize(("options", "snippet", "guidelines"), CASES)
async def test_a_search_result_carries_the_same_parts(options, snippet, guidelines):
    """The Responses path returns the definition in the search result instead of the request."""
    seen: list[list[ToolMessage]] = []

    class Searcher:
        async def stream(self, request):
            results = [m for m in request.messages if isinstance(m, ToolMessage)]
            seen.append(results)
            message = (
                AssistantMessage(content="done")
                if results
                else AssistantMessage(tool_calls=(ToolCall("s1", "tool_search", {"queries": ["stock"]}),))
            )
            yield ModelEvent.completed(ModelResponse(message))

    extensions = [ToolSearchExtension()]
    if options is not None:
        extensions.append(ToolGuidelinesExtension(**options))
    agent = await Agent.create(
        Searcher(), config=AgentRunConfig("search"), tools=[echo, stock_lookup], extensions=extensions
    )

    await agent.run("find stock")

    [definition] = json.loads(seen[-1][0].content)
    assert definition["description"] == expected(snippet, guidelines)
