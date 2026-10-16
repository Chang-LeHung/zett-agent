"""Opt-in end-to-end checks against a live gateway that serves both protocols.

The unit tests cover what each adapter sends and how it reads a recorded stream. These tests ask the
other question: does a real endpoint accept what the adapter sends, and does the adapter understand
what comes back. They need a gateway that speaks the OpenAI Responses API at ``/responses`` and the
Anthropic Messages API at ``/v1/messages``. Nothing runs, and nothing touches the network, unless the
checks are switched on.

Set ``ZETT_AGENT_LIVE_TESTS=1`` and ``ZETT_AGENT_GATEWAY_URL`` to switch them on. Each group also
names the model it needs, because one endpoint rarely serves every capability:

=========================================  ==================================================
Environment variable                       What it switches on
=========================================  ==================================================
``ZETT_AGENT_GATEWAY_URL``                 Gateway base URL, e.g. ``http://127.0.0.1:8787``
``ZETT_AGENT_GATEWAY_TOKEN``               Credential sent to the gateway; default ``test``
``ZETT_AGENT_GATEWAY_RESPONSES_MODEL``     Model for the Responses API checks
``ZETT_AGENT_GATEWAY_MESSAGES_MODEL``      Model for the Messages API checks
``ZETT_AGENT_GATEWAY_SEARCH_MODEL``        Model that runs hosted tools (web search, tool search)
``ZETT_AGENT_GATEWAY_TIMEOUT``             Seconds allowed per request; default ``180``
=========================================  ==================================================

A gateway that refuses a hosted tool type with HTTP 400 skips that check rather than failing it: a
service is entitled not to offer a tool, and the adapter's handling of the refusal is what the unit
tests cover. A stream that ends without a terminal response is never skipped, because that is the
adapter's own bug to catch.

Run them with::

    ZETT_AGENT_LIVE_TESTS=1 \\
    ZETT_AGENT_GATEWAY_URL=http://127.0.0.1:8787 \\
    ZETT_AGENT_GATEWAY_RESPONSES_MODEL=gpt-5.6-sol \\
    ZETT_AGENT_GATEWAY_MESSAGES_MODEL='claude-sonnet-5[1m]' \\
    ZETT_AGENT_GATEWAY_SEARCH_MODEL=kimi-k3 \\
    uv run pytest tests/test_live_gateway.py
"""

import os
from collections.abc import Sequence

import pytest

from zett_agent._compat import timeout
from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.extensions.server_tools.anthropic import AnthropicServerToolExtension
from zett_agent.extensions.server_tools.openai import OpenAIServerToolExtension
from zett_agent.extensions.tool_search import ToolSearchExtension
from zett_agent.messages import AssistantMessage, UserMessage
from zett_agent.model import (
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelResponse,
    ReasoningEffort,
    RetryOptions,
    ServerToolDefinition,
)
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.base import ProviderHTTPError
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.tools.base import tool

BASE = os.getenv("ZETT_AGENT_GATEWAY_URL", "")
TOKEN = os.getenv("ZETT_AGENT_GATEWAY_TOKEN", "test")
RESPONSES_MODEL = os.getenv("ZETT_AGENT_GATEWAY_RESPONSES_MODEL", "")
MESSAGES_MODEL = os.getenv("ZETT_AGENT_GATEWAY_MESSAGES_MODEL", "")
SEARCH_MODEL = os.getenv("ZETT_AGENT_GATEWAY_SEARCH_MODEL", "")
REQUEST_TIMEOUT = float(os.getenv("ZETT_AGENT_GATEWAY_TIMEOUT", "180"))

#: A retry would replay a long request on a failure these checks want to see immediately.
NO_RETRY = RetryOptions(max_retries=0)

pytestmark = pytest.mark.skipif(
    os.getenv("ZETT_AGENT_LIVE_TESTS") != "1" or not BASE,
    reason="Set ZETT_AGENT_LIVE_TESTS=1 and ZETT_AGENT_GATEWAY_URL to run live gateway checks",
)

needs_responses_model = pytest.mark.skipif(
    not RESPONSES_MODEL, reason="Set ZETT_AGENT_GATEWAY_RESPONSES_MODEL to run Responses checks"
)
needs_messages_model = pytest.mark.skipif(
    not MESSAGES_MODEL, reason="Set ZETT_AGENT_GATEWAY_MESSAGES_MODEL to run Messages checks"
)
needs_search_model = pytest.mark.skipif(
    not SEARCH_MODEL, reason="Set ZETT_AGENT_GATEWAY_SEARCH_MODEL to run hosted-tool checks"
)

#: Tool calls a check observes. Reset at the start of every agent run.
CALLS: list[str] = []


@tool(deferred=True, guidelines="Use for warehouse stock questions.")
def warehouse_stock(sku: str) -> str:
    """Return the units in stock for a SKU."""
    CALLS.append(f"stock:{sku}")
    return f"{sku} has 7 units"


@tool(deferred=True, guidelines="Use for billing questions.")
def billing_lookup(invoice: str) -> str:
    """Return the status of an invoice."""
    CALLS.append(f"billing:{invoice}")
    return f"{invoice} is paid"


@tool(guidelines="Use to repeat text.")
def echo(text: str) -> str:
    """Repeat the text."""
    CALLS.append(f"echo:{text}")
    return text


LOCAL_TOOLS = [echo, warehouse_stock, billing_lookup]


def responses_provider(model: str | None = None) -> OpenAIProvider:
    """A Responses API provider pointed at the gateway."""
    return OpenAIProvider(model=model or RESPONSES_MODEL, api_key=TOKEN, base_url=BASE, response=True, retry=NO_RETRY)


def messages_provider(model: str | None = None) -> AnthropicProvider:
    """A Messages API provider pointed at the gateway."""
    return AnthropicProvider(model or MESSAGES_MODEL, TOKEN, base_url=BASE, retry=NO_RETRY)


async def collect(provider, request: ModelRequest) -> list[ModelEvent]:
    """Drain one stream under a timeout, keeping every event it produced."""
    events: list[ModelEvent] = []
    async with timeout(REQUEST_TIMEOUT):
        async for event in provider.stream(request):
            events.append(event)
    return events


def final_response(events: Sequence[ModelEvent]) -> ModelResponse:
    """Return the one terminal response, failing loudly when there is none or more than one."""
    responses = [event.response for event in events if event.type is ModelEventType.RESPONSE and event.response]
    assert len(responses) == 1, f"expected exactly one terminal response, saw {len(responses)}"
    return responses[0]


def skip_when_tool_is_unsupported(error: ProviderHTTPError) -> None:
    """Skip a hosted-tool check the gateway refuses, instead of failing it."""
    if error.status_code == 400:
        pytest.skip(f"gateway does not offer this hosted tool: {error.reason or error}")
    raise error


async def run_agent(provider, *, extensions, prompt: str, tools: Sequence | None = None) -> AssistantMessage:
    """Run one agent turn, returning its answer and recording local tool calls in ``CALLS``."""
    CALLS.clear()
    config = AgentRunConfig(session_id="live-gateway")
    agent = await Agent.create(
        provider, tools=LOCAL_TOOLS if tools is None else tools, extensions=extensions, config=config
    )
    try:
        async with timeout(REQUEST_TIMEOUT):
            return await agent.run(prompt, config=config)
    finally:
        await provider.aclose()


# ------------------------------------------------------------------ the plain answer path


@needs_responses_model
async def test_responses_model_answers_and_reports_usage() -> None:
    """The Responses path streams text and a terminal response with usage."""
    provider = responses_provider()
    try:
        events = await collect(
            provider,
            ModelRequest(
                messages=(UserMessage(content="Reply with one short word: ping"),),
                reasoning_effort=ReasoningEffort.OFF,
            ),
        )
    finally:
        await provider.aclose()

    response = final_response(events)
    assert response.message.content
    # Not every endpoint reports input tokens; the generated count is the portable signal.
    assert response.usage.output_tokens > 0
    assert any(event.type is ModelEventType.TEXT_DELTA for event in events)


@needs_messages_model
async def test_messages_model_answers_and_reports_usage() -> None:
    """The Messages path streams text and a terminal response with usage."""
    provider = messages_provider()
    try:
        events = await collect(
            provider,
            ModelRequest(
                messages=(UserMessage(content="Reply with one short word: ping"),),
                reasoning_effort=ReasoningEffort.OFF,
            ),
        )
    finally:
        await provider.aclose()

    response = final_response(events)
    assert response.message.content
    assert response.usage.output_tokens > 0


# ------------------------------------------------------------------ hosted (server) tools


@needs_responses_model
async def test_responses_hosted_web_search_returns_a_result_or_is_refused_cleanly() -> None:
    """A hosted web search either reports itself through server-tool events or is refused.

    The answer may be empty when the service ran the search but produced no text; what must not happen
    is a crash, or a stream that ends without a terminal response.
    """
    provider = responses_provider()
    request = ModelRequest(
        messages=(UserMessage(content="Search the web for the current year."),),
        server_tools=(ServerToolDefinition("web_search"),),
        reasoning_effort=ReasoningEffort.OFF,
    )
    try:
        events = await collect(provider, request)
    except ProviderHTTPError as error:
        skip_when_tool_is_unsupported(error)
        return
    finally:
        await provider.aclose()

    final_response(events)
    reported = {
        ModelEventType.SERVER_TOOL_STARTED,
        ModelEventType.SERVER_TOOL_COMPLETED,
        ModelEventType.SERVER_TOOL_FAILED,
    }
    assert any(event.type in reported for event in events), "a hosted search must report its own activity"


@needs_search_model
async def test_messages_hosted_web_search_returns_a_result_or_is_refused_cleanly() -> None:
    """The same contract on the Anthropic Messages side."""
    provider = messages_provider(SEARCH_MODEL)
    request = ModelRequest(
        messages=(UserMessage(content="Search the web for the current year."),),
        server_tools=(ServerToolDefinition("web_search_20250305", {"name": "web_search", "max_uses": 1}),),
        reasoning_effort=ReasoningEffort.OFF,
    )
    try:
        events = await collect(provider, request)
    except ProviderHTTPError as error:
        skip_when_tool_is_unsupported(error)
        return
    finally:
        await provider.aclose()

    final_response(events)


@needs_responses_model
async def test_a_hosted_item_failure_is_not_reported_as_a_broken_stream() -> None:
    """A hosted tool that fails must not turn into an error, or into a missing terminal response.

    An MCP server that never answers ends its item with a failure event and the stream stops there.
    The service explained itself, so the adapter must keep what it has and finish the response.
    """
    provider = responses_provider()
    request = ModelRequest(
        messages=(UserMessage(content="Use the mcp tool to list its tools."),),
        server_tools=(
            ServerToolDefinition("mcp", {"server_label": "probe", "server_url": "https://example.invalid/mcp"}),
        ),
        reasoning_effort=ReasoningEffort.OFF,
    )
    try:
        events = await collect(provider, request)
    except ProviderHTTPError as error:
        skip_when_tool_is_unsupported(error)
        return
    finally:
        await provider.aclose()

    final_response(events)


# ------------------------------------------------------------------ tool search


@needs_responses_model
async def test_responses_local_tool_search_loads_a_deferred_tool() -> None:
    """The client-side search loads a deferred definition, and the loaded tool then runs."""
    provider = responses_provider()
    reply = await run_agent(provider, extensions=[ToolSearchExtension()], prompt="How many A1 units are in stock?")
    assert CALLS == ["stock:A1"]
    assert "7" in reply.content


@needs_messages_model
async def test_messages_local_tool_search_loads_a_deferred_tool() -> None:
    """The same client-side search over the Messages protocol."""
    provider = messages_provider()
    reply = await run_agent(provider, extensions=[ToolSearchExtension()], prompt="How many A1 units are in stock?")
    assert CALLS == ["stock:A1"]
    assert "7" in reply.content


@needs_search_model
async def test_messages_hosted_tool_search_loads_a_deferred_tool() -> None:
    """The service searches its own copy of the deferred tools, and the loaded tool then runs."""
    provider = messages_provider(SEARCH_MODEL)
    extension = AnthropicServerToolExtension(
        [ServerToolDefinition("tool_search_tool_regex_20251119", {"name": "tool_search_tool_regex"})]
    )
    try:
        reply = await run_agent(
            provider,
            extensions=[extension],
            prompt="How many A1 units are in stock?",
        )
    except ProviderHTTPError as error:
        skip_when_tool_is_unsupported(error)
        return

    assert CALLS == ["stock:A1"]
    assert "7" in reply.content


# ------------------------------------------------------------------ tools the client cannot run


@needs_responses_model
async def test_a_tool_the_client_cannot_run_does_not_break_the_turn() -> None:
    """A call to a tool the client never registered becomes a failed result, not a crash.

    Hosted computer use is often delegated back to the client, which may have nothing to run it with.
    The model must still finish its turn and say so.
    """
    provider = responses_provider()
    extension = OpenAIServerToolExtension(
        [ServerToolDefinition("computer_use", {"display_width": 1024, "display_height": 768, "environment": "linux"})]
    )
    try:
        # Only a plain tool: deferred ones would ask this endpoint for a hosted tool search it refuses.
        reply = await run_agent(
            provider, extensions=[extension], prompt="Take a screenshot of the desktop.", tools=[echo]
        )
    except ProviderHTTPError as error:
        skip_when_tool_is_unsupported(error)
        return

    assert isinstance(reply.content, str)
