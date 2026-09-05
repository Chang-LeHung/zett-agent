"""Opt-in live SDK checks; credentials remain in environment variables."""

import asyncio
import os
import ssl

import httpx
import pytest
import truststore

from kcs_agent import (
    Agent,
    AgentEventType,
    AnthropicProvider,
    DeepSeekProvider,
    ModelEventType,
    ModelRequest,
    ReasoningEffort,
    UserMessage,
    tool,
)

pytestmark = pytest.mark.skipif(
    os.getenv("KCS_AGENT_LIVE_TESTS") != "1" or not os.getenv("DEEPSEEK_API"),
    reason="Set KCS_AGENT_LIVE_TESTS=1 and DEEPSEEK_API to run live provider checks",
)


def provider_for(protocol: str):
    """Use the operating system trust store, including locally trusted proxy certificates."""
    transport = httpx.AsyncHTTPTransport(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    model = os.getenv("DEEPSEEK_API_MODEL", "deepseek-v4-flash")
    key = os.environ["DEEPSEEK_API"]
    match protocol:
        case "openai":
            return DeepSeekProvider(model=model, api_key=key, transport=transport)
        case "anthropic":
            return AnthropicProvider(
                model=os.getenv("DEEPSEEK_ANTHROPIC_MODEL", model),
                api_key=os.getenv("DEEPSEEK_ANTHROPIC_API_KEY", key),
                base_url=os.getenv("DEEPSEEK_ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"),
                transport=transport,
            )
        case _:
            raise ValueError("Unknown protocol")


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_live_stream(protocol: str) -> None:
    provider = provider_for(protocol)
    try:
        async with asyncio.timeout(60):
            events = [
                event
                async for event in provider.stream(
                    ModelRequest(
                        messages=(UserMessage(content="Reply with one short word: ping"),),
                        reasoning_effort=ReasoningEffort.OFF,
                    )
                )
            ]
        responses = [event.response for event in events if event.type == ModelEventType.RESPONSE]
        assert len(responses) == 1
        assert responses[0].message.content
        assert responses[0].usage.input_tokens > 0
        assert any(event.type == ModelEventType.TEXT_DELTA for event in events)
    finally:
        await provider.aclose()


@tool(guidelines="Use to compute the user's requested sum.")
def add(left: int, right: int) -> int:
    """Add two integer values."""
    return left + right


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("effort", [ReasoningEffort.OFF, ReasoningEffort.HIGH])
async def test_live_tool_round_trip(protocol: str, effort: ReasoningEffort) -> None:
    provider = provider_for(protocol)
    try:
        agent = Agent(
            provider,
            system_prompt="Call add exactly once to compute 2+3, then reply with the tool result only.",
            tools=[add],
            max_iterations=3,
        )
        async with asyncio.timeout(90):
            events = [
                event
                async for event in agent.stream(
                    UserMessage(content="Use add to compute 2+3."),
                    reasoning_effort=effort,
                )
            ]
        tools = [event for event in events if event.type == AgentEventType.TOOL_COMPLETED]
        assert len(tools) == 1
        assert tools[0].message.content == "5"
        assert events[-1].type == AgentEventType.RUN_COMPLETED
        assert "5" in events[-1].message.content
        assert sum(event.type == AgentEventType.MODEL_COMPLETED for event in events) == 2
        # Effort controls reasoning budget; the provider may return no visible
        # reasoning for a trivial request. Preserve whatever it actually returns.
        streamed_reasoning = "".join(event.delta for event in events if event.type == AgentEventType.REASONING_DELTA)
        completed_reasoning = "".join(
            event.response.message.reasoning or "" for event in events if event.type == AgentEventType.MODEL_COMPLETED
        )
        assert streamed_reasoning == completed_reasoning
    finally:
        await provider.aclose()
