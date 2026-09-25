"""Opt-in live checks against an unauthenticated Responses API gateway.

The default target is the local gateway at ``http://192.168.1.11:8787/v1``,
which serves ``deepseek-flash`` without a credential. Override the endpoint or
model with ``ZETT_AGENT_LIVE_RESPONSES_BASE_URL`` and
``ZETT_AGENT_LIVE_RESPONSES_MODEL``. Run with ``ZETT_AGENT_LIVE_TESTS=1``; the
credential-based checks in ``test_live_providers.py`` stay separate.
"""

from __future__ import annotations

import os

import pytest

from zett_agent import (
    Agent,
    AgentEventType,
    AgentRunConfig,
    ModelEventType,
    ModelRequest,
    OpenAIProvider,
    ReasoningEffort,
    UserMessage,
    tool,
)
from zett_agent._compat import timeout

pytestmark = pytest.mark.skipif(
    os.getenv("ZETT_AGENT_LIVE_TESTS") != "1",
    reason="Set ZETT_AGENT_LIVE_TESTS=1 to run live gateway checks",
)

BASE_URL = os.getenv("ZETT_AGENT_LIVE_RESPONSES_BASE_URL", "http://192.168.1.11:8787/v1")
MODEL = os.getenv("ZETT_AGENT_LIVE_RESPONSES_MODEL", "deepseek-flash")


def gateway() -> OpenAIProvider:
    """Build the Responses adapter; this gateway ignores the placeholder key."""
    return OpenAIProvider(model=MODEL, api_key="unused", base_url=BASE_URL, response=True)


@tool(guidelines="Use to compute the user's requested sum.")
def add(left: int, right: int) -> int:
    """Add two integer values."""
    return left + right


async def test_live_responses_streams_text() -> None:
    provider = gateway()
    try:
        async with timeout(60):
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
    finally:
        await provider.aclose()


async def test_live_responses_completes_a_local_tool_round_trip() -> None:
    provider = gateway()
    try:
        agent = await Agent.create(
            provider,
            system_prompt="Call add exactly once to compute 2+3, then reply with the tool result only.",
            tools=[add],
            max_iterations=3,
            config=AgentRunConfig(session_id="live-responses"),
        )
        async with timeout(90):
            events = [
                event
                async for event in agent.stream(
                    UserMessage(content="Use add to compute 2+3."),
                    config=AgentRunConfig(session_id="live-responses"),
                    reasoning_effort=ReasoningEffort.OFF,
                )
            ]
        completed = [event for event in events if event.type == AgentEventType.TOOL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].message.content == "5"
        assert events[-1].type == AgentEventType.RUN_COMPLETED
        assert "5" in events[-1].message.content
    finally:
        await provider.aclose()


async def test_live_responses_accepts_a_deferred_local_tool() -> None:
    calls: list[tuple[int, int]] = []

    @tool(deferred=True, guidelines="Use to compute the user's requested sum.")
    def deferred_add(left: int, right: int) -> int:
        """Add two integer values."""
        calls.append((left, right))
        return left + right

    provider = gateway()
    try:
        agent = await Agent.create(
            provider,
            system_prompt="Call deferred_add exactly once to compute 2+3, then reply with the tool result only.",
            tools=[deferred_add],
            max_iterations=3,
            config=AgentRunConfig(session_id="live-responses-deferred"),
        )
        async with timeout(90):
            events = [
                event
                async for event in agent.stream(
                    UserMessage(content="Use deferred_add to compute 2+3."),
                    config=AgentRunConfig(session_id="live-responses-deferred"),
                    reasoning_effort=ReasoningEffort.OFF,
                )
            ]
        assert calls == [(2, 3)], "a deferred tool must still run in the local executor"
        assert events[-1].type == AgentEventType.RUN_COMPLETED
        assert "5" in events[-1].message.content
    finally:
        await provider.aclose()
