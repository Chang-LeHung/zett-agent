"""A signed thinking block that the service refused is dropped, not replayed again.

Claude 5.x binds a thinking block to the system prompt, the tools, and the messages before it. When
one of them changed since the block was written, for example because older turns became a summary,
the service answers ``400 ... The block is bound to a different conversation``. Anthropic's
documented fallback is to leave the blocks out, so the provider drops the signed blocks of the
refused request, sends it once more, keeps them out of later requests, and logs that it did.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest

from zett_agent.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from zett_agent.model import ModelEventType, ModelRequest, ReasoningEffort
from zett_agent.providers import anthropic_compatible
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.anthropic_compatible import to_content_blocks
from zett_agent.providers.base import ProviderHTTPError

MODEL = "claude-sonnet"
BOUND = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different "
    'conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block".'
)
TAMPERED = "messages.1.content.0: Invalid `signature` in `thinking` block."


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


ANSWER = _sse(
    [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
)


def _refusal(message: str = BOUND) -> httpx.Response:
    return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": message}})


def _answer() -> httpx.Response:
    return httpx.Response(200, content=ANSWER, headers={"content-type": "text/event-stream"})


def _signed(signature: str) -> AssistantMessage:
    """An assistant turn that asked for a tool, with the thinking block that came with it."""
    return AssistantMessage(
        tool_calls=(ToolCall("toolu_1", "add", {"a": 1, "b": 2}),),
        provider="anthropic",
        model=MODEL,
        replay_blocks=(
            {"type": "thinking", "thinking": "hm", "signature": signature},
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1, "b": 2}},
        ),
    )


def _history(signature: str = "SIG-A") -> ModelRequest:
    return ModelRequest(
        messages=(
            SystemMessage(content="Be concise."),
            UserMessage(content="Add 1 and 2."),
            _signed(signature),
            ToolMessage(tool_call_id="toolu_1", name="add", content="3"),
        ),
        reasoning_effort=ReasoningEffort.LOW,
    )


def _provider(handler) -> AnthropicProvider:
    return AnthropicProvider(
        model=MODEL, api_key="sk", base_url="https://api.anthropic.com", transport=httpx.MockTransport(handler)
    )


async def _run(provider: AnthropicProvider, request: ModelRequest):
    return [event async for event in provider.stream(request)]


def _types(body: dict[str, Any], index: int) -> list[str]:
    return [block["type"] for block in body["messages"][index]["content"]]


def _refuse_first(bodies: list[dict[str, Any]], message: str = BOUND):
    """A handler that refuses the first request and answers every other one."""

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _refusal(message) if len(bodies) == 1 else _answer()

    return handler


async def test_a_refused_block_is_dropped_and_the_request_is_sent_again(caplog: pytest.LogCaptureFixture) -> None:
    bodies: list[dict[str, Any]] = []

    events = await _run(_provider(_refuse_first(bodies)), _history())

    assert len(bodies) == 2
    assert _types(bodies[0], 1) == ["thinking", "tool_use"]
    assert _types(bodies[1], 1) == ["tool_use"]
    assert [event.response.message.content for event in events if event.type == ModelEventType.RESPONSE] == ["Done."]
    # The model loses its earlier reasoning from here on, so the recovery is not silent.
    [record] = [r for r in caplog.records if r.name.endswith("anthropic_compatible")]
    assert record.levelname == "WARNING" and "refused 1 signed thinking block" in record.getMessage().replace("(s)", "")


async def test_a_forgotten_block_is_not_sent_by_later_requests() -> None:
    bodies: list[dict[str, Any]] = []
    provider = _provider(_refuse_first(bodies))

    await _run(provider, _history())
    await _run(provider, _history())

    # Request 1 is refused, request 2 is its retry, and request 3 never carries the block.
    assert len(bodies) == 3
    assert _types(bodies[2], 1) == ["tool_use"]


async def test_a_block_the_model_writes_afterwards_is_sent_as_usual() -> None:
    bodies: list[dict[str, Any]] = []
    provider = _provider(_refuse_first(bodies))

    await _run(provider, _history("SIG-OLD"))
    await _run(provider, _history("SIG-NEW"))

    assert _types(bodies[2], 1) == ["thinking", "tool_use"]
    assert bodies[2]["messages"][1]["content"][0]["signature"] == "SIG-NEW"


@pytest.mark.parametrize("message", [TAMPERED, "messages.0.content: Input should be a valid list"])
async def test_only_a_block_bound_to_another_conversation_is_retried(message: str) -> None:
    """A tampered signature is another failure, and dropping blocks does not cure it."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _refusal(message)

    with pytest.raises(ProviderHTTPError):
        await _run(_provider(handler), _history())
    assert calls == 1


async def test_a_refusal_without_any_signed_block_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _refusal()

    request = ModelRequest(messages=(UserMessage(content="Hi."),), reasoning_effort=ReasoningEffort.LOW)
    with pytest.raises(ProviderHTTPError):
        await _run(_provider(handler), request)
    assert calls == 1


async def test_the_request_is_sent_only_once_more() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _refusal()

    with pytest.raises(ProviderHTTPError):
        await _run(_provider(handler), _history())
    assert calls == 2


async def test_a_redacted_thinking_block_is_forgotten_by_its_data() -> None:
    bodies: list[dict[str, Any]] = []
    request = ModelRequest(
        messages=(
            UserMessage(content="Add 1 and 2."),
            AssistantMessage(
                tool_calls=(ToolCall("toolu_1", "add", {}),),
                provider="anthropic",
                model=MODEL,
                replay_blocks=(
                    {"type": "redacted_thinking", "data": "OPAQUE"},
                    {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}},
                ),
            ),
            ToolMessage(tool_call_id="toolu_1", name="add", content="3"),
        ),
        reasoning_effort=ReasoningEffort.LOW,
    )

    await _run(_provider(_refuse_first(bodies)), request)

    assert _types(bodies[0], 1) == ["redacted_thinking", "tool_use"]
    assert _types(bodies[1], 1) == ["tool_use"]


def test_blocks_listed_as_stale_are_left_out_of_a_replay() -> None:
    message = _signed("SIG-A")
    kept = to_content_blocks(message, model=MODEL, stale={"SIG-B"})
    left_out = to_content_blocks(message, model=MODEL, stale={"SIG-A"})
    assert [block["type"] for block in kept] == ["thinking", "tool_use"]
    assert [block["type"] for block in left_out] == ["tool_use"]


def test_the_memory_of_refused_blocks_is_bounded() -> None:
    provider = _provider(lambda request: _answer())
    limit = anthropic_compatible._STALE_LIMIT
    for index in range(limit + 10):
        provider._stale_thinking[f"SIG-{index}"] = None
    error = _FakeStatusError(400, BOUND)
    body = {"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "NEW"}]}]}

    assert provider._forget_thinking(error, body) is True  # type: ignore[arg-type]  # a stand-in for the SDK error
    assert len(provider._stale_thinking) == limit
    assert "NEW" in provider._stale_thinking and "SIG-0" not in provider._stale_thinking


class _FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _BindingService:
    """A stand-in for a service that binds each thinking block to the conversation before it."""

    def __init__(self) -> None:
        self.refused = 0
        self.bodies: list[dict[str, Any]] = []

    @staticmethod
    def _prefix(body: dict[str, Any], upto: int) -> str:
        def plain(value: Any) -> Any:
            # Cache markers move from request to request and say nothing about the conversation.
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items() if key != "cache_control"}
            if isinstance(value, list):
                return [plain(item) for item in value]
            return value

        prefix = plain([body.get("system"), body.get("tools"), body["messages"][:upto]])
        return hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        for index, message in enumerate(body["messages"]):
            for block in message["content"] if isinstance(message["content"], list) else []:
                if block["type"] == "thinking" and block["signature"] != self._prefix(body, index):
                    self.refused += 1
                    return _refusal()
        events = [
            {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 5}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": self._prefix(body, len(body["messages"]))},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Done."}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
            {"type": "message_stop"},
        ]
        return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})


async def test_a_conversation_that_was_compacted_keeps_working() -> None:
    """Keep-tail compaction: older turns become a summary and the recent turns keep their thinking."""
    service = _BindingService()
    provider = _provider(service)
    system = SystemMessage(content="Be concise.")
    question = UserMessage(content="Add 1 and 2.")

    first = ModelRequest(messages=(system, question), reasoning_effort=ReasoningEffort.LOW)
    [response] = [e.response for e in await _run(provider, first) if e.type == ModelEventType.RESPONSE]
    answer = response.message
    assert service.refused == 0 and answer.replay_blocks[0]["type"] == "thinking"

    # Unchanged history: the signed block is accepted and sent back.
    follow = UserMessage(content="And 3 plus 4?")
    await _run(
        provider, ModelRequest(messages=(system, question, answer, follow), reasoning_effort=ReasoningEffort.LOW)
    )
    assert service.refused == 0 and _types(service.bodies[-1], 1) == ["thinking", "text"]

    # The turn before `question` became a summary, so the block no longer matches what came before it.
    summary = UserMessage(content="[Conversation checkpoint]\nEarlier work, summarized.")
    compacted = ModelRequest(messages=(system, summary, question, answer, follow), reasoning_effort=ReasoningEffort.LOW)
    events = await _run(provider, compacted)

    assert service.refused == 1
    assert any(event.type == ModelEventType.RESPONSE for event in events)
    assert _types(service.bodies[-1], 2) == ["text"]

    # The next request of the same conversation does not hit the refusal again.
    await _run(provider, compacted)
    assert service.refused == 1
