"""Regression checks for replay fidelity, cache usage, and unbuffered streams."""

import asyncio
import json

import httpx

from zett_agent.messages import (
    AssistantMessage,
    ImageContent,
    ImageDetail,
    ImageUrlSource,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from zett_agent.model import (
    ModelRequest,
    ReasoningEffort,
)
from zett_agent.providers.anthropic import _to_anthropic_content_blocks
from zett_agent.providers.base import (
    _message_to_openai_payload,
    _usage_from_mapping,
)
from zett_agent.providers.deepseek import DeepSeekProvider
from zett_agent.providers.ollama import OllamaProvider
from zett_agent.providers.openai import OpenAIProvider


def test_tool_arguments_are_serializable_in_both_protocols() -> None:
    message = AssistantMessage(tool_calls=(ToolCall("call-1", "add", {"left": 2}),))
    openai = _message_to_openai_payload(message)
    anthropic = _to_anthropic_content_blocks(message)
    assert json.loads(openai["tool_calls"][0]["function"]["arguments"]) == {"left": 2}
    assert json.loads(json.dumps(anthropic))[0]["input"] == {"left": 2}


def test_model_messages_have_no_extension_context_fields_to_send_to_providers() -> None:
    message = UserMessage(content="Visible content")
    assert _message_to_openai_payload(message) == {"role": "user", "content": "Visible content"}
    assert _to_anthropic_content_blocks(message) == [{"type": "text", "text": "Visible content"}]
    assert not hasattr(message, "metadata")
    assert not hasattr(message, "tags")


def test_anthropic_signed_thinking_and_redacted_blocks_round_trip() -> None:
    blocks = (
        {"type": "thinking", "thinking": "Provider reasoning", "signature": "opaque-signature"},
        {"type": "redacted_thinking", "data": "opaque-data"},
        {"type": "tool_use", "id": "call-1", "name": "add", "input": {}},
    )
    message = AssistantMessage(
        tool_calls=(ToolCall("call-1", "add"),), provider="anthropic", model="claude", replay_blocks=blocks
    )
    assert _to_anthropic_content_blocks(message, model="claude") == list(blocks)
    assert _to_anthropic_content_blocks(message, model="other")[0]["type"] == "tool_use"
    result = ToolMessage(content="failed", tool_call_id="call-1", name="add", success=False)
    assert _to_anthropic_content_blocks(result)[0]["is_error"] is True


def test_image_detail_and_deepseek_cache_usage_survive_normalization() -> None:
    message = UserMessage(content=[ImageContent(ImageUrlSource("https://example.com/a.png"), ImageDetail.HIGH)])
    assert _message_to_openai_payload(message)["content"][0]["image_url"]["detail"] == "high"
    usage = _usage_from_mapping({"prompt_tokens": 100, "prompt_cache_hit_tokens": 80, "completion_tokens": 10})
    assert usage.cache_read_tokens == 80
    assert usage.total_tokens == 110


async def test_deepseek_replays_reasoning_and_uses_custom_endpoint() -> None:
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        assert str(request.url) == "https://gateway.example/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=('data: {"choices":[{"delta":{"content":"5"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
        )

    provider = DeepSeekProvider(
        "deepseek-v4-flash", "test", base_url="https://gateway.example/v1", transport=httpx.MockTransport(handler)
    )
    messages = (
        SystemMessage(content="Base"),
        AssistantMessage(reasoning="Retain this", tool_calls=(ToolCall("c", "add", {"a": 2}),)),
        ToolMessage(content="5", tool_call_id="c", name="add"),
    )
    try:
        _ = [event async for event in provider.stream(ModelRequest(messages, reasoning_effort=ReasoningEffort.LOW))]
        assert captured["messages"][1]["reasoning_content"] == "Retain this"
        assert captured["reasoning_effort"] == "low"
    finally:
        await provider.aclose()


async def test_openai_compatible_provider_replays_reasoning_content() -> None:
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=('data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
        )

    provider = OpenAIProvider(
        "reasoner",
        "test",
        base_url="https://gateway.example/v1",
        transport=httpx.MockTransport(handler),
    )
    messages = (
        AssistantMessage(reasoning="Retain this", provider="openai", model="reasoner"),
        UserMessage(content="Continue"),
    )
    try:
        _ = [event async for event in provider.stream(ModelRequest(messages))]
        assert captured["messages"][0]["reasoning_content"] == "Retain this"
    finally:
        await provider.aclose()


async def test_ollama_emits_first_delta_before_response_finishes_and_closes_on_cancel() -> None:
    class DelayedBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"message":{"role":"assistant","content":"first"},"done":false}\n'
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    body = DelayedBody()
    provider = OllamaProvider("test", transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body)))
    stream = provider.stream(ModelRequest((UserMessage(content="hi"),)))
    try:
        first = await asyncio.wait_for(anext(stream), timeout=2)
        assert first.delta == "first"
    finally:
        await stream.aclose()
        await provider.aclose()
    assert body.closed
