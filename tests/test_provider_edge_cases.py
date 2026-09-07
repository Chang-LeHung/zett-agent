import json

import httpx
import pytest

from zett_agent import (
    AnthropicProvider,
    AssistantMessage,
    DeepSeekProvider,
    GoogleProvider,
    ImageBytesSource,
    ImageContent,
    ImageUrlSource,
    ModelEventType,
    ModelRequest,
    OllamaProvider,
    OpenAIProvider,
    ProviderAuthError,
    ProviderResponseError,
    ReasoningEffort,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    ToolMessage,
    UserMessage,
)
from zett_agent.model import ModelEvent
from zett_agent.providers.anthropic import _to_anthropic_image_block
from zett_agent.providers.base import (
    _normalize_image_source,
    _parse_tool_calls,
    _reasoning_effort_to_budget,
    _to_model_data,
    _ToolCallAccumulator,
)


async def _collect(stream):
    return [event async for event in stream]


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (ReasoningEffort.OFF, 0),
        (ReasoningEffort.MINIMAL, 2048),
        (ReasoningEffort.LOW, 4096),
        (ReasoningEffort.MEDIUM, 8192),
        (ReasoningEffort.HIGH, 16384),
        (ReasoningEffort.XHIGH, 32768),
    ],
)
def test_reasoning_effort_budget_mapping_is_exhaustive(effort, budget):
    assert _reasoning_effort_to_budget(effort) == budget


def test_provider_data_normalization_supports_sdk_and_fallback_shapes():
    class ModernSDKValue:
        def model_dump(self, *, exclude_none):
            assert exclude_none is True
            return {"kind": "modern"}

    class LegacySDKValue:
        def dict(self):
            return {"kind": "legacy"}

    assert _to_model_data({"kind": "mapping"}) == {"kind": "mapping"}
    assert _to_model_data(ModernSDKValue()) == {"kind": "modern"}
    assert _to_model_data(LegacySDKValue()) == {"kind": "legacy"}
    assert _to_model_data(SystemMessage(content="instruction", attributes={"private": True})) == {
        "role": "system",
        "content": "instruction",
    }
    with pytest.raises(ProviderResponseError, match="did not normalize to an object"):
        _to_model_data(3)


def test_image_normalizers_reject_unsupported_or_malformed_sources():
    with pytest.raises(ProviderResponseError, match="Unsupported image source"):
        _normalize_image_source(object())
    with pytest.raises(ProviderResponseError, match="must contain base64"):
        _to_anthropic_image_block(ImageUrlSource("data:image/png,not-base64"))
    assert _to_anthropic_image_block(ImageBytesSource(b"image", "image/png")) == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
    }
    with pytest.raises(ProviderResponseError, match="Unsupported image content source"):
        _to_anthropic_image_block(object())


def test_tool_call_parser_orders_calls_generates_ids_and_rejects_missing_names():
    later = _ToolCallAccumulator(index=2, name="write", argument_buffer='{"path":"a"}')
    earlier = _ToolCallAccumulator(index=0, call_id="provider-id", name="read")
    assert _parse_tool_calls({2: later, 0: earlier}) == (
        ToolCall("provider-id", "read", {}),
        ToolCall("call_2", "write", {"path": "a"}),
    )

    with pytest.raises(ProviderResponseError, match="did not provide a name"):
        _parse_tool_calls({0: _ToolCallAccumulator(index=0, argument_buffer="{}")})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OpenAIProvider("model", ""),
        lambda: DeepSeekProvider("model", ""),
        lambda: AnthropicProvider("model", ""),
        lambda: GoogleProvider("model", ""),
    ],
)
def test_remote_providers_reject_empty_api_keys_before_creating_clients(factory):
    with pytest.raises(ValueError, match="api_key is required"):
        factory()


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (401, ProviderAuthError, "credential"),
        (400, ProviderResponseError, "invalid stream payload"),
        (422, ProviderResponseError, "invalid stream payload"),
        (500, ProviderResponseError, "invalid stream payload"),
        (429, ProviderResponseError, "stream request failed"),
    ],
)
async def test_openai_style_provider_maps_http_failures(status, error_type, message):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": "provider failure", "type": "request_error"}},
        )

    provider = OpenAIProvider("model", "key", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(error_type, match=message):
            await _collect(provider.stream(ModelRequest((UserMessage(content="hello"),))))
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, ProviderAuthError), (429, ProviderResponseError)],
)
async def test_anthropic_provider_maps_http_failures(status, error_type):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"type": "error", "error": {"type": "request_error", "message": "provider failure"}},
            headers={"request-id": "request-1"},
        )

    provider = AnthropicProvider("model", "key", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(error_type):
            await _collect(provider.stream(ModelRequest((UserMessage(content="hello"),))))
    finally:
        await provider.aclose()


async def test_openai_request_supports_temperature_forced_tool_and_no_tool_omission():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        body = b'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("model", "key", transport=httpx.MockTransport(handler), temperature=0.25)
    request = ModelRequest(
        (UserMessage(content="hello"),),
        tools=(ToolDefinition("extract", "Extract data", {"type": "object"}),),
        reasoning_effort=ReasoningEffort.OFF,
        tool_choice="extract",
    )
    try:
        events = await _collect(provider.stream(request))
    finally:
        await provider.aclose()

    assert captured["tools"][0]["function"]["name"] == "extract"
    assert "reasoning_effort" not in captured
    assert captured["temperature"] == 0.25
    assert captured["tool_choice"] == {"type": "function", "function": {"name": "extract"}}
    assert events[-1].type == ModelEventType.RESPONSE


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (ReasoningEffort.MINIMAL, "low"),
        (ReasoningEffort.HIGH, "high"),
        (ReasoningEffort.XHIGH, "max"),
    ],
)
def test_deepseek_v4_maps_reasoning_effort_levels(effort, expected):
    provider = object.__new__(DeepSeekProvider)
    provider.model = "deepseek-v4-flash"
    request = ModelRequest((UserMessage(content="hello"),), reasoning_effort=effort)
    assert provider._provider_specific_request_fields(request) == {"reasoning_effort": expected}


async def test_google_rejects_a_stream_without_a_finish_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        body = b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = GoogleProvider("gemini", "key", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProviderResponseError, match="without a finish reason"):
            await _collect(provider.stream(ModelRequest((UserMessage(content="hello"),))))
    finally:
        await provider.aclose()


async def test_google_serializes_multimodal_replay_and_tool_history():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        body = (
            b'data: {"candidates":[{"content":{"role":"model","parts":'
            b'[{"text":"thinking","thought":true},{"text":"answer"}]},"finishReason":"STOP"}],'
            b'"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":2,'
            b'"thoughtsTokenCount":3,"cachedContentTokenCount":4}}\n\n'
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    request = ModelRequest(
        (
            UserMessage(
                content=[
                    TextContent("inspect"),
                    ImageContent(ImageBytesSource(b"image", "image/png")),
                    ImageContent(ImageUrlSource("data:image/png;base64,aW1hZ2U=")),
                    ImageContent(ImageUrlSource("https://example.com/image.png")),
                ]
            ),
            AssistantMessage(
                content="working",
                tool_calls=(ToolCall("call-1", "read", {"path": "a.txt"}),),
            ),
            ToolMessage(tool_call_id="call-1", name="read", content="contents"),
            AssistantMessage(provider="google", model="gemini", replay_blocks=({"text": "signed replay"},)),
        ),
        tools=(ToolDefinition("read", "Read a file", {"type": "object"}),),
        tool_choice="read",
    )
    provider = GoogleProvider("gemini", "key", transport=httpx.MockTransport(handler))
    try:
        events = await _collect(provider.stream(request))
    finally:
        await provider.aclose()

    contents = captured["contents"]
    assert contents[0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert contents[0]["parts"][2]["inlineData"]["data"] == "aW1hZ2U="
    assert contents[0]["parts"][3]["fileData"]["fileUri"] == "https://example.com/image.png"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][1]["functionCall"]["name"] == "read"
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "read"
    assert contents[3]["parts"] == [{"text": "signed replay"}]
    assert [event.delta for event in events if event.type == ModelEventType.REASONING_DELTA] == ["thinking"]
    response = events[-1].response
    assert response.message.content == "answer"
    assert response.message.reasoning == "thinking"
    assert response.usage.cache_read_tokens == 4


async def test_ollama_rejects_external_images_and_non_terminal_streams():
    provider = OllamaProvider(
        "model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b'{"message":{"role":"assistant","content":"partial"},"done":false}\n',
                headers={"content-type": "application/x-ndjson"},
            )
        ),
    )
    external_image = UserMessage(content=[ImageContent(ImageUrlSource("https://example.com/image.png"))])
    try:
        with pytest.raises(ProviderResponseError, match="images must be bytes"):
            await _collect(provider.stream(ModelRequest((external_image,))))
        with pytest.raises(ProviderResponseError, match="without a terminal response"):
            await _collect(provider.stream(ModelRequest((UserMessage(content="hello"),))))
    finally:
        await provider.aclose()


async def test_ollama_serializes_images_assistant_tools_and_forced_tool_guidance():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        body = (
            b'{"message":{"role":"assistant","thinking":"think","content":"answer"},"done":false}\n'
            b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",'
            b'"prompt_eval_count":4,"eval_count":2}\n'
        )
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    request = ModelRequest(
        (
            SystemMessage(content="system"),
            UserMessage(
                content=[
                    TextContent("inspect"),
                    ImageContent(ImageBytesSource(b"image", "image/png")),
                    ImageContent(ImageUrlSource("data:image/png;base64,aW1hZ2U=")),
                ]
            ),
            AssistantMessage(
                content="working",
                reasoning="reasoning",
                tool_calls=(ToolCall("call-1", "read", {"path": "a.txt"}),),
            ),
            ToolMessage(tool_call_id="call-1", name="read", content="contents"),
        ),
        tools=(ToolDefinition("read", "Read a file", {"type": "object"}),),
        tool_choice="read",
    )
    provider = OllamaProvider("model", transport=httpx.MockTransport(handler))
    try:
        events = await _collect(provider.stream(request))
    finally:
        await provider.aclose()

    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": "Respond only by calling read."}
    assert messages[2]["images"] == ["aW1hZ2U=", "aW1hZ2U="]
    assert messages[3]["thinking"] == "reasoning"
    assert messages[3]["tool_calls"][0]["function"]["name"] == "read"
    assert messages[4] == {"role": "tool", "content": "contents", "tool_name": "read"}
    assert [event.delta for event in events if event.type == ModelEventType.REASONING_DELTA] == ["think"]
    assert events[-1].response.message.content == "answer"


def test_tool_call_delta_rejects_invalid_indices_and_empty_fragments():
    with pytest.raises(ValueError, match="cannot be negative"):
        ToolCallDelta(index=-1, name_delta="read")
    with pytest.raises(ValueError, match="must contain"):
        ToolCallDelta(index=0)


def test_model_event_factories_preserve_payloads():
    delta = ToolCallDelta(index=0, name_delta="read")
    assert ModelEvent.text("text").delta == "text"
    assert ModelEvent.reasoning("reasoning").delta == "reasoning"
    assert ModelEvent.tool_call(delta).tool_call_delta is delta
