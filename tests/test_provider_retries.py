"""Count real SDK HTTP attempts through isolated mock transports."""

import asyncio
import inspect
import json

import httpx
import pytest
from google.genai.errors import APIError as GoogleAPIError
from ollama import ResponseError as OllamaResponseError

from zett_agent import (
    AnthropicProvider,
    DeepSeekProvider,
    GoogleProvider,
    ModelEvent,
    ModelRequest,
    OllamaProvider,
    OpenAIProvider,
    RetryOptions,
    UserMessage,
)
from zett_agent.providers.base import ProviderError, RetryingProvider, retry_model_stream

PROVIDERS = [OpenAIProvider, DeepSeekProvider, AnthropicProvider, GoogleProvider, OllamaProvider]


@pytest.mark.parametrize("cls", PROVIDERS)
def test_provider_exposes_an_async_generator_named_stream(cls):
    assert cls.stream.__name__ == "stream"
    assert inspect.isasyncgenfunction(cls.stream)


def make_provider(cls, handler, retry=2):
    kwargs = {"model": "test", "transport": httpx.MockTransport(handler), "retry": RetryOptions(max_retries=retry)}
    if cls is not OllamaProvider:
        kwargs["api_key"] = "test-key"
    return cls(**kwargs)


def output(cls, complete=True):
    if cls in (OpenAIProvider, DeepSeekProvider):
        events = [{"choices": [{"delta": {"content": "ok"}}]}]
        if complete:
            events += [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    if cls is AnthropicProvider:
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "test",
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        ]
        if complete:
            events += [{"type": "message_stop"}]
        return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
    if cls is GoogleProvider:
        candidate = {"content": {"role": "model", "parts": [{"text": "ok"}]}}
        if complete:
            candidate["finishReason"] = "STOP"
        return f"data: {json.dumps({'candidates': [candidate]})}\n\n".encode()
    return (
        json.dumps({"model": "test", "message": {"role": "assistant", "content": "ok"}, "done": complete}) + "\n"
    ).encode()


@pytest.mark.parametrize("cls", PROVIDERS)
def test_sync_provider_stream_reuses_transport_and_retries(cls):
    """Exercise every official SDK through the blocking facade, without network."""
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"code": 503, "message": "temporary"}})
        return httpx.Response(200, content=output(cls), headers={"content-type": "text/event-stream"})

    provider = make_provider(cls, handler, retry=1)
    provider.retry = RetryOptions(max_retries=1, base_delay=0, max_delay=0)
    with provider.sync() as blocking:
        try:
            request = ModelRequest([UserMessage(content="hello")])
            for _ in range(2):
                with blocking.stream(request) as stream:
                    events = list(stream)
                assert events[-1].response.message.content == "ok"
            assert calls == 3
        finally:
            blocking.aclose()


@pytest.fixture
def no_backoff(monkeypatch):
    async def sleep(seconds):
        pass

    monkeypatch.setattr("zett_agent.providers.base.asyncio.sleep", sleep)


@pytest.mark.parametrize("cls", PROVIDERS)
@pytest.mark.parametrize("status", [429, 503])
async def test_transient_http_errors_retry_exactly_once_then_succeed(cls, status, no_backoff):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, json={"error": {"code": status, "message": "temporary"}})
        return httpx.Response(200, content=output(cls), headers={"content-type": "text/event-stream"})

    provider = make_provider(cls, handler, retry=1)
    try:
        events = [event async for event in provider.stream(ModelRequest([UserMessage(content="hi")]))]
        assert calls == 2
        assert events[-1].response.message.content == "ok"
        assert events[-1].response.message.model == "test"
        assert sum(event.delta == "ok" for event in events) == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize("cls", PROVIDERS)
@pytest.mark.parametrize("retry,expected", [(2, 3), (0, 1), (1, 2)])
async def test_retry_exhaustion_uses_model_configuration(cls, retry, expected, no_backoff):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"code": 503, "message": "temporary"}})

    provider = make_provider(cls, handler, retry=retry)
    try:
        with pytest.raises((ProviderError, GoogleAPIError, OllamaResponseError)):
            _ = [event async for event in provider.stream(ModelRequest([UserMessage(content="hi")]))]
        assert calls == expected
    finally:
        await provider.aclose()


@pytest.mark.parametrize("cls", PROVIDERS)
async def test_auth_errors_do_not_retry(cls, no_backoff):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"code": 401, "message": "unauthorized"}})

    provider = make_provider(cls, handler)
    try:
        with pytest.raises((ProviderError, GoogleAPIError, OllamaResponseError)):
            _ = [event async for event in provider.stream(ModelRequest([UserMessage(content="hi")]))]
        assert calls == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize("cls", PROVIDERS)
async def test_connection_error_before_output_can_retry(cls, no_backoff):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary connection failure", request=request)
        return httpx.Response(200, content=output(cls), headers={"content-type": "text/event-stream"})

    provider = make_provider(cls, handler, retry=1)
    try:
        events = [event async for event in provider.stream(ModelRequest([UserMessage(content="hi")]))]
        assert events[-1].response.message.content == "ok"
        assert calls == 2
    finally:
        await provider.aclose()


@pytest.mark.parametrize("cls", PROVIDERS)
async def test_partial_stream_failure_never_replays_visible_output(cls, no_backoff):
    calls = 0

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield output(cls, complete=False)
            raise httpx.ReadError("connection lost")

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=BrokenStream(), headers={"content-type": "text/event-stream"})

    provider = make_provider(cls, handler)
    events = []
    try:
        with pytest.raises(httpx.TransportError):
            async for event in provider.stream(ModelRequest([UserMessage(content="hi")])):
                events.append(event)
        assert any(event.delta == "ok" for event in events)
        assert calls == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize("retry", [-1, True, 1.5, "2"])
@pytest.mark.parametrize("cls", PROVIDERS)
def test_invalid_retry_is_rejected(cls, retry):
    with pytest.raises(ValueError, match="non-negative integer"):
        make_provider(cls, lambda request: None, retry=retry)


def test_model_request_does_not_configure_retry():
    with pytest.raises(TypeError, match="retry"):
        ModelRequest([], retry=1)


@pytest.mark.parametrize("field", ["base_delay", "max_delay"])
@pytest.mark.parametrize("value", [-1, True, "1", None, float("inf"), float("nan")])
def test_invalid_backoff_delay(field, value):
    with pytest.raises(ValueError, match=field):
        RetryOptions(**{field: value})


@pytest.mark.parametrize("cls", PROVIDERS)
def test_provider_requires_retry_options(cls):
    with pytest.raises(ValueError, match="RetryOptions"):
        kwargs = {"model": "test", "retry": 2}
        if cls is not OllamaProvider:
            kwargs["api_key"] = "test-key"
        cls(**kwargs)


@pytest.mark.parametrize("cls", PROVIDERS)
@pytest.mark.parametrize(
    "options,expected",
    [
        (RetryOptions(), [0.25, 0.5, 1.0]),
        (RetryOptions(base_delay=1, max_delay=5, max_retries=5), [1, 2, 4, 5, 5]),
        (RetryOptions(base_delay=10, max_delay=3, max_retries=2), [3, 3]),
        (RetryOptions(base_delay=0, max_retries=2), [0, 0]),
        (RetryOptions(max_delay=0, max_retries=2), [0, 0]),
        (RetryOptions(max_retries=0), []),
        (RetryOptions(base_delay=1e308, max_delay=1.7e308, max_retries=3), [1e308, 1.7e308, 1.7e308]),
    ],
)
async def test_all_providers_apply_bounded_exponential_backoff(cls, options, expected, monkeypatch):
    delays = []
    calls = 0

    async def sleep(seconds):
        delays.append(seconds)

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "unavailable", "type": "server_error"}})

    monkeypatch.setattr("zett_agent.providers.base.asyncio.sleep", sleep)
    provider = make_provider(cls, handler)
    provider.retry = options
    try:
        for _ in range(2):
            delays.clear()
            calls = 0
            with pytest.raises((ProviderError, GoogleAPIError, OllamaResponseError)):
                _ = [event async for event in provider.stream(ModelRequest([UserMessage(content="hello")]))]
            assert delays == expected
            assert calls == options.max_retries + 1
    finally:
        await provider.aclose()


async def test_cancellation_during_backoff_stops_retry(monkeypatch):
    entered = asyncio.Event()

    class Failing(RetryingProvider):
        calls = 0

        @retry_model_stream
        async def stream(self, request):
            self.calls += 1
            raise httpx.ConnectError("offline")
            yield ModelEvent.text("unused")

    async def sleep(seconds):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("zett_agent.providers.base.asyncio.sleep", sleep)
    provider = Failing()

    async def collect():
        return [event async for event in provider.stream(ModelRequest([]))]

    task = asyncio.create_task(collect())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.calls == 1
