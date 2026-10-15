"""A provider that goes silent is given up on, and the error says which wait ran out.

These use a real TCP server that misbehaves in the ways a stuck gateway does: it takes the
request and never answers, it starts a response and stops, or it is merely slow. The limits
are fractions of a second so the suite stays fast; the defaults are minutes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.messages import UserMessage
from zett_agent.model import DEFAULT_TIMEOUT_OPTIONS, ModelRequest, RetryOptions, TimeoutOptions
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.base import ProviderTimeoutError, RetryingProvider, guard_idle, timeout_error
from zett_agent.providers.deepseek import DeepSeekProvider
from zett_agent.providers.google import GoogleProvider
from zett_agent.providers.ollama import OLLAMA_TIMEOUT_OPTIONS, OllamaProvider
from zett_agent.providers.openai import OpenAIProvider

REQUEST = ModelRequest(messages=(UserMessage(content="hi"),))
FAST = TimeoutOptions(connect=1.0, first_byte=0.6, idle=0.6)
RETRIES = RetryOptions(max_retries=3, base_delay=0.01, max_delay=0.02)


# ------------------------------------------------------------------- a bad server


def chunked(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class Server:
    """Answer each connection as one of a few misbehaving servers, and count the connections.

    Modes: ``silent`` takes the request and never answers; ``stall`` sends the response headers
    and one chunk, then nothing; ``slow`` waits ``delay`` seconds and then answers fully;
    ``trickle`` answers fully but sends one event every ``delay`` seconds.
    """

    def __init__(self, mode: str, *, head: str, first: bytes, events: list[bytes], tail: bytes = b"", delay: float = 0):
        self.mode, self.head, self.first, self.events, self.tail, self.delay = mode, head, first, events, tail, delay
        self.connections = 0
        self.url = ""
        self._server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> Server:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            length = next(
                (
                    int(line.split(b":")[1])
                    for line in request.split(b"\r\n")
                    if line.lower().startswith(b"content-length")
                ),
                0,
            )
            await reader.readexactly(length)
            if self.mode == "silent":
                await asyncio.sleep(3600)
            if self.mode == "slow":
                await asyncio.sleep(self.delay)
            writer.write(self.head.encode())
            if self.mode == "stall":
                writer.write(chunked(self.first))
                await writer.drain()
                await asyncio.sleep(3600)
            for event in self.events:
                writer.write(chunked(event))
                await writer.drain()
                if self.mode == "trickle":
                    await asyncio.sleep(self.delay)
            writer.write(chunked(self.tail) if self.tail else b"")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()


def sse_head(kind: str = "text/event-stream") -> str:
    return f"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nTransfer-Encoding: chunked\r\n\r\n"


def dumps(value: object) -> bytes:
    return json.dumps(value).encode()


# What each protocol sends for "hi", split into the first piece and the rest.
CHAT_FIRST = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
CHAT_REST = [b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n', b"data: [DONE]\n\n"]
RESPONSES_FIRST = (
    b"event: response.output_text.delta\ndata: "
    + dumps(
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "m",
            "output_index": 0,
            "content_index": 0,
            "delta": "hi",
        }
    )
    + b"\n\n"
)
ANTHROPIC_FIRST = (
    b"event: message_start\ndata: "
    + dumps({"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}})
    + b"\n\n"
    b"event: content_block_start\ndata: "
    + dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    + b"\n\n"
    b"event: content_block_delta\ndata: "
    + dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}})
    + b"\n\n"
)
ANTHROPIC_REST = [
    b"event: message_delta\ndata: "
    + dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}})
    + b"\n\n",
    b"event: message_stop\ndata: " + dumps({"type": "message_stop"}) + b"\n\n",
]
OLLAMA_FIRST = (
    dumps({"model": "m", "created_at": "t", "message": {"role": "assistant", "content": "hi"}, "done": False}) + b"\n"
)
OLLAMA_REST = [
    dumps(
        {
            "model": "m",
            "created_at": "t",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        }
    )
    + b"\n"
]
GOOGLE_FIRST = (
    b"data: "
    + dumps({"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "index": 0}]})
    + b"\r\n\r\n"
)
GOOGLE_REST = [
    b"data: "
    + dumps({"candidates": [{"content": {"role": "model", "parts": []}, "index": 0, "finishReason": "STOP"}]})
    + b"\r\n\r\n"
]
RESPONSES_REST = [
    b"event: response.completed\ndata: "
    + dumps(
        {
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "r", "created_at": 0, "model": "m", "object": "response", "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [], "status": "completed",
                "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
            },
        }
    )
    + b"\n\n"
]  # fmt: skip

#: name -> (maker, first piece, remaining pieces, content type of the response)
PROTOCOLS = {
    "chat": (lambda url, **o: OpenAIProvider("m", "k", base_url=url + "/v1", **o), CHAT_FIRST, CHAT_REST, "text/event-stream"),
    "responses": (lambda url, **o: OpenAIProvider("m", "k", base_url=url + "/v1", response=True, **o), RESPONSES_FIRST, RESPONSES_REST, "text/event-stream"),
    "deepseek": (lambda url, **o: DeepSeekProvider("m", "k", base_url=url, **o), CHAT_FIRST, CHAT_REST, "text/event-stream"),
    "anthropic": (lambda url, **o: AnthropicProvider("m", "k", base_url=url, **o), ANTHROPIC_FIRST, ANTHROPIC_REST, "text/event-stream"),
    "ollama": (lambda url, **o: OllamaProvider("m", base_url=url, **o), OLLAMA_FIRST, OLLAMA_REST, "application/x-ndjson"),
    "google": (lambda url, **o: GoogleProvider("gemini-2.5-flash", api_key="k", base_url=url, **o), GOOGLE_FIRST, GOOGLE_REST, "text/event-stream"),
}  # fmt: skip


async def collect(provider) -> list:
    return [event async for event in provider.stream(REQUEST)]


# ------------------------------------------------------------------------ options


def test_the_defaults_wait_for_headers_for_a_minute_and_for_silence_for_ten():
    assert (DEFAULT_TIMEOUT_OPTIONS.connect, DEFAULT_TIMEOUT_OPTIONS.first_byte, DEFAULT_TIMEOUT_OPTIONS.idle) == (
        5.0,
        60.0,
        600.0,
    )
    # A local model may have to be loaded from disk before it answers.
    assert OLLAMA_TIMEOUT_OPTIONS.first_byte == 600.0 and OLLAMA_TIMEOUT_OPTIONS.connect == 5.0


@pytest.mark.parametrize(
    "bad",
    [
        {"connect": 0},
        {"idle": -1},
        {"first_byte": True},
        {"idle": "5"},
        {"connect": float("inf")},
        {"first_byte": float("nan")},
    ],
)
def test_nonsense_limits_are_rejected(bad):
    with pytest.raises(ValueError):
        TimeoutOptions(**bad)


def test_a_limit_can_be_switched_off_with_none():
    options = TimeoutOptions(connect=None, first_byte=None, idle=None)

    assert (options.connect, options.first_byte, options.idle) == (None, None, None)


@pytest.mark.parametrize("name", list(PROTOCOLS))
def test_every_provider_takes_the_option_and_rejects_anything_else(name):
    make = PROTOCOLS[name][0]
    provider = make("http://127.0.0.1:9", timeout=FAST)

    assert provider.timeout == FAST
    with pytest.raises(ValueError, match="TimeoutOptions"):
        make("http://127.0.0.1:9", timeout=30)


# ------------------------------------------------------- a provider that is silent


@pytest.mark.parametrize("name", list(PROTOCOLS))
async def test_a_provider_that_never_answers_is_given_up_on_and_asked_once(name):
    make, first, rest, kind = PROTOCOLS[name]
    async with Server("silent", head=sse_head(kind), first=first, events=rest) as server:
        provider = make(server.url, retry=RETRIES, timeout=FAST)
        started = time.monotonic()
        try:
            with pytest.raises(ProviderTimeoutError) as raised:
                await collect(provider)
        finally:
            await provider.aclose()

    error = raised.value
    assert error.phase == "first_byte" and error.seconds == FAST.first_byte
    assert "no response" in str(error) and "0.6 seconds" in str(error)
    # Retrying a silent provider would wait the whole limit again, so it is not retried.
    assert server.connections == 1
    assert time.monotonic() - started < FAST.first_byte + 1.5


@pytest.mark.parametrize("name", ["chat", "responses", "anthropic", "ollama", "google"])
async def test_a_response_that_goes_quiet_after_starting_fails_at_the_idle_limit(name):
    make, first, rest, kind = PROTOCOLS[name]
    seen = []
    async with Server("stall", head=sse_head(kind), first=first, events=rest) as server:
        provider = make(server.url, retry=RETRIES, timeout=FAST)
        try:
            with pytest.raises(ProviderTimeoutError) as raised:
                async for event in provider.stream(REQUEST):
                    seen.append(event)
        finally:
            await provider.aclose()

    assert raised.value.phase == "idle" and raised.value.seconds == FAST.idle
    assert "stopped sending data" in str(raised.value)
    # What arrived before it went quiet was delivered, so a UI has shown it.
    assert any(getattr(event, "delta", "") == "hi" for event in seen)
    assert server.connections == 1  # output had already started, so it is never restarted


# ----------------------------------------------------- slow is not the same as stuck


@pytest.mark.parametrize("name", ["chat", "responses", "anthropic", "ollama", "google"])
async def test_a_slow_provider_that_does_answer_is_not_cut_off(name):
    make, first, rest, kind = PROTOCOLS[name]
    async with Server("slow", head=sse_head(kind), first=first, events=[first, *rest], delay=0.3) as server:
        provider = make(server.url, retry=RETRIES, timeout=TimeoutOptions(first_byte=3.0, idle=3.0))
        try:
            events = await collect(provider)
        finally:
            await provider.aclose()

    assert events[-1].response.message.content == "hi"


async def test_a_long_response_is_not_limited_by_its_total_time():
    """Only silence is limited: a stream that keeps arriving can run as long as it likes."""
    pieces = [b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'] * 8 + CHAT_REST
    async with Server("trickle", head=sse_head(), first=CHAT_FIRST, events=pieces, delay=0.25) as server:
        provider = OpenAIProvider(
            "m", "k", base_url=server.url + "/v1", timeout=TimeoutOptions(first_byte=1.0, idle=0.6)
        )
        started = time.monotonic()
        try:
            events = await collect(provider)
        finally:
            await provider.aclose()

    assert events[-1].response.message.content == "x" * 8
    assert time.monotonic() - started > 1.5  # longer than either limit


async def test_a_limit_set_to_none_waits_as_long_as_it_takes():
    async with Server("slow", head=sse_head(), first=CHAT_FIRST, events=[CHAT_FIRST, *CHAT_REST], delay=0.8) as server:
        provider = OpenAIProvider(
            "m", "k", base_url=server.url + "/v1", timeout=TimeoutOptions(connect=2.0, first_byte=None, idle=None)
        )
        try:
            events = await collect(provider)
        finally:
            await provider.aclose()

    assert events[-1].response.message.content == "hi"


async def test_the_headers_limit_is_independent_of_the_idle_limit():
    """A short idle limit must not shorten the wait for the headers, which httpx would do with one read limit."""
    async with Server("slow", head=sse_head(), first=CHAT_FIRST, events=[CHAT_FIRST, *CHAT_REST], delay=0.7) as server:
        provider = OpenAIProvider(
            "m", "k", base_url=server.url + "/v1", timeout=TimeoutOptions(first_byte=3.0, idle=0.3)
        )
        try:
            events = await collect(provider)
        finally:
            await provider.aclose()

    assert events[-1].response.message.content == "hi"


# --------------------------------------------------------------------- what retries


async def test_a_failed_connection_is_retried_because_it_costs_only_the_connect_limit():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectTimeout("no route")

    provider = OpenAIProvider("m", "k", transport=httpx.MockTransport(handler), retry=RETRIES, timeout=FAST)
    try:
        with pytest.raises(ProviderTimeoutError) as raised:
            await collect(provider)
    finally:
        await provider.aclose()

    assert raised.value.phase == "connect" and "connect" in str(raised.value)
    assert len(attempts) == 1 + RETRIES.max_retries


async def test_a_read_timeout_from_the_transport_is_a_typed_error_and_is_not_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ReadTimeout("silent")

    provider = OpenAIProvider("m", "k", transport=httpx.MockTransport(handler), retry=RETRIES, timeout=FAST)
    try:
        with pytest.raises(ProviderTimeoutError) as raised:
            await collect(provider)
    finally:
        await provider.aclose()

    assert raised.value.phase == "idle" and len(attempts) == 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderTimeoutError("x", phase="connect"), True),
        (ProviderTimeoutError("x", phase="first_byte"), False),
        (ProviderTimeoutError("x", phase="idle"), False),
        (httpx.ConnectTimeout("x"), True),
        (httpx.ReadTimeout("x"), False),
        (httpx.WriteTimeout("x"), False),
        (httpx.ConnectError("x"), True),
    ],
)
def test_only_a_connection_that_failed_is_worth_retrying(error, expected):
    assert RetryingProvider._retryable(error) is expected


def test_an_sdk_timeout_is_recognized_from_its_cause_whatever_the_sdk_calls_it():
    provider = OpenAIProvider("m", "k", timeout=TimeoutOptions(connect=2.0, first_byte=3.0, idle=4.0))

    class SdkTimeout(Exception):
        pass

    def wrapped(cause: BaseException) -> BaseException:
        error = SdkTimeout("Request timed out.")
        error.__cause__ = cause
        return error

    read = provider._normalized(wrapped(httpx.ReadTimeout("x")))
    connect = provider._normalized(wrapped(httpx.ConnectTimeout("x")))
    pool = provider._normalized(wrapped(httpx.PoolTimeout("x")))
    other = ValueError("unrelated")

    assert isinstance(read, ProviderTimeoutError) and (read.phase, read.seconds) == ("idle", 4.0)
    assert isinstance(connect, ProviderTimeoutError) and (connect.phase, connect.seconds) == ("connect", 2.0)
    assert pool.phase == "connect"
    assert provider._normalized(other) is other
    already = ProviderTimeoutError("x", phase="first_byte", seconds=1.0)
    assert provider._normalized(wrapped(already)) is already


def test_the_messages_say_which_wait_ran_out():
    assert "connect" in str(timeout_error("connect", 5)) and "5 seconds" in str(timeout_error("connect", 5))
    assert "no response" in str(timeout_error("first_byte", 120)) and "gateway" in str(timeout_error("first_byte", 120))
    assert "stopped sending data" in str(timeout_error("idle", 600)) and "incomplete" in str(timeout_error("idle", 600))
    assert "0.5 seconds" in str(timeout_error("idle", 0.5))
    assert timeout_error("idle", None).seconds is None


# --------------------------------------------------------------------- the guard


async def numbers(*delays: float) -> AsyncIterator[int]:
    for index, delay in enumerate(delays):
        await asyncio.sleep(delay)
        yield index


async def test_the_guard_passes_items_through_and_ends_with_the_stream():
    items = [item async for item in guard_idle(numbers(0, 0, 0), TimeoutOptions(idle=1.0))]

    assert items == [0, 1, 2]


async def test_the_guard_limits_every_wait_separately():
    options = TimeoutOptions(idle=0.5)

    ok = [item async for item in guard_idle(numbers(0.3, 0.3, 0.3), options)]  # 0.9 s in all, never 0.5 apart
    assert ok == [0, 1, 2]
    with pytest.raises(ProviderTimeoutError) as raised:
        [item async for item in guard_idle(numbers(0.1, 0.9), options)]
    assert raised.value.phase == "idle"


async def test_the_first_wait_can_have_its_own_limit():
    options = TimeoutOptions(first_byte=2.0, idle=0.3)

    items = [item async for item in guard_idle(numbers(0.8, 0.1), options, first=options.first_byte)]
    assert items == [0, 1]  # the first item took longer than idle, which only applies afterwards
    with pytest.raises(ProviderTimeoutError) as raised:
        [item async for item in guard_idle(numbers(0.9), TimeoutOptions(idle=5), first=0.2)]
    assert raised.value.phase == "first_byte" and raised.value.seconds == 0.2


async def test_with_no_limits_the_guard_just_waits():
    items = [item async for item in guard_idle(numbers(0.2), TimeoutOptions(idle=None))]

    assert items == [0]


# ------------------------------------------------------------- reaching the host


async def test_the_agent_run_fails_with_the_typed_error_so_the_host_can_tell_the_user():
    async with Server("silent", head=sse_head(), first=CHAT_FIRST, events=CHAT_REST) as server:
        provider = OpenAIProvider("m", "k", base_url=server.url + "/v1", retry=RETRIES, timeout=FAST)
        agent = await Agent.create(provider, config=AgentRunConfig("silent"))
        try:
            with pytest.raises(ProviderTimeoutError, match="no response"):
                await agent.run("hello")
            # The failed request was cleaned up, so the same agent accepts another one. If it had
            # been left active this would raise a protocol error about a request in progress.
            with pytest.raises(ProviderTimeoutError, match="no response"):
                await agent.run("hello again")
        finally:
            await provider.aclose()
