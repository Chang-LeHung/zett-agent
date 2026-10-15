"""Each provider's stop values, read through the real SDK and an HTTP transport.

Nothing here talks to a live service. The bodies are built from what each provider's
documentation says it sends, so a value the adapter fails to read shows up as a failing
case rather than as a refusal the user never sees.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from fake_gemini import API_KEY, FakeGemini
from zett_agent.messages import UserMessage
from zett_agent.model import ModelRequest
from zett_agent.providers.anthropic import AnthropicProvider
from zett_agent.providers.base import ProviderAuthError, ProviderResponseError, RetryOptions
from zett_agent.providers.deepseek import DeepSeekProvider
from zett_agent.providers.google import GoogleProvider
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.providers.stop_reasons import refusal_from_response_error, refusal_from_status_error
from zett_agent.stop import StopReason

REQUEST = ModelRequest(messages=(UserMessage(content="hello"),))
NO_RETRY = RetryOptions(max_retries=0)


async def collect(provider, request=REQUEST):
    events = [event async for event in provider.stream(request)]
    return events[-1].response


def chat_body(*chunks: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"


def sse_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def chat_provider(body: bytes, cls=OpenAIProvider, **options):
    transport = httpx.MockTransport(lambda request: sse_response(body))
    return cls(model="m", api_key="key", transport=transport, retry=NO_RETRY, **options)


def error_provider(status: int, error: dict[str, Any], *, response=False):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": error})

    return OpenAIProvider(
        model="m", api_key="key", transport=httpx.MockTransport(handler), retry=NO_RETRY, response=response
    )


def text(content: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": content}}]}


def finish(reason: str | None, **extra: Any) -> dict[str, Any]:
    return {"choices": [{"delta": {}, "finish_reason": reason, **extra}]}


# ------------------------------------------------- Chat Completions and its copies

#: What each provider's documentation lists as the values of ``finish_reason``.
CHAT_FAMILY = {
    "openai": {
        "stop": StopReason.END_TURN,
        "length": StopReason.MAX_OUTPUT,
        "tool_calls": StopReason.TOOL_USE,
        "content_filter": StopReason.REFUSAL,
        "function_call": StopReason.TOOL_USE,
    },
    "deepseek": {
        "stop": StopReason.END_TURN,
        "length": StopReason.MAX_OUTPUT,
        "content_filter": StopReason.REFUSAL,
        "tool_calls": StopReason.TOOL_USE,
        "insufficient_system_resource": StopReason.INTERRUPTED,
        "aborted": StopReason.INTERRUPTED,
    },
    "glm": {
        "stop": StopReason.END_TURN,
        "tool_calls": StopReason.TOOL_USE,
        "length": StopReason.MAX_OUTPUT,
        "sensitive": StopReason.REFUSAL,
        "model_context_window_exceeded": StopReason.CONTEXT_FULL,
        "network_error": StopReason.INTERRUPTED,
    },
    "kimi": {"stop": StopReason.END_TURN, "length": StopReason.MAX_OUTPUT, "tool_calls": StopReason.TOOL_USE},
    "openrouter": {
        "tool_calls": StopReason.TOOL_USE,
        "stop": StopReason.END_TURN,
        "length": StopReason.MAX_OUTPUT,
        "content_filter": StopReason.REFUSAL,
        "error": StopReason.INTERRUPTED,
    },
    "mistral": {
        "stop": StopReason.END_TURN,
        "length": StopReason.MAX_OUTPUT,
        "model_length": StopReason.CONTEXT_FULL,
        "error": StopReason.INTERRUPTED,
        "tool_calls": StopReason.TOOL_USE,
    },
}
CASES = [(vendor, raw, reason) for vendor, values in CHAT_FAMILY.items() for raw, reason in values.items()]


@pytest.mark.parametrize(("vendor", "raw", "expected"), CASES, ids=[f"{v}-{r}" for v, r, _ in CASES])
async def test_every_documented_chat_finish_reason_is_read(vendor, raw, expected):
    provider = chat_provider(chat_body(text("part"), finish(raw)))
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.finish_reason == raw and response.stop.reason is expected and response.stop.raw == raw
    assert response.message.content == "part"


async def test_deepseek_reads_the_same_values():
    provider = chat_provider(chat_body(text("part"), finish("insufficient_system_resource")), DeepSeekProvider)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.INTERRUPTED
    assert response.stop.category == "insufficient_system_resource"


async def test_a_refusal_field_is_kept_shown_and_makes_the_stop_a_refusal():
    """The model finished with ``stop``, and the only sign it declined is the refusal text."""
    body = chat_body(
        {"choices": [{"delta": {"refusal": "I can't "}}]},
        {"choices": [{"delta": {"refusal": "help with that."}}]},
        finish("stop"),
    )
    provider = chat_provider(body)
    try:
        events = [event async for event in provider.stream(REQUEST)]
    finally:
        await provider.aclose()

    response = events[-1].response
    assert response.message.content == "I can't help with that."
    assert [event.delta for event in events if event.type.value == "text_delta"] == ["I can't ", "help with that."]
    assert response.stop.reason is StopReason.REFUSAL
    assert response.stop.message == "I can't help with that." and response.stop.category == "refusal"
    assert response.stop.raw == "stop"


async def test_a_normal_answer_has_no_refusal_and_keeps_its_text():
    provider = chat_provider(chat_body(text("fine"), finish("stop")))
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.ok and response.stop.message is None and response.message.content == "fine"


async def test_a_content_filter_with_a_refusal_keeps_the_text_as_the_message():
    body = chat_body({"choices": [{"delta": {"refusal": "Not allowed."}}]}, finish("content_filter"))
    provider = chat_provider(body)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and response.stop.message == "Not allowed."
    assert response.stop.category == "content_filter"


async def test_a_gateway_keeps_the_upstream_value_in_the_detail():
    """OpenRouter normalizes ``finish_reason`` and reports the provider's own as ``native_finish_reason``."""
    body = chat_body(text("par"), finish("content_filter", native_finish_reason="SAFETY"))
    provider = chat_provider(body)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and response.stop.raw == "content_filter"
    assert response.stop.detail == {"native_finish_reason": "SAFETY"}


def mid_stream_error(code: Any = "server_error", message: str = "Provider disconnected unexpectedly") -> dict[str, Any]:
    """The chunk OpenRouter sends when a provider fails after the 200 was committed."""
    return {
        "id": "gen-1",
        "error": {"code": code, "message": message},
        "choices": [{"delta": {"content": ""}, "finish_reason": "error"}],
    }


async def test_an_error_after_some_output_ends_the_response_as_an_interruption_and_keeps_the_text():
    """The SDK raises on a chunk with a top-level ``error``; the adapter turns that into a stop."""
    provider = chat_provider(chat_body(text("par"), mid_stream_error()))
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    stop = response.stop
    assert response.message.content == "par"
    assert stop.reason is StopReason.INTERRUPTED and stop.raw == "error" and stop.category == "error"
    assert stop.message == "Provider disconnected unexpectedly"
    assert stop.detail["error"] == {"code": "server_error", "message": "Provider disconnected unexpectedly"}


async def test_an_error_after_a_tool_call_began_is_an_interruption_too():
    call = {"index": 0, "id": "c1", "function": {"name": "bash", "arguments": '{"comm'}}
    body = chat_body({"choices": [{"delta": {"tool_calls": [call]}}]}, mid_stream_error())
    provider = chat_provider(body)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.INTERRUPTED


async def test_an_error_as_the_first_event_is_a_typed_error_so_it_can_be_retried():
    """OpenRouter: the error can be the only event. Nothing was produced, so it is raised and retried."""
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return sse_response(chat_body(mid_stream_error(code=502, message="upstream down")))

    provider = OpenAIProvider(
        model="m",
        api_key="key",
        transport=httpx.MockTransport(handler),
        retry=RetryOptions(max_retries=2, base_delay=0.001, max_delay=0.01),
    )
    try:
        with pytest.raises(ProviderResponseError, match="upstream down"):
            await collect(provider)
    finally:
        await provider.aclose()

    assert len(attempts) == 3  # a 5xx code is retried, then the typed error reaches the caller


async def test_a_failure_that_is_not_transient_is_not_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return sse_response(chat_body(mid_stream_error(code="invalid_prompt", message="rejected")))

    provider = OpenAIProvider(
        model="m",
        api_key="key",
        transport=httpx.MockTransport(handler),
        retry=RetryOptions(max_retries=2, base_delay=0.001, max_delay=0.01),
    )
    try:
        with pytest.raises(ProviderResponseError, match="rejected"):
            await collect(provider)
    finally:
        await provider.aclose()

    assert len(attempts) == 1


async def test_a_native_value_equal_to_the_normalized_one_is_not_repeated():
    provider = chat_provider(chat_body(text("x"), finish("stop", native_finish_reason="stop")))
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.detail == {}


async def test_the_adapter_keeps_the_tool_call_and_the_loop_is_what_withholds_it():
    call = {"index": 0, "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    body = chat_body({"choices": [{"delta": {"tool_calls": [call]}}]}, finish("content_filter"))
    provider = chat_provider(body)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and len(response.message.tool_calls) == 1


# ----------------------------------------------- refusals that arrive as HTTP errors

KIMI = {"type": "content_filter", "message": "The request was rejected because it was considered high risk"}


async def test_kimis_content_filter_error_is_answered_as_a_refusal():
    provider = error_provider(400, KIMI)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.message.content == "" and response.message.tool_calls == ()
    assert response.stop.reason is StopReason.REFUSAL and response.stop.category == "content_filter"
    assert "considered high risk" in response.stop.message and response.stop.detail["status"] == 400


MODERATION = {
    "code": 403,
    "message": "Your input was flagged",
    "metadata": {
        "reasons": ["violence"],
        "flagged_input": "do the bad thing",
        "provider_name": "OpenAI",
        "model_slug": "openai/gpt-4o",
    },
}
GUARDRAIL = {
    "code": 403,
    "message": "Request blocked: prompt injection patterns detected",
    "metadata": {"patterns": ["ignore all previous instructions"]},
}


async def test_an_openrouter_moderation_block_is_a_refusal_with_its_reasons():
    provider = error_provider(403, MODERATION)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    stop = response.stop
    assert (
        stop.reason is StopReason.REFUSAL and stop.category == "moderation" and stop.message == "Your input was flagged"
    )
    assert stop.detail["reasons"] == ["violence"] and stop.detail["provider_name"] == "OpenAI"
    assert stop.detail["flagged_input"] == "do the bad thing" and stop.detail["status"] == 403


async def test_an_openrouter_guardrail_block_is_a_refusal_with_its_patterns():
    provider = error_provider(403, GUARDRAIL)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.category == "guardrail" and response.stop.detail["patterns"] == [
        "ignore all previous instructions"
    ]


async def test_the_same_refusal_is_recognized_on_the_responses_endpoint():
    provider = error_provider(400, KIMI, response=True)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, {"type": "invalid_request_error", "message": "bad parameter"}),
        (403, {"code": 403, "message": "budget used", "metadata": {"limit_source": "key"}}),
        (403, {"code": 403, "message": "forbidden"}),
        (422, {"message": "unprocessable"}),
        (500, {"message": "server broke"}),
        (429, {"type": "engine_overloaded_error", "message": "busy"}),
    ],
)
async def test_other_http_errors_stay_errors(status, error):
    provider = error_provider(status, error)
    try:
        with pytest.raises(ProviderResponseError) as raised:
            await collect(provider)
    finally:
        await provider.aclose()

    assert not hasattr(raised.value, "stop")


async def test_a_rejected_credential_is_still_an_auth_error():
    provider = error_provider(401, {"message": "bad key"})
    try:
        with pytest.raises(ProviderAuthError):
            await collect(provider)
    finally:
        await provider.aclose()


def test_the_status_error_reader_is_strict_about_what_it_accepts():
    assert refusal_from_status_error(400, None) is None
    assert refusal_from_status_error(400, "text") is None
    assert refusal_from_status_error(400, {"code": "content_filter"}).category == "content_filter"
    assert refusal_from_status_error(200, KIMI) is None
    assert refusal_from_status_error(403, {"metadata": "x"}) is None
    assert refusal_from_status_error(403, {"metadata": {"reasons": ["a"]}}).message is None


def test_the_responses_error_reader_only_takes_policy_codes():
    assert refusal_from_response_error({"code": "bio_policy", "message": "no"}).category == "bio_policy"
    assert refusal_from_response_error({"code": "image_content_policy_violation"}).message is None
    for code in ("server_error", "rate_limit_exceeded", "invalid_image", None):
        assert refusal_from_response_error({"code": code, "message": "x"}) is None


# ------------------------------------------------------------- OpenAI Responses


def sse_events(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def response_object(output: list[dict[str, Any]], *, status="completed", incomplete=None) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "incomplete_details": incomplete,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def responses_provider(events: list[dict[str, Any]]) -> OpenAIProvider:
    transport = httpx.MockTransport(lambda request: sse_response(sse_events(events)))
    return OpenAIProvider("m", "key", transport=transport, retry=NO_RETRY, response=True)


def message_item(*parts: dict[str, Any]) -> dict[str, Any]:
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": list(parts)}


def terminal(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": kind, "sequence_number": 9, "response": payload}


FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "bash",
    "arguments": '{"command": "ls"}',
    "status": "completed",
}


async def test_an_incomplete_response_reports_the_reason_it_stopped():
    cases = {"max_output_tokens": StopReason.MAX_OUTPUT, "content_filter": StopReason.REFUSAL}
    for reason, expected in cases.items():
        payload = response_object([message_item({"type": "output_text", "text": "par", "annotations": []})])
        payload.update(status="incomplete", incomplete_details={"reason": reason})
        provider = responses_provider([terminal("response.incomplete", payload)])
        try:
            response = await collect(provider)
        finally:
            await provider.aclose()

        assert response.stop.reason is expected and response.stop.raw == reason


async def test_a_completed_response_is_a_normal_stop():
    payload = response_object([message_item({"type": "output_text", "text": "fine", "annotations": []})])
    provider = responses_provider([terminal("response.completed", payload)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.ok and response.stop.raw == "completed" and response.message.content == "fine"


async def test_a_refusal_content_block_makes_a_completed_response_a_refusal():
    payload = response_object([message_item({"type": "refusal", "refusal": "I can't help with that."})])
    provider = responses_provider([terminal("response.completed", payload)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.message.content == "I can't help with that."
    assert response.stop.reason is StopReason.REFUSAL and response.stop.message == "I can't help with that."
    assert response.stop.raw == "completed" and response.stop.category == "refusal"


async def test_streamed_refusal_text_is_shown_and_explains_the_stop():
    payload = response_object([message_item({"type": "refusal", "refusal": "No can do."})])
    events = [
        {"type": "response.refusal.delta", "sequence_number": 1, "item_id": "m", "output_index": 0,
         "content_index": 0, "delta": "No can "},
        {"type": "response.refusal.delta", "sequence_number": 2, "item_id": "m", "output_index": 0,
         "content_index": 0, "delta": "do."},
        terminal("response.completed", payload),
    ]  # fmt: skip
    provider = responses_provider(events)
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and response.stop.message == "No can do."


async def test_a_refusal_removes_the_function_call_from_the_provider_record_too():
    """A call replayed without its output would be rejected on the next request."""
    payload = response_object(
        [message_item({"type": "refusal", "refusal": "No."}), FUNCTION_CALL],
        status="incomplete",
        incomplete={"reason": "content_filter"},
    )
    provider = responses_provider([terminal("response.incomplete", payload)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and len(response.message.tool_calls) == 1
    assert [block["type"] for block in response.message.replay_blocks] == ["message"]


async def test_a_function_call_is_kept_in_the_record_when_the_step_was_normal():
    payload = response_object([FUNCTION_CALL])
    provider = responses_provider([terminal("response.completed", payload)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.TOOL_USE
    assert [block["type"] for block in response.message.replay_blocks] == ["function_call"]


@pytest.mark.parametrize("code", ["bio_policy", "image_content_policy_violation"])
async def test_a_policy_failure_is_answered_as_a_refusal(code):
    failed = response_object([], status="failed")
    failed["error"] = {"code": code, "message": "The request was blocked by policy."}
    provider = responses_provider([terminal("response.failed", failed)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and response.stop.category == code
    assert response.stop.message == "The request was blocked by policy."


async def test_any_other_failure_still_raises():
    failed = response_object([], status="failed")
    failed["error"] = {"code": "server_error", "message": "boom"}
    provider = responses_provider([terminal("response.failed", failed)])
    try:
        with pytest.raises(ProviderResponseError, match="boom"):
            await collect(provider)
    finally:
        await provider.aclose()


# --------------------------------------------------------------------- Anthropic


def anthropic_sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def anthropic_provider(events: list[dict[str, Any]]) -> AnthropicProvider:
    transport = httpx.MockTransport(lambda request: sse_response(anthropic_sse(events)))
    return AnthropicProvider("claude", "key", transport=transport, retry=NO_RETRY)


def start() -> dict[str, Any]:
    return {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}}


def text_block(content: str) -> list[dict[str, Any]]:
    return [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": content}},
        {"type": "content_block_stop", "index": 0},
    ]


def tool_block(index: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
        },
        {"type": "content_block_stop", "index": index},
    ]


def end(reason: str, details: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    delta: dict[str, Any] = {"stop_reason": reason}
    if details is not None:
        delta["stop_details"] = details
    return [{"type": "message_delta", "delta": delta, "usage": {"output_tokens": 1}}, {"type": "message_stop"}]


ANTHROPIC = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_OUTPUT,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "pause_turn": StopReason.PAUSED,
    "refusal": StopReason.REFUSAL,
    "model_context_window_exceeded": StopReason.CONTEXT_FULL,
}


@pytest.mark.parametrize(("raw", "expected"), list(ANTHROPIC.items()))
async def test_every_documented_anthropic_stop_reason_is_read(raw, expected):
    provider = anthropic_provider([start(), *text_block("part"), *end(raw)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is expected and response.stop.raw == raw and response.message.content == "part"


async def test_a_refusal_carries_its_category_explanation_and_the_rest_of_stop_details():
    details = {
        "type": "refusal",
        "category": "cyber",
        "explanation": "This request was declined because it could enable cyber harm.",
        "recommended_model": "claude-opus-4-8",
        "fallback_credit_token": "tok-1",
        "fallback_has_prefill_claim": False,
    }
    provider = anthropic_provider([start(), *text_block("Here is the sta"), *end("refusal", details)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    stop = response.stop
    assert stop.reason is StopReason.REFUSAL and stop.category == "cyber"
    assert stop.message == "This request was declined because it could enable cyber harm."
    assert stop.recommended_model == "claude-opus-4-8"
    assert stop.detail == {"fallback_credit_token": "tok-1", "fallback_has_prefill_claim": False}
    assert response.message.content == "Here is the sta"  # the partial output is kept, not discarded


async def test_a_refusal_with_a_null_category_is_still_a_refusal_with_a_label():
    details = {"type": "refusal", "category": None, "explanation": None}
    provider = anthropic_provider([start(), *end("refusal", details)])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and response.stop.category == "refusal"
    assert response.stop.message is None and response.stop.detail == {}


async def test_a_refusal_after_a_complete_tool_use_removes_the_block_from_the_replay():
    """A ``tool_use`` replayed without a ``tool_result`` is rejected by the API."""
    provider = anthropic_provider([start(), *text_block("About to run it. "), *tool_block(1), *end("refusal")])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.REFUSAL and len(response.message.tool_calls) == 1
    assert [block["type"] for block in response.message.replay_blocks] == ["text"]


async def test_a_normal_tool_use_keeps_its_block():
    provider = anthropic_provider([start(), *tool_block(0), *end("tool_use")])
    try:
        response = await collect(provider)
    finally:
        await provider.aclose()

    assert response.stop.reason is StopReason.TOOL_USE
    assert [block["type"] for block in response.message.replay_blocks] == ["tool_use"]


# ------------------------------------------------------------------------ Google


@pytest.fixture
def gemini():
    with FakeGemini() as server:
        yield server


async def google_response(server, prompt: str):
    provider = GoogleProvider("gemini-2.5-flash", api_key=API_KEY, base_url=server.url, retry=NO_RETRY)
    try:
        return await collect(provider, ModelRequest(messages=(UserMessage(content=prompt),)))
    finally:
        await provider.aclose()


async def test_a_google_safety_refusal_keeps_the_partial_text_and_names_the_category(gemini):
    response = await google_response(gemini, "refused halfway")

    assert response.message.content == "Here is the start of"
    assert response.stop.reason is StopReason.REFUSAL and response.stop.raw == "SAFETY"
    assert response.stop.category == "SAFETY"
    # The Gemini API path of google-genai drops ``finishMessage`` (only Vertex keeps it), so
    # there is no sentence to show here; the category is the whole explanation.
    assert response.stop.message is None


async def test_a_blocked_prompt_is_a_refusal_not_a_protocol_error(gemini):
    """A blocked prompt has no candidate at all, so it used to surface as 'no finish reason'."""
    response = await google_response(gemini, "blocked prompt")

    assert response.message.content == "" and response.message.tool_calls == ()
    assert response.stop.reason is StopReason.REFUSAL and response.stop.category == "PROHIBITED_CONTENT"
    assert response.stop.message == "Prompt blocked" and response.stop.detail == {"blocked": "prompt"}


async def test_a_google_refusal_after_a_function_call_removes_the_call_from_the_replay(gemini):
    response = await google_response(gemini, "refused after a call")

    assert response.stop.reason is StopReason.REFUSAL
    assert len(response.message.tool_calls) == 1
    assert not any("function_call" in part for part in response.message.replay_blocks)


async def test_google_reports_stop_as_a_tool_use_when_there_is_a_function_call(gemini):
    provider = GoogleProvider("gemini-2.5-flash", api_key=API_KEY, base_url=server_url(gemini), retry=NO_RETRY)
    try:
        response = await collect(
            provider,
            ModelRequest(
                messages=(UserMessage(content="weather in Paris and Rome"),),
                tools=(_weather(),),
            ),
        )
    finally:
        await provider.aclose()

    assert response.finish_reason == "STOP" and response.stop.reason is StopReason.TOOL_USE
    assert any("function_call" in part for part in response.message.replay_blocks)


def server_url(server) -> str:
    return server.url


def _weather():
    from zett_agent.model import ToolDefinition

    return ToolDefinition("get_weather", "Get weather", {"type": "object", "properties": {"city": {"type": "string"}}})


async def test_google_truncation_is_max_output(gemini):
    response = await google_response(gemini, "too long")

    assert response.stop.reason is StopReason.MAX_OUTPUT and response.stop.truncated


async def test_a_normal_google_answer_is_a_normal_stop(gemini):
    response = await google_response(gemini, "hello there")

    assert response.stop.reason is StopReason.END_TURN and response.stop.ok


async def test_a_dropped_connection_after_some_output_is_not_mistaken_for_a_reported_error():
    """The SDK's connection and timeout errors are APIErrors too, but the provider said nothing."""
    import openai

    class Dropping:
        response = None

        def __aiter__(self):
            return self._chunks()

        async def _chunks(self):
            yield type(
                "Chunk",
                (),
                {
                    "usage": None,
                    "choices": [
                        type(
                            "C",
                            (),
                            {
                                "delta": type(
                                    "D",
                                    (),
                                    {"content": "par", "refusal": None, "reasoning_content": None, "tool_calls": None},
                                )(),
                                "finish_reason": None,
                            },
                        )()
                    ],
                },
            )()
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

        async def close(self):
            pass

    provider = OpenAIProvider(
        model="m", api_key="key", transport=httpx.MockTransport(lambda r: sse_response(b"")), retry=NO_RETRY
    )

    async def request(_):
        return Dropping()

    provider._request = request  # type: ignore[method-assign]
    try:
        with pytest.raises(openai.APIConnectionError):
            await collect(provider)
    finally:
        await provider.aclose()
