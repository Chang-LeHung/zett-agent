"""Stop reasons: the vocabulary, how a response and a message carry it, and how it is stored."""

from __future__ import annotations

import json

import pytest

from zett_agent.exceptions import AgentError, AgentStopError
from zett_agent.messages import AssistantMessage, ToolCall
from zett_agent.model import ModelResponse
from zett_agent.stop import StopInfo, StopReason, classify_stop
from zett_agent.storage import decode_messages, encode_messages

# ------------------------------------------------------------------ vocabulary

#: Every raw value the supported providers document, and what each one means.
DOCUMENTED = [
    # Anthropic
    ("end_turn", StopReason.END_TURN),
    ("max_tokens", StopReason.MAX_OUTPUT),
    ("stop_sequence", StopReason.END_TURN),
    ("tool_use", StopReason.TOOL_USE),
    ("pause_turn", StopReason.PAUSED),
    ("refusal", StopReason.REFUSAL),
    ("model_context_window_exceeded", StopReason.CONTEXT_FULL),
    # OpenAI Chat, and the providers that copy it
    ("stop", StopReason.END_TURN),
    ("length", StopReason.MAX_OUTPUT),
    ("tool_calls", StopReason.TOOL_USE),
    ("content_filter", StopReason.REFUSAL),
    ("function_call", StopReason.TOOL_USE),
    # OpenAI Responses
    ("completed", StopReason.END_TURN),
    ("max_output_tokens", StopReason.MAX_OUTPUT),
    # DeepSeek
    ("insufficient_system_resource", StopReason.INTERRUPTED),
    ("aborted", StopReason.INTERRUPTED),
    # GLM
    ("sensitive", StopReason.REFUSAL),
    ("network_error", StopReason.INTERRUPTED),
    # OpenRouter, Mistral
    ("error", StopReason.INTERRUPTED),
    ("model_length", StopReason.CONTEXT_FULL),
    # Google
    ("STOP", StopReason.END_TURN),
    ("MAX_TOKENS", StopReason.MAX_OUTPUT),
    ("SAFETY", StopReason.REFUSAL),
    ("RECITATION", StopReason.REFUSAL),
    ("BLOCKLIST", StopReason.REFUSAL),
    ("PROHIBITED_CONTENT", StopReason.REFUSAL),
    ("SPII", StopReason.REFUSAL),
    ("IMAGE_SAFETY", StopReason.REFUSAL),
    ("IMAGE_PROHIBITED_CONTENT", StopReason.REFUSAL),
    ("IMAGE_RECITATION", StopReason.REFUSAL),
    ("MODEL_ARMOR", StopReason.REFUSAL),
    ("JAILBREAK", StopReason.REFUSAL),
]


@pytest.mark.parametrize(("raw", "expected"), DOCUMENTED)
def test_every_documented_value_maps_to_its_meaning(raw, expected):
    stop = classify_stop(raw)

    assert stop.reason is expected
    assert stop.raw == raw


@pytest.mark.parametrize(
    "raw",
    [
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "OTHER",
        "LANGUAGE",
        "NO_IMAGE",
        "FINISH_REASON_UNSPECIFIED",
        "incomplete",
        "load",
        "a_value_from_next_year",
        "",
    ],
)
def test_a_value_the_library_does_not_know_is_other_and_kept(raw):
    stop = classify_stop(raw)

    assert stop.reason is StopReason.OTHER and stop.raw == raw and not stop.ok


def test_only_finishing_and_asking_for_tools_are_normal():
    normal = {StopReason.END_TURN, StopReason.TOOL_USE}

    assert {reason for reason in StopReason if classify_stop(reason.value).ok} <= normal
    assert classify_stop("stop").ok and classify_stop("tool_calls").ok
    assert not any(classify_stop(raw).ok for raw in ("length", "content_filter", "pause_turn", "error", "nope"))


def test_lookup_ignores_case_and_surrounding_space():
    assert classify_stop("  Content_Filter ").reason is StopReason.REFUSAL


def test_a_step_that_finished_but_has_tool_calls_asked_for_tools():
    """Gemini reports STOP whether or not it called a function."""
    assert classify_stop("STOP", has_tool_calls=True).reason is StopReason.TOOL_USE
    assert classify_stop("stop", has_tool_calls=False).reason is StopReason.END_TURN
    assert classify_stop("length", has_tool_calls=True).reason is StopReason.MAX_OUTPUT


def test_a_step_with_no_reason_is_not_reported_as_a_failure():
    assert classify_stop(None).reason is StopReason.END_TURN
    assert classify_stop(None, has_tool_calls=True).reason is StopReason.TOOL_USE


def test_a_refusal_or_interruption_without_a_category_takes_the_raw_value():
    assert classify_stop("content_filter").category == "content_filter"
    assert classify_stop("network_error").category == "network_error"
    assert classify_stop("SAFETY").category == "SAFETY"
    assert classify_stop("length").category is None
    assert classify_stop("content_filter", category="cyber").category == "cyber"


def test_the_stop_carries_what_the_provider_said_in_plain_fields():
    stop = classify_stop(
        "refusal", category="cyber", message="Declined.", recommended_model="m-2", detail={"token": "t"}
    )

    assert (stop.category, stop.message, stop.recommended_model, dict(stop.detail)) == (
        "cyber",
        "Declined.",
        "m-2",
        {"token": "t"},
    )
    assert classify_stop("stop").detail == {}


def test_properties_answer_the_questions_a_host_asks():
    assert classify_stop("refusal").refused and not classify_stop("length").refused
    assert classify_stop("length").truncated and classify_stop("model_context_window_exceeded").truncated
    assert not classify_stop("refusal").truncated and not classify_stop("stop").truncated


def test_describe_is_one_line_a_person_can_read():
    assert classify_stop("stop").describe() == "end_turn (stop)"
    assert StopInfo(StopReason.END_TURN).describe() == "end_turn"
    assert classify_stop("content_filter").describe() == "refusal (content_filter)"
    assert classify_stop("refusal", category="cyber", message="Declined.").describe() == "refusal (cyber): Declined."
    assert StopInfo(StopReason.OTHER, raw="weird").describe() == "other (weird)"
    assert StopInfo(StopReason.REFUSAL, message="No.").describe() == "refusal: No."


def test_the_stop_is_immutable():
    with pytest.raises(AttributeError):
        classify_stop("stop").reason = StopReason.REFUSAL  # type: ignore[misc]


# -------------------------------------------------------------- model response


def test_a_response_derives_its_stop_when_an_adapter_gives_none():
    response = ModelResponse(AssistantMessage(content="x"), finish_reason="content_filter")

    assert response.stop is not None and response.stop.reason is StopReason.REFUSAL
    assert ModelResponse(AssistantMessage(content="x"), finish_reason="length").stop.truncated
    call = ToolCall("1", "t", {})
    assert ModelResponse(AssistantMessage(tool_calls=(call,)), finish_reason="STOP").stop.reason is StopReason.TOOL_USE


def test_a_stop_the_adapter_supplies_is_not_overwritten():
    given = StopInfo(StopReason.REFUSAL, raw="stop", category="cyber")

    response = ModelResponse(AssistantMessage(content="x"), finish_reason="stop", stop=given)

    assert response.stop is given


# ----------------------------------------------------------------- the message


def test_raise_for_stop_returns_a_normal_message_and_raises_for_the_rest():
    normal = AssistantMessage(content="ok")
    refused = AssistantMessage(content="", stop=classify_stop("refusal", category="cyber", message="Declined."))

    assert normal.raise_for_stop() is normal
    assert AssistantMessage(content="ok", stop=classify_stop("stop")).raise_for_stop().content == "ok"
    with pytest.raises(AgentStopError) as raised:
        refused.raise_for_stop()
    assert raised.value.stop is refused.stop and raised.value.message is refused
    assert "refusal (cyber): Declined." in str(raised.value) and isinstance(raised.value, AgentError)


def test_a_stop_survives_storage_and_old_messages_without_one_still_load():
    stop = classify_stop("refusal", category="cyber", message="Declined.", recommended_model="m-2", detail={"a": [1]})
    stored = encode_messages([AssistantMessage(content="part", stop=stop), AssistantMessage(content="plain")])

    restored = decode_messages(stored)

    assert restored[0].stop == stop and restored[0].stop.reason is StopReason.REFUSAL
    assert restored[1].stop is None
    legacy = json.loads(stored)
    del legacy[0]["data"]["stop"]
    del legacy[1]["data"]["stop"]
    assert [message.stop for message in decode_messages(json.dumps(legacy))] == [None, None]
