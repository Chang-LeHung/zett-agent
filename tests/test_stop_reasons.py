"""Stop reasons: the vocabulary, how the loop acts on each, and how the host is told."""

from __future__ import annotations

import json

import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.dispatcher import AgentEventDispatcher
from zett_agent.events import AgentEvent, AgentEventType
from zett_agent.exceptions import AgentError, AgentStopError
from zett_agent.extensions.async_subagent import AsyncSubAgentExtension, SubAgentTask, SubAgentTaskStatus
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.refusal_replay import RefusalReplay, RefusalReplayExtension
from zett_agent.extensions.subagent import SubAgentDefinition, SubAgentResult, subagent_report
from zett_agent.messages import AssistantMessage, ToolCall, UserMessage
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.stop import StopInfo, StopReason, classify_stop
from zett_agent.storage import decode_messages, encode_messages
from zett_agent.tools.base import tool

CONFIG = AgentRunConfig("stop-reasons")


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


# ------------------------------------------------------------------ the loop


@tool(guidelines="Use to record a call.")
async def record(label: str) -> str:
    """Record one call.

    Args:
        label: What to record.
    """
    CALLS.append(label)
    return "recorded"


CALLS: list[str] = []


class Scripted:
    """Answer each request with the next scripted response and keep every request."""

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if response.message.content:
            yield ModelEvent.text(response.message.content)
        yield ModelEvent.completed(response)


def step(content="", *, raw=None, calls=(), stop=None) -> ModelResponse:
    return ModelResponse(AssistantMessage(content=content, tool_calls=tuple(calls)), finish_reason=raw, stop=stop)


def call(label="x") -> ToolCall:
    return ToolCall("call-1", "record", {"label": label})


async def run(*responses, extensions=None, max_pause=None):
    CALLS.clear()
    model = Scripted(*responses)
    agent = await Agent.create(model, config=CONFIG, tools=[record], extensions=extensions)
    if max_pause is not None:
        agent.max_pause_continuations = max_pause
    events = [event async for event in agent.stream("Question")]
    return model, events


def stopped(events) -> list[AgentEvent]:
    return [event for event in events if event.type is AgentEventType.MODEL_STOPPED]


def final(events) -> AssistantMessage:
    return next(event.message for event in events if event.type is AgentEventType.RUN_COMPLETED)


async def test_a_normal_answer_changes_nothing():
    model, events = await run(step("Done.", raw="stop"))

    assert stopped(events) == [] and final(events).stop is None and final(events).content == "Done."
    assert len(model.requests) == 1


async def test_a_normal_tool_round_trip_reports_nothing():
    model, events = await run(step(raw="tool_calls", calls=[call("a")]), step("Done.", raw="stop"))

    assert CALLS == ["a"] and stopped(events) == [] and len(model.requests) == 2


async def test_a_refusal_ends_the_run_and_is_reported():
    refusal = classify_stop("refusal", category="cyber", message="Declined.")
    model, events = await run(step("I can't help", raw="refusal", stop=refusal))

    [event] = stopped(events)
    assert event.stop is refusal and event.stop.reason is StopReason.REFUSAL
    assert event.message.content == "I can't help" and event.response.stop is refusal
    assert final(events).stop == event.stop and final(events).content == "I can't help"
    assert len(model.requests) == 1


async def test_the_stopped_event_follows_the_completed_one():
    _, events = await run(step("No.", raw="content_filter"))

    kinds = [event.type for event in events]
    assert kinds.index(AgentEventType.MODEL_COMPLETED) + 1 == kinds.index(AgentEventType.MODEL_STOPPED)
    assert kinds.index(AgentEventType.MODEL_STOPPED) < kinds.index(AgentEventType.RUN_COMPLETED)


@pytest.mark.parametrize("raw", ["refusal", "content_filter", "sensitive", "SAFETY"])
async def test_a_tool_call_that_came_with_a_refusal_is_not_run(raw):
    model, events = await run(step(raw=raw, calls=[call("rm -rf")]))

    assert CALLS == []
    message = final(events)
    assert message.tool_calls == () and message.stop.reason is StopReason.REFUSAL
    assert message.stop.detail["withheld_tool_calls"] == ["record"]
    assert not any(event.type is AgentEventType.TOOL_STARTED for event in events)
    assert len(model.requests) == 1


@pytest.mark.parametrize("raw", ["error", "network_error", "aborted", "insufficient_system_resource"])
async def test_a_tool_call_from_an_interrupted_stream_is_not_run_either(raw):
    """The stream ended mid-way, so the call cannot be trusted to be whole."""
    _, events = await run(step("half", raw=raw, calls=[call()]))

    assert CALLS == []
    assert final(events).stop.reason is StopReason.INTERRUPTED and final(events).tool_calls == ()
    assert stopped(events)[0].stop.detail["withheld_tool_calls"] == ["record"]


async def test_the_withheld_calls_are_gone_from_the_history_the_next_request_sends():
    model = Scripted(step(raw="refusal", calls=[call()]), step("Fine.", raw="stop"))
    agent = await Agent.create(model, config=CONFIG, tools=[record])
    await agent.run("first")
    CALLS.clear()

    await agent.run("second")

    assistants = [m for m in model.requests[1].messages if isinstance(m, AssistantMessage)]
    assert assistants and all(m.tool_calls == () for m in assistants) and CALLS == []


async def test_a_truncated_answer_is_reported_and_the_run_still_finishes():
    model, events = await run(step("cut off mid", raw="length"))

    [event] = stopped(events)
    assert event.stop.reason is StopReason.MAX_OUTPUT and event.stop.truncated
    assert final(events).content == "cut off mid" and final(events).stop.reason is StopReason.MAX_OUTPUT


async def test_a_truncation_in_the_middle_of_tool_work_is_reported_while_the_work_goes_on():
    model, events = await run(step(raw="length", calls=[call("kept")]), step("Done.", raw="stop"))

    assert CALLS == ["kept"] and len(model.requests) == 2
    assert [event.stop.reason for event in stopped(events)] == [StopReason.MAX_OUTPUT]
    assert final(events).stop is None


async def test_context_full_and_unknown_values_are_reported():
    _, full = await run(step("x", raw="model_context_window_exceeded"))
    _, other = await run(step("x", raw="a_value_from_next_year"))

    assert stopped(full)[0].stop.reason is StopReason.CONTEXT_FULL
    assert stopped(other)[0].stop.reason is StopReason.OTHER and stopped(other)[0].stop.raw == "a_value_from_next_year"


async def test_a_paused_step_is_continued_in_the_same_turn():
    turns = []

    class Count(AgentExtension):
        async def before_turn(self, context):
            turns.append("start")

    model, events = await run(
        step("part one ", raw="pause_turn"),
        step("part two", raw="pause_turn"),
        step("done", raw="end_turn"),
        extensions=[Count()],
    )

    assert len(model.requests) == 3 and stopped(events) == []
    assert turns == ["start"]  # one turn, three model calls
    assert final(events).content == "done" and final(events).stop is None
    # The paused text stays in the history, so the continuation picks up where it stopped.
    assert [m.content for m in model.requests[2].messages if isinstance(m, AssistantMessage)] == [
        "part one ",
        "part two",
    ]


async def test_pauses_stop_being_continued_at_the_limit_and_are_then_reported():
    model, events = await run(
        step("a", raw="pause_turn"),
        step("b", raw="pause_turn"),
        step("c", raw="pause_turn"),
        max_pause=2,
    )

    assert len(model.requests) == 3
    [event] = stopped(events)
    assert event.stop.reason is StopReason.PAUSED and final(events).stop.reason is StopReason.PAUSED


async def test_a_pause_that_was_continued_is_not_recorded_but_one_that_ended_the_run_is():
    """A restored session reports what is on the stored messages, so a continued pause must not be."""
    model = Scripted(step("a", raw="pause_turn"), step("b", raw="pause_turn"), step("c", raw="pause_turn"))
    agent = await Agent.create(model, config=CONFIG, extensions=[InMemoryMessageAccumulator()])
    agent.max_pause_continuations = 2

    await agent.run("Question")

    assistants = [m for m in agent.get_state(CONFIG.session_id).messages if isinstance(m, AssistantMessage)]
    assert [m.content for m in assistants] == ["a", "b", "c"]
    assert [m.stop.reason if m.stop else None for m in assistants] == [None, None, StopReason.PAUSED]


async def test_a_pause_is_reported_at_once_when_continuing_is_switched_off():
    model, events = await run(step("a", raw="pause_turn"), max_pause=0)

    assert len(model.requests) == 1 and stopped(events)[0].stop.reason is StopReason.PAUSED


async def test_the_pause_count_starts_over_after_a_step_that_is_not_paused():
    model, events = await run(
        step("a", raw="pause_turn"),
        step(raw="tool_calls", calls=[call()]),
        step("b", raw="pause_turn"),
        step("done", raw="end_turn"),
        max_pause=1,
    )

    assert len(model.requests) == 4 and stopped(events) == [] and final(events).content == "done"


async def test_a_stop_an_adapter_supplies_wins_over_the_raw_value():
    stop = StopInfo(StopReason.REFUSAL, raw="stop", category="refusal", message="I can't do that.")
    _, events = await run(step("I can't do that.", raw="stop", stop=stop))

    assert stopped(events)[0].stop.message == "I can't do that." and final(events).stop.reason is StopReason.REFUSAL


async def test_the_dispatcher_routes_the_new_event():
    seen = []

    class Handler(AgentEventDispatcher):
        async def on_model_stopped_event(self, event):
            seen.append(event.stop.reason)

    _, events = await run(step("No.", raw="content_filter"))
    for event in events:
        await Handler().dispatch(event)

    assert seen == [StopReason.REFUSAL]


async def test_extensions_see_the_stop_in_after_model():
    seen = []

    class Watch(AgentExtension):
        async def after_model(self, context, response):
            seen.append((response.stop.reason, response.message.stop, response.message.tool_calls))

    await run(step(raw="refusal", calls=[call()]), extensions=[Watch()])

    reason, on_message, calls = seen[0]
    assert reason is StopReason.REFUSAL and on_message.reason is StopReason.REFUSAL and calls == ()


async def test_the_run_returns_a_refusal_as_a_message_and_does_not_raise():
    model = Scripted(step("Nope.", raw="refusal"))
    agent = await Agent.create(model, config=CONFIG)

    message = await agent.run("Question")

    assert message.content == "Nope." and message.stop.reason is StopReason.REFUSAL
    with pytest.raises(AgentStopError):
        message.raise_for_stop()


# ------------------------------------------------------------------ replaying


async def history_after_refusal(extension):
    model = Scripted(
        step("partial before the refusal", raw="refusal"),
        step("Fine.", raw="stop"),
    )
    agent = await Agent.create(
        model, config=CONFIG, extensions=[InMemoryMessageAccumulator(), *([extension] if extension else [])]
    )
    await agent.run("first")
    await agent.run("second")
    return model


async def test_by_default_a_refused_answer_is_sent_to_the_model_again():
    model = await history_after_refusal(None)

    texts = [m.content for m in model.requests[1].messages if isinstance(m, AssistantMessage)]
    assert texts == ["partial before the refusal"]


async def test_keeping_is_the_default_of_the_extension_too():
    model = await history_after_refusal(RefusalReplayExtension())

    assert [m.content for m in model.requests[1].messages if isinstance(m, AssistantMessage)] == [
        "partial before the refusal"
    ]


async def test_dropping_leaves_the_refused_answer_out_of_the_request_only():
    extension = RefusalReplayExtension(replay=RefusalReplay.DROP)
    model = Scripted(step("partial before the refusal", raw="refusal"), step("Fine.", raw="stop"))
    agent = await Agent.create(model, config=CONFIG, extensions=[InMemoryMessageAccumulator(), extension])
    await agent.run("first")

    await agent.run("second")

    sent = model.requests[1].messages
    assert not any(isinstance(m, AssistantMessage) for m in sent)
    assert [m.content for m in sent if isinstance(m, UserMessage)] == ["first", "second"]
    kept = agent.get_state(CONFIG.session_id).messages
    assert any(isinstance(m, AssistantMessage) and m.content == "partial before the refusal" for m in kept)


async def test_dropping_does_not_touch_answers_that_finished():
    extension = RefusalReplayExtension("drop")
    model = Scripted(step("a real answer", raw="stop"), step("Fine.", raw="stop"))
    agent = await Agent.create(model, config=CONFIG, extensions=[InMemoryMessageAccumulator(), extension])
    await agent.run("first")
    await agent.run("second")

    assert [m.content for m in model.requests[1].messages if isinstance(m, AssistantMessage)] == ["a real answer"]


async def test_other_reasons_can_be_dropped_too():
    extension = RefusalReplayExtension("drop", reasons=[StopReason.INTERRUPTED])
    model = Scripted(step("half", raw="error"), step("Fine.", raw="stop"))
    agent = await Agent.create(model, config=CONFIG, extensions=[InMemoryMessageAccumulator(), extension])
    await agent.run("first")
    await agent.run("second")

    assert not any(isinstance(m, AssistantMessage) for m in model.requests[1].messages)


def test_the_replay_setting_is_validated():
    with pytest.raises(ValueError):
        RefusalReplayExtension("sometimes")
    with pytest.raises(ValueError):
        RefusalReplayExtension(reasons=["not-a-reason"])


# ------------------------------------------------------------------ subagents


def test_a_subagent_report_says_so_when_the_subagent_did_not_finish():
    refused = AssistantMessage(content="", stop=classify_stop("refusal", category="cyber", message="Declined."))
    partial = AssistantMessage(content="half a report", stop=classify_stop("length"))
    fine = AssistantMessage(content="whole report", stop=None)

    assert subagent_report(fine) == "whole report"
    assert subagent_report(AssistantMessage(content="x", stop=classify_stop("stop"))) == "x"
    assert subagent_report(refused).startswith("[The subagent stopped before finishing: refusal (cyber): Declined.")
    assert "incomplete" in subagent_report(refused)
    text = subagent_report(partial)
    assert text.startswith("[The subagent stopped") and text.endswith("half a report")


def test_the_subagent_result_carries_the_stop_as_data_for_the_parent_model():
    stop = classify_stop("refusal", category="cyber")
    result = SubAgentResult(session_id="s", parent_session_id="p", subagent_type="explore", content="x", stop=stop)

    dumped = result.model_dump(mode="json")

    assert dumped["stop"]["reason"] == "refusal" and dumped["stop"]["category"] == "cyber"
    assert SubAgentResult(session_id="s", parent_session_id="p", subagent_type="e", content="x").stop is None


def task(**fields) -> SubAgentTask:
    base = {
        "task_id": "t1",
        "parent_session_id": "p",
        "session_id": "c",
        "subagent_type": "explore",
        "description": "look",
        "prompt": "go",
        "status": SubAgentTaskStatus.COMPLETED,
        "content": "report",
    }
    return SubAgentTask(**{**base, **fields})


def test_a_background_task_that_stopped_early_is_still_completed_but_says_what_happened():
    extension = AsyncSubAgentExtension(
        [SubAgentDefinition(name="explore", description="look around", system_prompt="You explore.", model=Scripted())]
    )
    early = task(stop=classify_stop("refusal", category="cyber", message="Declined."))
    clean = task()

    assert clean.stop is None
    info = extension.task_info(early, include_output=True)
    assert info.status is SubAgentTaskStatus.COMPLETED and info.stop.reason is StopReason.REFUSAL
    assert "incomplete" in info.note and "refusal (cyber)" in info.note
    assert "ended early (refusal (cyber): Declined.)" in extension.completion_notice(early)
    assert extension.completion_notice(clean).endswith("Call task_output to read its report.")
    assert extension.task_info(clean).stop is None and extension.task_info(clean).note.startswith("Finished")
