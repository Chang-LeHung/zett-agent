"""Stop reasons: the vocabulary, how the loop acts on each, and how the host is told."""

from __future__ import annotations

import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.dispatcher import AgentEventDispatcher
from zett_agent.events import AgentEvent, AgentEventType
from zett_agent.exceptions import AgentStopError
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.messages import AssistantMessage, ToolCall
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.stop import StopInfo, StopReason, classify_stop
from zett_agent.tools.base import tool

CONFIG = AgentRunConfig("stop-reasons")


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
