"""Resuming a session that a failure, a stop, or a crash interrupted.

The history is the only record of how a request ended, so these tests build the histories
that each kind of interruption leaves and check what the model is shown, what is written
back, and that nothing is ever run again on its own.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.client import AgentClient, create_agent
from zett_agent.dispatcher import AgentEventDispatcher
from zett_agent.events import AgentEventType, AgentPhase
from zett_agent.exceptions import AgentError, AgentProtocolError, AgentResumeError
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.jsonl import JSONLExtension
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.sqlite import SQLiteSessionExtension
from zett_agent.messages import AgentMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.resume import ResumeKind, ResumePoint, classify_resume
from zett_agent.stop import StopInfo, StopReason, classify_stop
from zett_agent.sync import SyncAgent
from zett_agent.tools.base import tool

RAN: list[str] = []


@tool(guidelines="Use to write a file.")
async def write(name: str) -> str:
    """Write one file.

    Args:
        name: File name.
    """
    RAN.append(name)
    return f"wrote {name}"


class Scripted:
    """Answer each request with the next scripted step, or fail with it, and keep every request."""

    retry = RetryOptions(max_retries=0)

    def __init__(self, *steps: AssistantMessage | Exception) -> None:
        self.steps = list(steps)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        yield ModelEvent.completed(ModelResponse(step, stop=step.stop))


def call(call_id: str, name: str = "a") -> ToolCall:
    return ToolCall(call_id, "write", {"name": name})


def result(call_id: str, content: str = "wrote", *, success: bool = True) -> ToolMessage:
    return ToolMessage(tool_call_id=call_id, name="write", content=content, success=success)


def stopped(content: str, reason: StopReason) -> AssistantMessage:
    return AssistantMessage(content=content, stop=StopInfo(reason, raw=reason.value))


USER = UserMessage(content="Please write two files")


def dialogue(messages) -> list:
    """The messages without the leading system instructions, so positions mean the same everywhere."""
    return [message for message in messages if not isinstance(message, SystemMessage)]


def kinds(messages) -> list[str]:
    """One word per message, so a history reads as a sequence."""
    out = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, AssistantMessage):
            out.append("assistant+tools" if message.tool_calls else "assistant")
        elif isinstance(message, ToolMessage):
            out.append("tool" if message.success else "tool!")
        elif isinstance(message, AgentMessage):
            out.append("agent")
        else:
            out.append("user")
    return out


@pytest.fixture(autouse=True)
def _forget_runs():
    RAN.clear()


@pytest.fixture
async def storage(tmp_path):
    extension = SQLiteSessionExtension(tmp_path / "resume.sqlite")
    yield extension
    await extension.storage.close()


async def seed(storage: SQLiteSessionExtension, session_id: str, messages) -> None:
    """Write a history the way an interrupted process left it."""
    await storage.storage.create_session(session_id=session_id)
    for message in messages:
        await storage.storage.append(session_id, "interrupted-request", message)


async def agent_for(storage, session_id: str, *steps, **options):
    model = Scripted(*steps)
    agent = await Agent.create(model, config=AgentRunConfig(session_id), tools=[write], extensions=[storage], **options)
    return agent, model


async def log(storage, session_id: str) -> list:
    return [record.message for record in await storage.list_raw_messages(session_id)]


# ------------------------------------------------------------------ reading a tail


@pytest.mark.parametrize(
    ("history", "kind"),
    [
        ([USER], ResumeKind.CALL_MODEL),
        ([AgentMessage(content="internal")], ResumeKind.CALL_MODEL),
        ([USER, AssistantMessage(tool_calls=(call("1"),)), result("1")], ResumeKind.CALL_MODEL),
        ([USER, AssistantMessage(tool_calls=(call("1"), call("2"))), result("1"), result("2")], ResumeKind.CALL_MODEL),
        # An in-process cancel already answered the call, so only the model is left to call.
        (
            [USER, AssistantMessage(tool_calls=(call("1"),)), result("1", "Request cancelled", success=False)],
            ResumeKind.CALL_MODEL,
        ),
        ([USER, stopped("part", StopReason.PAUSED)], ResumeKind.CALL_MODEL),
        ([USER, AssistantMessage(tool_calls=(call("1"),))], ResumeKind.RECOVER_TOOLS),
        ([USER, AssistantMessage(tool_calls=(call("1"), call("2"))), result("1")], ResumeKind.RECOVER_TOOLS),
        ([USER, stopped("cut", StopReason.INTERRUPTED)], ResumeKind.RETRY_RESPONSE),
        ([USER, stopped("cut", StopReason.MAX_OUTPUT)], ResumeKind.CONTINUE_RESPONSE),
    ],
    ids=[
        "user",
        "internal",
        "one-tool-done",
        "two-tools-done",
        "cancelled-tool",
        "paused",
        "no-results",
        "partial-results",
        "interrupted",
        "max-output",
    ],
)
def test_the_last_message_says_how_the_request_was_interrupted(history, kind):
    assert classify_resume(history).kind is kind


def test_system_messages_are_not_the_tail():
    history = [SystemMessage(content="be brief"), USER, SystemMessage(content="late instruction")]

    assert classify_resume(history).kind is ResumeKind.CALL_MODEL


def test_only_the_calls_without_a_result_are_reported():
    point = classify_resume([USER, AssistantMessage(tool_calls=(call("1", "a"), call("2", "b"))), result("2")])

    assert [item.id for item in point.unanswered_tool_calls] == ["1"]
    assert "write" in point.description and point.stop is None


def test_the_point_carries_the_stop_it_resumes_from():
    stop = classify_stop("length")

    point = classify_resume([USER, AssistantMessage(content="cut", stop=stop)])

    assert point.stop is stop and "output limit" in point.description


@pytest.mark.parametrize(
    ("history", "reason"),
    [
        ([], "empty"),
        ([SystemMessage(content="only instructions")], "empty"),
        ([USER, AssistantMessage(content="All done")], "completed"),
        ([USER, AssistantMessage(content="All done", stop=classify_stop("stop"))], "completed"),
        ([USER, stopped("", StopReason.REFUSAL)], "refused"),
        ([USER, stopped("part", StopReason.CONTEXT_FULL)], "context_full"),
        ([USER, stopped("part", StopReason.OTHER)], "unknown_stop"),
        ([result("1")], "malformed"),
        ([USER, result("1")], "malformed"),
    ],
)
def test_a_history_that_cannot_be_resumed_says_why(history, reason):
    with pytest.raises(AgentResumeError) as raised:
        classify_resume(history)

    assert raised.value.reason == reason and isinstance(raised.value, AgentError)
    assert str(raised.value)


def test_a_context_that_was_full_is_an_error_and_not_a_retry():
    """Resuming would send the same history again and fill the window again."""
    with pytest.raises(AgentResumeError, match="context window"):
        classify_resume([USER, stopped("part", StopReason.CONTEXT_FULL)])


# ----------------------------------------------------- asking without changing anything


async def test_resume_point_reads_the_session_and_changes_nothing(storage):
    await seed(storage, "s", [USER, AssistantMessage(tool_calls=(call("1"),))])
    agent, model = await agent_for(storage, "s")

    point = await agent.resume_point()

    assert isinstance(point, ResumePoint) and point.kind is ResumeKind.RECOVER_TOOLS
    assert kinds(await log(storage, "s")) == ["user", "assistant+tools"]  # nothing was written
    assert model.requests == [] and RAN == []  # no model call, no tool run


@pytest.mark.parametrize(
    "history",
    [
        [USER, AssistantMessage(content="All done")],
        [USER, stopped("", StopReason.REFUSAL)],
        [USER, stopped("part", StopReason.CONTEXT_FULL)],
    ],
    ids=["completed", "refused", "context-full"],
)
async def test_resume_point_is_none_when_there_is_nothing_to_resume(storage, history):
    await seed(storage, "s", history)
    agent, _ = await agent_for(storage, "s")

    assert await agent.resume_point() is None


async def test_a_session_that_does_not_exist_has_nothing_to_resume(storage):
    agent, _ = await agent_for(storage, "never-used")

    assert await agent.resume_point() is None
    with pytest.raises(AgentResumeError) as raised:
        await agent.resume()
    assert raised.value.reason == "empty"


async def test_asking_leaves_the_session_free_to_run(storage):
    await seed(storage, "s", [USER])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="fine"))

    await agent.resume_point()
    await agent.resume_point()

    assert (await agent.resume()).content == "fine"


async def test_asking_while_the_session_is_busy_is_refused_like_any_second_request(storage):
    gate = asyncio.Event()

    class Slow:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            await gate.wait()
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="late")))

    agent = await Agent.create(Slow(), config=AgentRunConfig("busy"), extensions=[storage])
    running = asyncio.create_task(agent.run("hello"))
    await asyncio.sleep(0.05)
    try:
        with pytest.raises(AgentProtocolError, match="current phase"):
            await agent.resume_point()
    finally:
        gate.set()
        await running


# ----------------------------------------------------------- what resuming does


async def test_a_model_call_that_never_got_an_answer_is_simply_made_again(storage):
    """The network case: the user message was saved, the provider never answered."""
    await seed(storage, "s", [USER])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="Done."))

    answer = await agent.resume()

    assert answer.content == "Done."
    assert kinds(model.requests[0].messages) == ["user"]  # no second user message, no marker
    assert kinds(await log(storage, "s")) == ["user", "assistant"]


async def test_the_same_failure_end_to_end_on_one_agent(storage):
    agent, model = await agent_for(storage, "s", httpx.ConnectError("offline"), AssistantMessage(content="Back."))

    with pytest.raises(httpx.ConnectError):
        await agent.run("Please write two files")
    assert (await agent.resume_point()).kind is ResumeKind.CALL_MODEL
    answer = await agent.resume()

    assert answer.content == "Back."
    assert kinds(model.requests[1].messages) == ["user"]
    assert kinds(await log(storage, "s")) == ["user", "assistant"]
    assert await agent.resume_point() is None  # and it is finished now


async def test_a_failure_after_the_tools_ran_is_resumed_without_running_them_again(storage):
    agent, model = await agent_for(
        storage,
        "s",
        AssistantMessage(tool_calls=(call("1", "a"), call("2", "b"))),
        httpx.ConnectError("offline"),
        AssistantMessage(content="Both files written."),
    )
    with pytest.raises(httpx.ConnectError):
        await agent.run("Please write two files")
    assert RAN == ["a", "b"]

    answer = await agent.resume()

    assert answer.content == "Both files written." and RAN == ["a", "b"]
    assert kinds(model.requests[2].messages) == ["user", "assistant+tools", "tool", "tool"]


async def test_tools_a_crash_interrupted_are_reported_to_the_model_and_never_run(storage):
    await seed(
        storage,
        "s",
        [USER, AssistantMessage(tool_calls=(call("1", "a"), call("2", "b"))), result("1", "wrote a")],
    )
    agent, model = await agent_for(storage, "s", AssistantMessage(content="Checked; b is missing, writing it."))

    answer = await agent.resume()

    assert answer.content.startswith("Checked") and RAN == []
    sent = dialogue(model.requests[0].messages)
    assert kinds(sent) == ["user", "assistant+tools", "tool", "tool!"]
    assert sent[2].content == "wrote a"  # the result that did arrive is kept, not forgotten
    missing = sent[3]
    assert missing.tool_call_id == "2" and missing.success is False
    assert "may or may not have completed" in missing.content and "Check the current state" in missing.content


async def test_what_the_model_was_told_is_written_to_the_log(storage):
    await seed(storage, "s", [USER, AssistantMessage(tool_calls=(call("1"),))])
    agent, _ = await agent_for(storage, "s", AssistantMessage(content="ok"))

    await agent.resume()

    assert kinds(await log(storage, "s")) == ["user", "assistant+tools", "tool!", "assistant"]
    assert await agent.resume_point() is None


async def test_a_response_cut_off_by_a_failure_is_kept_and_the_model_is_told_to_start_again(storage):
    await seed(storage, "s", [USER, stopped("Here is the start of", StopReason.INTERRUPTED)])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="The whole answer."))

    await agent.resume()

    sent = dialogue(model.requests[0].messages)
    assert kinds(sent) == ["user", "assistant", "agent"]
    assert sent[1].content == "Here is the start of"
    assert "cut off by an interruption" in sent[2].content and "again from the beginning" in sent[2].content
    assert sent[2].attributes == {"resume": "retry_response"}
    assert kinds(await log(storage, "s")) == ["user", "assistant", "agent", "assistant"]


async def test_a_response_that_hit_the_output_limit_is_continued_not_restarted(storage):
    await seed(storage, "s", [USER, stopped("A very long answer that", StopReason.MAX_OUTPUT)])
    agent, model = await agent_for(storage, "s", AssistantMessage(content=" was finished."))

    await agent.resume()

    sent = dialogue(model.requests[0].messages)
    assert kinds(sent) == ["user", "assistant", "agent"]
    assert "output limit" in sent[2].content and "Continue exactly where it stopped" in sent[2].content
    assert sent[2].attributes == {"resume": "continue_response"}


async def test_a_paused_response_is_continued_by_calling_the_model_again(storage):
    await seed(storage, "s", [USER, stopped("searching", StopReason.PAUSED)])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="found it"))

    await agent.resume()

    assert kinds(model.requests[0].messages) == ["user", "assistant"]  # nothing is added


async def test_a_cancelled_tool_leaves_a_history_that_is_resumed_by_calling_the_model(storage):
    entered = asyncio.Event()

    @tool(guidelines="Wait until cancelled.")
    async def hang() -> str:
        """Block until the request is cancelled."""
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    model = Scripted(AssistantMessage(tool_calls=(ToolCall("h1", "hang", {}),)), AssistantMessage(content="Stopped."))
    agent = await Agent.create(model, config=AgentRunConfig("c"), tools=[hang], extensions=[storage])
    running = asyncio.create_task(agent.run("start"))
    await asyncio.wait_for(entered.wait(), 2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert (await agent.resume_point()).kind is ResumeKind.CALL_MODEL
    answer = await agent.resume()

    assert answer.content == "Stopped."
    assert kinds(model.requests[1].messages) == ["user", "assistant+tools", "tool!"]
    assert model.requests[1].messages[-1].content == "Request cancelled before tool completion"


@pytest.mark.parametrize(
    ("history", "reason"),
    [
        ([USER, AssistantMessage(content="All done")], "completed"),
        ([USER, stopped("", StopReason.REFUSAL)], "refused"),
        ([USER, stopped("part", StopReason.CONTEXT_FULL)], "context_full"),
    ],
)
async def test_resume_raises_when_there_is_nothing_to_resume_and_calls_nothing(storage, history, reason):
    await seed(storage, "s", history)
    agent, model = await agent_for(storage, "s")

    with pytest.raises(AgentResumeError) as raised:
        await agent.resume()

    assert raised.value.reason == reason and model.requests == []
    assert kinds(await log(storage, "s")) == kinds(history)  # and nothing was written


async def test_a_failed_resume_attempt_leaves_the_agent_usable(storage):
    await seed(storage, "s", [USER, AssistantMessage(content="All done")])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="a new answer"))

    with pytest.raises(AgentResumeError):
        await agent.resume()

    assert agent.get_state("s").phase is AgentPhase.FAILED
    assert (await agent.run("another question")).content == "a new answer"


async def test_resuming_twice_does_the_work_once(storage):
    await seed(storage, "s", [USER, AssistantMessage(tool_calls=(call("1"),))])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="ok"))

    await agent.resume()
    with pytest.raises(AgentResumeError) as raised:
        await agent.resume()

    assert raised.value.reason == "completed" and len(model.requests) == 1


async def test_a_resume_that_fails_again_can_be_resumed_again(storage):
    await seed(storage, "s", [USER])
    agent, model = await agent_for(
        storage, "s", httpx.ConnectError("still offline"), AssistantMessage(content="at last")
    )

    with pytest.raises(httpx.ConnectError):
        await agent.resume()
    assert (await agent.resume()).content == "at last"
    assert kinds(await log(storage, "s")) == ["user", "assistant"]


async def test_a_resume_that_fails_after_recovering_tools_does_not_stack_markers(storage):
    await seed(storage, "s", [USER, AssistantMessage(tool_calls=(call("1"),))])
    agent, model = await agent_for(storage, "s", httpx.ConnectError("offline"), AssistantMessage(content="ok"))

    with pytest.raises(httpx.ConnectError):
        await agent.resume()
    assert kinds(await log(storage, "s")) == ["user", "assistant+tools", "tool!"]
    await agent.resume()

    assert kinds(await log(storage, "s")) == ["user", "assistant+tools", "tool!", "assistant"]


# ------------------------------------------------- a plain run after a crash is valid


async def test_a_new_message_after_a_crash_closes_the_interrupted_tools_first(storage):
    await seed(
        storage,
        "s",
        [USER, AssistantMessage(tool_calls=(call("1", "a"), call("2", "b"))), result("1", "wrote a")],
    )
    agent, model = await agent_for(storage, "s", AssistantMessage(content="Hello."))

    await agent.run("hello again")

    sent = dialogue(model.requests[0].messages)
    assert kinds(sent) == ["user", "assistant+tools", "tool", "tool!", "user"]
    assert sent[2].content == "wrote a" and RAN == []
    assert kinds(await log(storage, "s")) == ["user", "assistant+tools", "tool", "tool!", "user", "assistant"]


async def test_a_normal_second_message_adds_nothing(storage):
    agent, model = await agent_for(storage, "s", AssistantMessage(content="one"), AssistantMessage(content="two"))

    await agent.run("first")
    await agent.run("second")

    assert kinds(model.requests[1].messages) == ["user", "assistant", "user"]


# ------------------------------------------------------------- persistence tail rule


async def test_a_broken_batch_in_the_middle_of_the_history_is_still_left_out(storage):
    await seed(
        storage,
        "s",
        [
            USER,
            AssistantMessage(tool_calls=(call("1"),)),  # never answered, and the history went on
            UserMessage(content="a later message"),
            AssistantMessage(content="a later answer"),
        ],
    )
    agent, model = await agent_for(storage, "s", AssistantMessage(content="fine"))

    await agent.run("next")

    assert kinds(model.requests[0].messages) == ["user", "user", "assistant", "user"]


async def test_results_that_answer_nothing_are_not_kept_at_the_end(storage):
    await seed(storage, "s", [USER, AssistantMessage(tool_calls=(call("1"),)), result("1"), result("stray")])
    agent, model = await agent_for(storage, "s", AssistantMessage(content="fine"))

    await agent.run("next")

    assert kinds(model.requests[0].messages) == ["user", "user"]


# ---------------------------------------------------------------- events and hooks


async def test_resume_streams_the_same_events_as_a_run(storage):
    await seed(storage, "s", [USER])
    agent, _ = await agent_for(storage, "s", AssistantMessage(content="Done."))

    events = [event async for event in agent.stream_resume()]

    types = [event.type for event in events]
    assert types[0] is AgentEventType.MODEL_STARTED and types[-1] is AgentEventType.RUN_COMPLETED
    assert AgentEventType.MODEL_COMPLETED in types
    assert events[-1].message.content == "Done."


async def test_resume_reports_an_abnormal_stop_like_a_run(storage):
    await seed(storage, "s", [USER])
    agent, _ = await agent_for(storage, "s", stopped("cut", StopReason.MAX_OUTPUT))

    events = [event async for event in agent.stream_resume()]

    assert [event.stop.reason for event in events if event.type is AgentEventType.MODEL_STOPPED] == [
        StopReason.MAX_OUTPUT
    ]
    assert (await agent.resume_point()).kind is ResumeKind.CONTINUE_RESPONSE  # and it can be resumed again


async def test_the_request_hooks_run_but_the_message_hook_does_not(storage):
    seen: list[str] = []

    class Watch(AgentExtension):
        async def on_message(self, context):
            seen.append("on_message")

        async def before_run(self, context):
            seen.append(f"before_run:{context.input_message}")

        async def after_run(self, context, result):
            seen.append("after_run")

    await seed(storage, "s", [USER])
    model = Scripted(AssistantMessage(content="Done."))
    agent = await Agent.create(model, config=AgentRunConfig("s"), extensions=[storage, Watch()])

    await agent.resume()

    assert seen == ["before_run:None", "after_run"]


async def test_metadata_and_tags_reach_the_log_of_a_resumed_request(storage):
    await seed(storage, "s", [USER])
    agent, _ = await agent_for(storage, "s", AssistantMessage(content="Done."))

    await agent.resume(metadata={"source": "resume"}, tags={"kind": "retry"})

    records = await storage.list_raw_messages("s")
    assert records[-1].metadata == {"source": "resume"} and records[-1].tags == {"kind": "retry"}


async def test_a_resume_has_its_own_request_id(storage):
    await seed(storage, "s", [USER])
    agent, _ = await agent_for(storage, "s", AssistantMessage(content="Done."))

    await agent.resume()

    records = await storage.list_raw_messages("s")
    assert records[0].request_id == "interrupted-request" and records[-1].request_id != "interrupted-request"


async def test_the_jsonl_extension_records_a_resumed_turn_from_the_last_user_message(tmp_path):
    jsonl = JSONLExtension(tmp_path / "turns")
    model = Scripted(httpx.ConnectError("offline"), AssistantMessage(content="Back."))
    agent = await Agent.create(model, config=AgentRunConfig("j"), extensions=[InMemoryMessageAccumulator(), jsonl])

    with pytest.raises(httpx.ConnectError):
        await agent.run("Please write two files")
    await agent.resume()

    import json

    lines = [json.loads(line) for line in jsonl.session_path("j").read_text().splitlines()]
    assert [line["status"] for line in lines] == ["failed", "completed"]
    assert len(lines[1]["messages"]) == 2  # the user message, then the answer


# ------------------------------------------------------------------- the client


class Recorder(AgentEventDispatcher):
    def __init__(self) -> None:
        self.types: list[AgentEventType] = []

    async def on_model_completed_event(self, event):
        self.types.append(event.type)

    async def on_run_completed_event(self, event):
        self.types.append(event.type)


async def test_the_client_resumes_with_callbacks_and_returns_the_answer(storage):
    await seed(storage, "s", [USER])
    recorder = Recorder()
    client = await create_agent(
        Scripted(AssistantMessage(content="Done.")),
        config=AgentRunConfig("s"),
        extensions=[storage],
    )
    client = AgentClient(client.agent, event_dispatcher=recorder)

    assert (await client.resume_point()).kind is ResumeKind.CALL_MODEL
    answer = await client.resume()

    assert answer.content == "Done."
    assert recorder.types == [AgentEventType.MODEL_COMPLETED, AgentEventType.RUN_COMPLETED]


async def test_the_client_streams_a_resume(storage):
    await seed(storage, "s", [USER])
    agent, _ = await agent_for(storage, "s", AssistantMessage(content="Done."))
    client = AgentClient(agent)

    events = [event async for event in client.stream_resume()]

    assert events[-1].type is AgentEventType.RUN_COMPLETED


async def test_the_client_raises_when_there_is_nothing_to_resume(storage):
    await seed(storage, "s", [USER, AssistantMessage(content="All done")])
    agent, _ = await agent_for(storage, "s")

    with pytest.raises(AgentResumeError):
        await AgentClient(agent).resume()


# ------------------------------------------------------------------- synchronous


def test_the_synchronous_agent_resumes(tmp_path):
    model = Scripted(httpx.ConnectError("offline"), AssistantMessage(content="Back."))
    with SyncAgent(
        model, config=AgentRunConfig("sync"), extensions=[SQLiteSessionExtension(tmp_path / "s.sqlite")]
    ) as agent:
        with pytest.raises(httpx.ConnectError):
            agent.run("hello")

        point = agent.resume_point()
        assert point is not None and point.kind is ResumeKind.CALL_MODEL
        assert agent.resume().content == "Back."
        assert agent.resume_point() is None


def test_the_synchronous_agent_streams_a_resume(tmp_path):
    model = Scripted(httpx.ConnectError("offline"), AssistantMessage(content="Back."))
    with SyncAgent(
        model, config=AgentRunConfig("sync"), extensions=[SQLiteSessionExtension(tmp_path / "s.sqlite")]
    ) as agent:
        with pytest.raises(httpx.ConnectError):
            agent.run("hello")
        with agent.stream_resume() as events:
            types = [event.type for event in events]

    assert types[-1] is AgentEventType.RUN_COMPLETED


def test_the_synchronous_agent_raises_when_there_is_nothing_to_resume(tmp_path):
    with SyncAgent(
        Scripted(AssistantMessage(content="Hi.")), extensions=[SQLiteSessionExtension(tmp_path / "s.sqlite")]
    ) as agent:
        agent.run("hello")
        with pytest.raises(AgentResumeError):
            agent.resume()


# --------------------------------------------------------- failures while asking


class BrokenStorage(AgentExtension):
    """Fail while the session is being restored, as an unreachable database does."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def on_state(self, context):
        raise self.error


async def test_a_failure_while_reading_the_session_reaches_the_caller_and_frees_the_session():
    agent = await Agent.create(
        Scripted(AssistantMessage(content="fine")),
        config=AgentRunConfig("s"),
        extensions=[BrokenStorage(OSError("database is locked"))],
    )

    with pytest.raises(OSError, match="locked"):
        await agent.resume_point()

    assert agent.get_state("s").phase is AgentPhase.FAILED
    with pytest.raises(OSError):  # the session was released, so asking again reaches the same failure
        await agent.resume_point()


async def test_cancelling_the_question_releases_the_session():
    gate = asyncio.Event()

    class Hang(AgentExtension):
        async def on_state(self, context):
            await gate.wait()

    agent = await Agent.create(Scripted(), config=AgentRunConfig("s"), extensions=[Hang()])
    asking = asyncio.create_task(agent.resume_point())
    await asyncio.sleep(0.05)
    asking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asking

    assert agent.get_state("s").phase is AgentPhase.CANCELLED
    gate.set()
    assert await agent.resume_point() is None


async def test_a_bad_parallel_setting_is_rejected_before_anything_starts(storage):
    agent, model = await agent_for(storage, "s")

    with pytest.raises(ValueError, match="parallel_tool_call"):
        await agent.resume(parallel_tool_call="yes")  # type: ignore[arg-type]

    assert model.requests == []
