"""How ``Agent`` is put together: which class owns what, and the helpers its pieces share.

``Agent`` is one public class whose private machinery lives in mixins, one per concern. Nothing here
changes what an agent does; it keeps the structure from drifting in ways that would break something
else, such as the documented methods leaving the class, or a lifecycle hook being named wrongly where
the compiler cannot see it.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from zett_agent._agent.hooks import ExtensionHooksMixin
from zett_agent._agent.lifecycle import RequestLifecycleMixin, final_answer
from zett_agent._agent.model_step import ModelStepMixin
from zett_agent._agent.recovery import RecoveryMixin
from zett_agent._agent.tool_runner import ToolRunnerMixin
from zett_agent._agent.turns import TurnLoopMixin
from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.events import AgentEvent, AgentEventType
from zett_agent.exceptions import AgentProtocolError
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.external import ExternalEvent
from zett_agent.messages import AssistantMessage, ImageContent, ImageUrlSource, TextContent, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelEventType, ModelResponse, ModelUsage, RetryOptions
from zett_agent.tools.base import ToolExecutionMode, tool

SOURCE = Path(__file__).resolve().parents[1] / "src" / "zett_agent"
MIXINS = (
    ExtensionHooksMixin,
    RequestLifecycleMixin,
    TurnLoopMixin,
    ModelStepMixin,
    ToolRunnerMixin,
    RecoveryMixin,
)


# ------------------------------------------------------------------ the structure


def test_agent_is_composed_of_one_mixin_per_concern():
    assert all(mixin in Agent.__mro__ for mixin in MIXINS)
    # Each concern lives in a module of its own, so a change to one does not sit beside another.
    assert len({mixin.__module__ for mixin in MIXINS}) == len(MIXINS)


def test_the_documented_methods_are_defined_on_agent_itself():
    """The API reference lists what ``vars(Agent)`` holds, so a method moved into a mixin would vanish from it."""
    public = {
        "create",
        "initialize",
        "get_state",
        "pending",
        "is_conflict",
        "emit_external_event",
        "run",
        "stream",
        "resume",
        "stream_resume",
        "resume_point",
        "compact",
    }

    assert public <= set(vars(Agent))
    for mixin in MIXINS:
        assert not {name for name in vars(mixin) if not name.startswith("_")}, mixin.__name__


def test_the_mixins_declare_no_state_of_their_own():
    """State is assigned in ``Agent.__init__``, so there is one place to read what an agent holds."""
    for mixin in MIXINS:
        assert "__init__" not in vars(mixin), mixin.__name__


def test_the_mixins_do_not_load_agent_when_they_are_imported():
    """They run inside ``Agent``, so they may name it for types but may not import it back."""
    modules = ", ".join(
        f"zett_agent._agent.{name}" for name in ("hooks", "lifecycle", "model_step", "recovery", "tool_runner", "turns")
    )
    code = f"import sys, {modules}; print('zett_agent.agent' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"


def broadcast_hooks() -> set[str]:
    """Every hook name the runtime passes to ``_broadcast``, read from the source."""
    names: set[str] = set()
    for path in [SOURCE / "agent.py", *sorted((SOURCE / "_agent").glob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_broadcast"
                and isinstance(node.args[0], ast.Constant)
            ):
                names.add(node.args[0].value)
    return names


def test_every_hook_the_runtime_broadcasts_is_a_hook_extensions_have():
    """A hook is named by a string, so a typo would only show up when that lifecycle point ran."""
    hooks = broadcast_hooks()

    assert hooks == {
        "on_tool",
        "on_state",
        "on_message",
        "before_run",
        "before_turn",
        "after_turn",
        "before_tool",
        "after_tool",
        "after_model",
        "after_run",
        "on_success",
        "on_error",
    }
    for hook in hooks:
        assert callable(getattr(AgentExtension, hook)), hook


async def test_a_broadcast_calls_every_extension_in_priority_order_and_stops_at_a_failure():
    calls: list[str] = []

    class Mark(AgentExtension):
        def __init__(self, name: str, priority: int, *, fail: bool = False) -> None:
            self.name, self.priority, self.fail = name, priority, fail

        async def before_run(self, context):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(self.name)

    model = Scripted(AssistantMessage(content="fine"))
    agent = await Agent.create(model, extensions=[Mark("late", 20), Mark("early", 10), Mark("broken", 15, fail=True)])

    with pytest.raises(RuntimeError, match="broken"):
        await agent.run("hello")

    assert calls == ["early", "broken"]  # in priority order, and "late" was never reached


# --------------------------------------------------------------------- helpers


class Scripted:
    retry = RetryOptions(max_retries=0)

    def __init__(self, *steps) -> None:
        self.steps = list(steps)

    async def stream(self, request):
        step = self.steps.pop(0)
        if isinstance(step, list):
            for event in step:
                yield event
        else:
            yield ModelEvent.completed(ModelResponse(step))


async def test_a_run_whose_events_end_without_an_answer_is_a_protocol_error():
    async def nothing() -> AsyncIterator[AgentEvent]:
        return
        yield  # pragma: no cover - makes this an async generator

    with pytest.raises(AgentProtocolError, match="final answer"):
        await final_answer(nothing())


async def test_the_final_answer_is_the_one_the_run_completed_with():
    answer = AssistantMessage(content="done")

    async def events() -> AsyncIterator[AgentEvent]:
        yield AgentEvent(AgentEventType.TEXT_DELTA, "s", delta="x")
        yield AgentEvent(AgentEventType.RUN_COMPLETED, "s", message=answer)

    assert await final_answer(events()) is answer


def test_run_options_fall_back_to_the_agents_defaults_and_reject_a_non_boolean():
    agent = Agent(None, parallel_tool_call=False)

    assert agent._resolve_run_options(None, None) == (agent.reasoning_effort, False)
    assert agent._resolve_run_options(None, True)[1] is True
    with pytest.raises(ValueError, match="parallel_tool_call"):
        agent._resolve_run_options(None, "yes")  # type: ignore[arg-type]


# ------------------------------------------------------- what a model may stream


def event(kind: ModelEventType, **fields) -> ModelEvent:
    return ModelEvent(kind, **fields)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (ModelEventType.TOOL_CALL_DELTA, "Missing tool-call delta"),
        (ModelEventType.SERVER_TOOL_STARTED, "Missing server-tool call"),
        (ModelEventType.SERVER_TOOL_INPUT_DELTA, "Missing server-tool input delta"),
        (ModelEventType.SERVER_TOOL_COMPLETED, "Missing server-tool result"),
        (ModelEventType.SERVER_TOOL_FAILED, "Missing server-tool result"),
        (ModelEventType.RESPONSE, "Missing model response"),
    ],
)
async def test_an_event_without_the_payload_its_type_promises_is_a_protocol_error(kind, message):
    agent = await Agent.create(Scripted([event(kind)]), config=AgentRunConfig("s"))

    with pytest.raises(AgentProtocolError, match=message):
        await agent.run("hello")


async def test_events_after_the_final_response_are_a_protocol_error():
    final = ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))
    agent = await Agent.create(Scripted([final, ModelEvent.text("late")]), config=AgentRunConfig("s"))

    with pytest.raises(AgentProtocolError, match="after its final response"):
        await agent.run("hello")


async def test_a_stream_that_ends_without_a_response_is_a_protocol_error():
    agent = await Agent.create(Scripted([ModelEvent.text("part")]), config=AgentRunConfig("s"))

    with pytest.raises(AgentProtocolError, match="without a response"):
        await agent.run("hello")


async def test_every_relayed_event_reaches_the_caller_as_the_event_of_the_same_name():
    from zett_agent.model import ServerToolCall, ServerToolInputDelta, ServerToolResult, ToolCallDelta

    stream = [
        ModelEvent.reasoning("b"),
        ModelEvent.text("a"),
        ModelEvent.tool_call(ToolCallDelta(index=0, id_delta="c1", name_delta="x", arguments_delta="{")),
        ModelEvent.server_tool_started(ServerToolCall("s1", "web_search", {})),
        ModelEvent.server_tool_input(ServerToolInputDelta("s1", "q")),
        ModelEvent.server_tool_completed(ServerToolResult("s1", "web_search", {})),
        ModelEvent.completed(ModelResponse(AssistantMessage(content="ab"), usage=ModelUsage())),
    ]
    agent = await Agent.create(Scripted(stream), config=AgentRunConfig("s"))

    kinds = [item.type for item in [e async for e in agent.stream("hello")]]

    for expected in (
        AgentEventType.TEXT_DELTA,
        AgentEventType.REASONING_DELTA,
        AgentEventType.TOOL_CALL_DELTA,
        AgentEventType.SERVER_TOOL_STARTED,
        AgentEventType.SERVER_TOOL_INPUT_DELTA,
        AgentEventType.SERVER_TOOL_COMPLETED,
    ):
        assert expected in kinds


async def test_an_event_type_the_runtime_does_not_know_is_ignored():
    shown: list[AgentEvent] = []

    class Context:
        async def _emit_event(self, *args, **fields):
            shown.append(args)

    await ModelStepMixin._relay_model_event(Context(), SimpleNamespace(type="from_the_future"))

    assert shown == []


# ----------------------------------------------------------------- tools


def test_a_handler_may_return_content_parts_and_they_are_not_serialized():
    part = TextContent("a note")
    image = ImageContent(ImageUrlSource("https://example.com/a.png"))

    @tool(guidelines="Do nothing.")
    def noop() -> str:
        """Do nothing."""
        return ""

    registered = noop

    assert ToolRunnerMixin._tool_content(part, registered) == [part]
    assert ToolRunnerMixin._tool_content(image, registered) == [image]
    assert ToolRunnerMixin._tool_content([part, image], registered) == [part, image]
    assert ToolRunnerMixin._tool_content({"a": 1}, registered) == '{"a": 1}'
    assert ToolRunnerMixin._tool_content([], registered) == "[]"


async def test_steering_between_two_serial_tools_skips_the_second():
    ran: list[str] = []
    steered = False

    @tool(guidelines="Record.", execution_mode=ToolExecutionMode.SERIAL)
    async def record(value: str) -> str:
        """Record a value.

        Args:
            value: What to record.
        """
        ran.append(value)
        return value

    calls = (ToolCall("a", "record", {"value": "a"}), ToolCall("b", "record", {"value": "b"}))
    agent = await Agent.create(
        Scripted(AssistantMessage(tool_calls=calls), AssistantMessage(content="redirected")),
        config=AgentRunConfig("s", request_id="r"),
        tools=[record],
    )

    async for item in agent.stream("original"):
        if item.type is AgentEventType.TOOL_COMPLETED and not steered:
            steered = True
            event = ExternalEvent("steering_message", {"session_id": "s", "request_id": "r", "content": "new task"})
            assert await asyncio.to_thread(agent.emit_external_event, event)

    assert ran == ["a"]  # the second call was skipped, not run
    skipped = [m for m in agent.get_state("s").messages if isinstance(m, ToolMessage) and not m.success]
    assert [m.tool_call_id for m in skipped] == ["b"]


# -------------------------------------------------------------------- recovery


async def test_only_the_latest_tool_batch_is_closed_and_an_older_broken_one_is_left_alone():
    appended: list[ToolMessage] = []

    class Context:
        state = SimpleNamespace(
            messages=[
                AssistantMessage(tool_calls=(ToolCall("old", "x", {}),)),  # never answered, and history went on
                AssistantMessage(content="a later answer"),
                AssistantMessage(tool_calls=(ToolCall("new", "x", {}),)),
            ]
        )

        async def append_message(self, message, timing):
            appended.append(message)

    await RecoveryMixin._close_unanswered_tool_calls(SimpleNamespace(), Context(), "stopped")

    assert [message.tool_call_id for message in appended] == ["new"]
    assert appended[0].content == "stopped" and appended[0].success is False
