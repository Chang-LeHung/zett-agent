"""Background subagents: starting, inspecting, waiting, cancelling, notifying, and subclassing."""

import asyncio
import json

import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.async_subagent import (
    AsyncSubAgentExtension,
    SubAgentTask,
    SubAgentTaskInfo,
    SubAgentTaskStatus,
)
from zett_agent.extensions.subagent import SubAgentDefinition, SubAgentExtension
from zett_agent.messages import AgentMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.tools.base import tool

START, LIST, STATUS, OUTPUT, CANCEL = "task_start", "task_list", "task_status", "task_output", "task_cancel"


class ChildModel:
    """A child that waits for ``gate`` (when given) and then reports or fails."""

    def __init__(self, gate: asyncio.Event | None = None, *, fail: bool = False) -> None:
        self.gate = gate
        self.fail = fail
        self.started = asyncio.Event()
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("child exploded")
        prompt = next(m for m in request.messages if isinstance(m, UserMessage)).content
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=f"report for: {prompt}")))


class ParentModel:
    """A parent that answers each request with the next scripted step."""

    def __init__(self, *steps) -> None:
        self.steps = list(steps)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        step = self.steps.pop(0) if self.steps else "done"
        if callable(step):
            step = await step()
        message = AssistantMessage(tool_calls=(step,)) if isinstance(step, ToolCall) else AssistantMessage(content=step)
        yield ModelEvent.completed(ModelResponse(message))


def definition(child: ChildModel, name: str = "worker") -> SubAgentDefinition:
    return SubAgentDefinition(
        name=name, description="do background work.", system_prompt="You are a worker.", model=child
    )


def start_call(prompt: str = "dig", call_id: str = "c1", subagent: str = "worker") -> ToolCall:
    return ToolCall(call_id, START, {"description": "dig in", "prompt": prompt, "subagent_type": subagent})


def last_tool_result(model: ParentModel) -> str:
    return next(m for m in reversed(model.requests[-1].messages) if isinstance(m, ToolMessage)).content


async def make_agent(parent, extension, session="parent", **options):
    return await Agent.create(parent, config=AgentRunConfig(session), extensions=[extension], **options)


async def test_start_returns_at_once_and_output_waits_for_the_report():
    gate = asyncio.Event()
    child = ChildModel(gate)
    extension = AsyncSubAgentExtension([definition(child)])
    parent = ParentModel(start_call(), "started")
    agent = await make_agent(parent, extension)

    assert (await agent.run("go")).content == "started"
    [task] = extension.tasks("parent")
    assert task.status is SubAgentTaskStatus.RUNNING
    assert '"status":"running"' in last_tool_result(parent).replace(" ", "")

    gate.set()
    await extension.wait(task, timeout=5)
    info = extension.task_info(task, include_output=True)
    assert (info.status, info.content) == (SubAgentTaskStatus.COMPLETED, "report for: dig")
    assert task.parent_session_id == "parent" and task.session_id != "parent"
    await extension.close()


async def test_the_parent_can_wait_for_a_report_through_the_output_tool():
    child = ChildModel()
    extension = AsyncSubAgentExtension([definition(child)])
    parent = ParentModel(
        start_call(),
        ToolCall("c2", OUTPUT, {"task_id": "task-1", "wait_seconds": 5}),
        "all done",
    )
    agent = await make_agent(parent, extension)

    assert (await agent.run("go")).content == "all done"
    assert "report for: dig" in last_tool_result(parent)
    await extension.close()


async def test_a_failing_child_is_a_failed_task_not_a_failed_tool_call():
    child = ChildModel(fail=True)
    extension = AsyncSubAgentExtension([definition(child)])
    parent = ParentModel(start_call(), ToolCall("c2", OUTPUT, {"task_id": "task-1", "wait_seconds": 5}), "ok")
    agent = await make_agent(parent, extension)

    await agent.run("go")

    result = next(m for m in parent.requests[-1].messages if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert result.success
    assert "child exploded" in result.content and '"status":"failed"' in result.content.replace(" ", "")
    [task] = extension.tasks("parent")
    assert task.status is SubAgentTaskStatus.FAILED and task.content is None


async def test_cancel_stops_a_running_task_and_leaves_a_finished_one_alone():
    child = ChildModel(asyncio.Event())
    extension = AsyncSubAgentExtension([definition(child)])
    parent = ParentModel(start_call(), ToolCall("c2", CANCEL, {"task_id": "task-1"}), "cancelled it")
    agent = await make_agent(parent, extension)

    await agent.run("go")

    [task] = extension.tasks("parent")
    assert task.status is SubAgentTaskStatus.CANCELLED and task.finished_at is not None
    assert '"status":"cancelled"' in last_tool_result(parent).replace(" ", "")
    await extension.cancel(task)  # no effect on a finished task
    assert task.status is SubAgentTaskStatus.CANCELLED


async def test_concurrency_is_limited_per_session():
    child = ChildModel(asyncio.Event())
    extension = AsyncSubAgentExtension([definition(child)], max_concurrent=1)
    parent = ParentModel(start_call(call_id="c1"), start_call("again", call_id="c2"), "end")
    agent = await make_agent(parent, extension)

    await agent.run("go")

    second = next(m for m in parent.requests[-1].messages if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert not second.success and "limit 1" in second.content
    await extension.close()


async def test_a_session_cannot_see_another_sessions_tasks():
    extension = AsyncSubAgentExtension([definition(ChildModel(asyncio.Event()))])
    agent = await make_agent(ParentModel(start_call(), "ok"), extension, "owner")
    await agent.run("go")

    assert len(extension.tasks("owner")) == 1 and extension.tasks("intruder") == ()
    with pytest.raises(ValueError, match="Unknown task 'task-1'"):
        extension.get_task("intruder", "task-1")
    await extension.close()


async def test_list_and_status_report_every_task_without_reports():
    gate = asyncio.Event()
    child = ChildModel(gate)
    extension = AsyncSubAgentExtension([definition(child)])
    parent = ParentModel(
        start_call(call_id="c1"),
        start_call("two", call_id="c2"),
        ToolCall("c3", LIST, {}),
        ToolCall("c4", STATUS, {"task_id": "task-2"}),
        "end",
    )
    agent = await make_agent(parent, extension)

    await agent.run("go")

    messages = {
        m.tool_call_id: json.loads(m.content) for m in parent.requests[-1].messages if isinstance(m, ToolMessage)
    }
    assert [item["task_id"] for item in messages["c3"]] == ["task-1", "task-2"]
    assert all(item["status"] == "running" and item["content"] is None for item in messages["c3"])
    assert messages["c4"]["task_id"] == "task-2" and messages["c4"]["description"] == "dig in"
    gate.set()
    await extension.close()


@tool(guidelines="Use to do nothing.")
async def ping() -> str:
    """Do nothing."""
    return "pong"


async def test_the_parent_is_told_before_its_next_model_call_when_a_task_ends_mid_request():
    gate = asyncio.Event()
    child = ChildModel(gate)
    extension = AsyncSubAgentExtension([definition(child)])

    async def finish_the_task_then_ping():
        gate.set()
        await extension.wait(extension.tasks("parent")[0], timeout=5)
        return ToolCall("p1", "ping", {})

    parent = ParentModel(start_call(), finish_the_task_then_ping, "final")
    agent = await make_agent(parent, extension, tools=[ping])

    assert (await agent.run("go")).content == "final"

    before = parent.requests[1].messages
    after = parent.requests[2].messages
    assert not any(isinstance(m, AgentMessage) for m in before)
    notices = [m for m in after if isinstance(m, AgentMessage)]
    assert len(notices) == 1
    assert notices[0].content.startswith("Background task task-1 (worker: dig in) finished.")
    assert "task_output" in notices[0].content


async def test_a_report_the_model_already_read_is_not_announced_again():
    extension = AsyncSubAgentExtension([definition(ChildModel())])
    parent = ParentModel(
        start_call(),
        ToolCall("c2", OUTPUT, {"task_id": "task-1", "wait_seconds": 5}),
        "read it",
        "later",
    )
    agent = await make_agent(parent, extension)

    await agent.run("go")
    await agent.run("anything new?")

    assert not any(isinstance(m, AgentMessage) for request in parent.requests for m in request.messages)


async def test_a_task_that_ends_between_requests_is_announced_by_the_next_one():
    gate = asyncio.Event()
    child = ChildModel(gate)
    extension = AsyncSubAgentExtension([definition(child)])
    parent = ParentModel(start_call(), "started", "welcome back")
    agent = await make_agent(parent, extension)
    await agent.run("go")
    [task] = extension.tasks("parent")

    gate.set()
    await extension.wait(task, timeout=5)
    assert not task.notified

    await agent.run("what happened?")

    messages = parent.requests[-1].messages
    notice = next(m for m in messages if isinstance(m, AgentMessage))
    assert "task-1" in notice.content and "finished" in notice.content
    assert messages.index(notice) < max(i for i, m in enumerate(messages) if isinstance(m, UserMessage))
    assert task.notified


async def test_notifications_can_be_turned_off():
    child = ChildModel()
    extension = AsyncSubAgentExtension([definition(child)], notify_parent=False)
    parent = ParentModel(start_call(), ToolCall("c2", OUTPUT, {"task_id": "task-1", "wait_seconds": 5}), "end")
    agent = await make_agent(parent, extension)

    await agent.run("go")
    await agent.run("again")

    assert not any(isinstance(m, AgentMessage) for request in parent.requests for m in request.messages)


async def test_finished_tasks_beyond_the_retention_limit_are_forgotten_oldest_first():
    extension = AsyncSubAgentExtension([definition(ChildModel())], max_retained=1)
    parent = ParentModel(
        start_call(call_id="c1"),
        ToolCall("w1", OUTPUT, {"task_id": "task-1", "wait_seconds": 5}),
        start_call("two", call_id="c2"),
        ToolCall("w2", OUTPUT, {"task_id": "task-2", "wait_seconds": 5}),
        "end",
    )
    agent = await make_agent(parent, extension)

    await agent.run("go")

    assert [task.task_id for task in extension.tasks("parent")] == ["task-2"]


async def test_unknown_subagent_types_are_rejected_with_the_available_names():
    extension = AsyncSubAgentExtension([definition(ChildModel())])
    parent = ParentModel(start_call(subagent="ghost"), "end")
    agent = await make_agent(parent, extension)

    await agent.run("go")

    result = next(m for m in parent.requests[-1].messages if isinstance(m, ToolMessage))
    assert not result.success
    assert extension.tasks("parent") == ()


async def test_the_start_schema_lists_the_configured_subagents_and_guidelines():
    extension = AsyncSubAgentExtension([definition(ChildModel(), "worker"), definition(ChildModel(), "scout")])
    parent = ParentModel("hi")
    agent = await make_agent(parent, extension)

    await agent.run("go")

    tools = {t.name: t for t in parent.requests[0].tools}
    assert set(tools) == {START, LIST, STATUS, OUTPUT, CANCEL}
    assert tools[START].parameters["properties"]["subagent_type"]["enum"] == ["worker", "scout"]


async def test_it_coexists_with_the_foreground_extension():
    parent = ParentModel("hi")
    agent = await Agent.create(
        parent,
        config=AgentRunConfig("both"),
        extensions=[SubAgentExtension([definition(ChildModel())]), AsyncSubAgentExtension([definition(ChildModel())])],
    )

    await agent.run("go")

    assert {t.name for t in parent.requests[0].tools} == {"task", START, LIST, STATUS, OUTPUT, CANCEL}


async def test_close_cancels_everything_still_running():
    child = ChildModel(asyncio.Event())
    extension = AsyncSubAgentExtension([definition(child)])
    agent = await make_agent(ParentModel(start_call(), "ok"), extension)
    await agent.run("go")

    await extension.close()

    assert extension.tasks("parent")[0].status is SubAgentTaskStatus.CANCELLED


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_limits_are_validated(value):
    with pytest.raises(ValueError, match="max_concurrent"):
        AsyncSubAgentExtension([definition(ChildModel())], max_concurrent=value)


def test_retention_must_not_be_negative():
    with pytest.raises(ValueError, match="max_retained"):
        AsyncSubAgentExtension([definition(ChildModel())], max_retained=-1)


# ---------------------------------------------------------------- subclassing


class Customized(AsyncSubAgentExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.child_systems: list[str] = []

    def build_tools(self, context, subagents):
        return tuple(t for t in super().build_tools(context, subagents) if t.name != CANCEL)

    def task_info(self, task, *, include_output=False):
        info = super().task_info(task, include_output=include_output)
        return info.model_copy(update={"note": f"custom note for {task.task_id}"})

    def completion_notice(self, task):
        return None

    async def task_started(self, task):
        self.events.append(f"started {task.task_id}")

    async def task_finished(self, task):
        self.events.append(f"finished {task.task_id} {task.status.value}")
        raise RuntimeError("a broken observer must not break the task")

    async def _create_child(self, definition, child_session_id, context):
        self.child_systems.append(definition.system_prompt)
        return await super()._create_child(definition, child_session_id, context)


async def test_a_subclass_can_change_tools_info_notices_hooks_and_child_creation():
    extension = Customized([definition(ChildModel())])
    parent = ParentModel(start_call(), ToolCall("c2", OUTPUT, {"task_id": "task-1", "wait_seconds": 5}), "end")
    agent = await make_agent(parent, extension)

    await agent.run("go")

    assert CANCEL not in {t.name for t in parent.requests[0].tools}
    assert "custom note for task-1" in last_tool_result(parent)
    assert extension.events == ["started task-1", "finished task-1 completed"]
    assert extension.child_systems == ["You are a worker."]
    assert extension.tasks("parent")[0].status is SubAgentTaskStatus.COMPLETED
    assert not any(isinstance(m, AgentMessage) for m in parent.requests[-1].messages)


async def test_task_info_and_enums_are_public_values():
    task = SubAgentTask("task-9", "p", "child", "worker", "d", "prompt")
    info = AsyncSubAgentExtension([definition(ChildModel())]).task_info(task)

    assert isinstance(info, SubAgentTaskInfo)
    assert info.status is SubAgentTaskStatus.RUNNING and "task_output" in (info.note or "")
    assert task.elapsed >= 0 and not task.finished
    assert SystemMessage  # imported for parity with the foreground tests
