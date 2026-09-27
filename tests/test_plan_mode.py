"""Model-proposed Plan Mode confirmation, activation, and tool boundaries."""

import asyncio
import json

import pytest

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.events import AgentEventType
from zett_agent.exceptions import (
    AgentIterationLimitError,
    AgentProtocolError,
)
from zett_agent.extensions.coding import CodingExtension
from zett_agent.extensions.external import ExternalEvent
from zett_agent.extensions.plan_mode import (
    ENTER_PLAN_MODE_EVENT_NAME,
    ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
    ENTER_PLAN_MODE_TOOL_NAME,
    EXIT_PLAN_MODE_EVENT_NAME,
    EXIT_PLAN_MODE_RESPONSE_EVENT_NAME,
    EXIT_PLAN_MODE_TOOL_NAME,
    PLAN_MODE_ENTERED_EVENT_NAME,
    PLAN_MODE_EXITED_EVENT_NAME,
    PLAN_MODE_SYSTEM_PROMPT,
    PlanModeExtension,
)
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)


class ScriptedModel:
    def __init__(self, *responses: AssistantMessage) -> None:
        self.responses = list(responses)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(self.responses.pop(0)))


async def test_enter_completion_is_emitted_even_without_another_model_iteration():
    plan_mode = PlanModeExtension()
    agent = await Agent.create(
        ScriptedModel(proposal()),
        config=AgentRunConfig("last-step"),
        extensions=[plan_mode],
        max_iterations=1,
    )
    events = []
    with pytest.raises(AgentIterationLimitError):
        async for event in agent.stream("Plan this"):
            events.append(event)
            if event.name == ENTER_PLAN_MODE_EVENT_NAME:
                assert agent.emit_external_event(
                    ExternalEvent(
                        ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
                        {
                            "session_id": "last-step",
                            "tool_call_id": "enter-1",
                            "approved": True,
                        },
                    ),
                    config=AgentRunConfig("last-step"),
                ) == ["PlanModeExtension"]
    assert any(event.type is AgentEventType.TOOL_COMPLETED for event in events), [
        (event.type, event.error) for event in events
    ]
    entered = next(i for i, event in enumerate(events) if event.name == PLAN_MODE_ENTERED_EVENT_NAME)
    assert events[entered + 1].type is AgentEventType.TOOL_COMPLETED
    assert sum(event.type is AgentEventType.MODEL_STARTED for event in events) == 1
    assert plan_mode._entered_events == set()


def proposal(call_id: str = "enter-1") -> AssistantMessage:
    return AssistantMessage(
        tool_calls=(
            ToolCall(
                call_id,
                ENTER_PLAN_MODE_TOOL_NAME,
                {"reason": "The change crosses storage, API, and UI boundaries."},
            ),
        )
    )


def exit_proposal(call_id: str = "exit-1") -> AssistantMessage:
    return AssistantMessage(
        tool_calls=(
            ToolCall(
                call_id,
                EXIT_PLAN_MODE_TOOL_NAME,
                {"plan": "# Plan\n\n1. Update storage.\n2. Add API tests."},
            ),
        )
    )


async def run_with_decision(agent: Agent, *, approved: bool):
    events = []
    confirmation_ready = asyncio.Event()

    async def consume():
        async for event in agent.stream("Handle this complex change"):
            events.append(event)
            if event.name == ENTER_PLAN_MODE_EVENT_NAME:
                confirmation_ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(confirmation_ready.wait(), timeout=1)
    confirmation = next(event for event in events if event.name == ENTER_PLAN_MODE_EVENT_NAME)
    accepted = agent.emit_external_event(
        ExternalEvent(
            name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
            payload={
                "session_id": confirmation.session_id,
                "tool_call_id": confirmation.tool_calls[0].id,
                "approved": approved,
            },
        )
    )
    assert accepted == ["PlanModeExtension"]
    await asyncio.wait_for(task, timeout=1)
    return events


async def run_with_exit_decision(agent: Agent, *, approved: bool):
    events = []
    confirmation_ready = asyncio.Event()

    async def consume():
        async for event in agent.stream("Submit the completed plan"):
            events.append(event)
            if event.name == EXIT_PLAN_MODE_EVENT_NAME:
                confirmation_ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(confirmation_ready.wait(), timeout=1)
    confirmation = next(event for event in events if event.name == EXIT_PLAN_MODE_EVENT_NAME)
    assert agent.emit_external_event(
        ExternalEvent(
            name=EXIT_PLAN_MODE_RESPONSE_EVENT_NAME,
            payload={
                "session_id": confirmation.session_id,
                "tool_call_id": confirmation.tool_calls[0].id,
                "approved": approved,
            },
        )
    )
    await asyncio.wait_for(task, timeout=1)
    return events


async def test_model_proposal_waits_for_approval_then_enters_plan_mode():
    model = ScriptedModel(proposal(), AssistantMessage(content="Detailed plan"))
    plan_mode = PlanModeExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("planning"),
        system_prompt="Normal implementation prompt",
        extensions=[CodingExtension(), ToolGuidelinesExtension(), plan_mode],
    )

    events = await run_with_decision(agent, approved=True)

    assert plan_mode.is_plan_mode("planning") is True
    confirmation = next(event for event in events if event.name == ENTER_PLAN_MODE_EVENT_NAME)
    assert confirmation.payload == {
        "session_id": "planning",
        "tool_call_id": "enter-1",
        "question": "Would you like to enter Plan Mode?",
        "options": ["Enter Plan Mode", "Continue without Plan Mode"],
        "reason": "The change crosses storage, API, and UI boundaries.",
        "response_event": ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
    }
    event_types = [(event.type, event.name) for event in events]
    assert event_types.index((AgentEventType.CUSTOM, ENTER_PLAN_MODE_EVENT_NAME)) < event_types.index(
        (AgentEventType.TOOL_STARTED, None)
    )
    assert event_types.index((AgentEventType.CUSTOM, PLAN_MODE_ENTERED_EVENT_NAME)) < event_types.index(
        (AgentEventType.TOOL_COMPLETED, None)
    )
    assert [event.payload for event in events if event.name == PLAN_MODE_ENTERED_EVENT_NAME] == [
        {"session_id": "planning", "active": True}
    ]

    initial_tools = {tool.name for tool in model.requests[0].tools}
    assert ENTER_PLAN_MODE_TOOL_NAME in initial_tools
    assert "run_shell" in initial_tools
    plan_tools = {tool.name for tool in model.requests[1].tools}
    assert plan_tools == {
        "read_file",
        "view_image",
        "write_file",
        "replace_in_file",
        "delete_file",
        "glob",
        "grep",
        "run_shell",
        EXIT_PLAN_MODE_TOOL_NAME,
    }
    instructions = [message.content for message in model.requests[1].messages if isinstance(message, SystemMessage)]
    assert len(instructions) == 1
    assert instructions[0].startswith(PLAN_MODE_SYSTEM_PROMPT)
    assert "Normal implementation prompt" not in instructions[0]
    assert "## write_file" in instructions[0]
    assert "## run_shell" in instructions[0]
    assert f"## {EXIT_PLAN_MODE_TOOL_NAME}" in instructions[0]

    decision = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert json.loads(decision.content) == {"entered": True, "message": "Plan Mode is active"}


async def test_declined_proposal_keeps_normal_mode_prompt_and_tools():
    model = ScriptedModel(proposal(), AssistantMessage(content="Continue normally"))
    plan_mode = PlanModeExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("declined"),
        system_prompt="Normal implementation prompt",
        extensions=[CodingExtension(), ToolGuidelinesExtension(), plan_mode],
    )

    events = await run_with_decision(agent, approved=False)

    assert plan_mode.is_plan_mode("declined") is False
    assert all(event.name != PLAN_MODE_ENTERED_EVENT_NAME for event in events)
    assert ENTER_PLAN_MODE_TOOL_NAME in {tool.name for tool in model.requests[1].tools}
    instructions = [message.content for message in model.requests[1].messages if isinstance(message, SystemMessage)]
    assert instructions[0] == "Normal implementation prompt"
    decision = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert json.loads(decision.content) == {"entered": False, "message": "The user declined Plan Mode"}


def test_external_response_cannot_proactively_enter_plan_mode():
    plan_mode = PlanModeExtension()
    agent = Agent(ScriptedModel(), extensions=[plan_mode])

    accepted = agent.emit_external_event(
        ExternalEvent(
            name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
            payload={"session_id": "session", "tool_call_id": "missing", "approved": True},
        )
    )

    assert accepted == []
    assert plan_mode.is_plan_mode("session") is False


async def test_malformed_external_decision_becomes_failed_tool_result():
    model = ScriptedModel(proposal(), AssistantMessage(content="Handled failure"))
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("malformed"), extensions=[plan_mode])
    events = []
    ready = asyncio.Event()

    async def consume():
        async for event in agent.stream("Plan"):
            events.append(event)
            if event.name == ENTER_PLAN_MODE_EVENT_NAME:
                ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), timeout=1)
    assert agent.emit_external_event(
        ExternalEvent(
            name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
            payload={"session_id": "malformed", "tool_call_id": "enter-1", "approved": "yes"},
        )
    )
    await asyncio.wait_for(task, timeout=1)

    assert plan_mode.is_plan_mode("malformed") is False
    failed = next(event for event in events if event.type == AgentEventType.TOOL_FAILED)
    assert isinstance(failed.message, ToolMessage)
    assert failed.message.success is False
    assert "valid boolean" in failed.message.content


async def test_invalid_tool_arguments_do_not_wait_for_external_input():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("invalid", ENTER_PLAN_MODE_TOOL_NAME, {}),)),
        AssistantMessage(content="Recovered"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("invalid"), extensions=[plan_mode])

    events = [event async for event in agent.stream("Plan")]

    assert all(event.name != ENTER_PLAN_MODE_EVENT_NAME for event in events)
    assert next(event for event in events if event.type == AgentEventType.TOOL_FAILED).message.success is False
    assert plan_mode.is_plan_mode("invalid") is False


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"reason": ""},
        {"reason": "   \n\t"},
        {"reason": "Valid reason", "unknown": True},
    ],
)
async def test_empty_or_invalid_enter_request_never_opens_confirmation(arguments):
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("invalid-enter", ENTER_PLAN_MODE_TOOL_NAME, arguments),)),
        AssistantMessage(content="Continue without planning"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("empty-enter"), extensions=[plan_mode])

    events = [event async for event in agent.stream("Handle the request")]

    assert all(event.name != ENTER_PLAN_MODE_EVENT_NAME for event in events)
    failed = [event for event in events if event.type == AgentEventType.TOOL_FAILED]
    assert len(failed) == 1
    assert failed[0].message.success is False
    assert plan_mode.is_plan_mode("empty-enter") is False


async def test_cancelling_confirmation_wait_rejects_late_response():
    model = ScriptedModel(proposal())
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("cancelled"), extensions=[plan_mode])
    stream = agent.stream("Plan")
    confirmation = None
    async for event in stream:
        if event.name == ENTER_PLAN_MODE_EVENT_NAME:
            confirmation = event
            break
    await stream.aclose()

    assert confirmation is not None
    assert (
        agent.emit_external_event(
            ExternalEvent(
                name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
                payload={"session_id": "cancelled", "tool_call_id": "enter-1", "approved": True},
            )
        )
        == []
    )
    assert plan_mode.is_plan_mode("cancelled") is False


async def test_active_plan_mode_rejects_tools_outside_its_capability_set():
    model = ScriptedModel(
        proposal(),
        AssistantMessage(content="Plan ready"),
        AssistantMessage(tool_calls=(ToolCall("task-1", "task", {}),)),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("restricted"), extensions=[plan_mode])
    await run_with_decision(agent, approved=True)

    with pytest.raises(AgentProtocolError, match="not allowed in Plan Mode"):
        await agent.run("Continue planning")


async def test_active_plan_mode_executes_filesystem_and_shell_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel(
        proposal(),
        AssistantMessage(content="Plan ready"),
        AssistantMessage(
            tool_calls=(
                ToolCall("write", "write_file", {"path": "plan.md", "content": "# Plan"}),
                ToolCall("read", "read_file", {"path": "plan.md"}),
                ToolCall("shell", "run_shell", {"command": "printf plan-mode"}),
            )
        ),
        AssistantMessage(content="Updated plan"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("plan-tools"),
        extensions=[plan_mode],
        parallel_tool_call=False,
    )
    await run_with_decision(agent, approved=True)

    await agent.run("Maintain the planning artifact")

    results = {
        message.name: message.content for message in model.requests[3].messages if isinstance(message, ToolMessage)
    }
    assert (tmp_path / "plan.md").read_text() == "# Plan"
    assert results["read_file"] == "# Plan"
    assert results["run_shell"] == "plan-mode"
    assert all(message.success for message in model.requests[3].messages if isinstance(message, ToolMessage))


async def test_model_submits_plan_and_approved_exit_restores_normal_mode():
    model = ScriptedModel(
        proposal(),
        AssistantMessage(content="Planning started"),
        exit_proposal(),
        AssistantMessage(content="Ready to implement"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("exit-approved"),
        system_prompt="Normal implementation prompt",
        extensions=[CodingExtension(), ToolGuidelinesExtension(), plan_mode],
    )
    await run_with_decision(agent, approved=True)

    events = await run_with_exit_decision(agent, approved=True)

    assert plan_mode.is_plan_mode("exit-approved") is False
    confirmation = next(event for event in events if event.name == EXIT_PLAN_MODE_EVENT_NAME)
    assert confirmation.payload == {
        "session_id": "exit-approved",
        "tool_call_id": "exit-1",
        "question": "Is this plan ready to leave Plan Mode?",
        "options": ["Approve and Exit", "Keep Planning"],
        "plan": "# Plan\n\n1. Update storage.\n2. Add API tests.",
        "response_event": EXIT_PLAN_MODE_RESPONSE_EVENT_NAME,
    }
    assert [event.payload for event in events if event.name == PLAN_MODE_EXITED_EVENT_NAME] == [
        {"session_id": "exit-approved", "active": False}
    ]
    event_types = [(event.type, event.name) for event in events]
    assert event_types.index((AgentEventType.CUSTOM, EXIT_PLAN_MODE_EVENT_NAME)) < event_types.index(
        (AgentEventType.TOOL_STARTED, None)
    )
    assert event_types.index((AgentEventType.CUSTOM, PLAN_MODE_EXITED_EVENT_NAME)) < event_types.index(
        (AgentEventType.TOOL_COMPLETED, None)
    )

    restored_tools = {tool.name for tool in model.requests[3].tools}
    assert ENTER_PLAN_MODE_TOOL_NAME in restored_tools
    assert EXIT_PLAN_MODE_TOOL_NAME not in restored_tools
    assert "run_shell" in restored_tools
    instructions = [message.content for message in model.requests[3].messages if isinstance(message, SystemMessage)]
    assert instructions[0] == "Normal implementation prompt"
    assert all(PLAN_MODE_SYSTEM_PROMPT not in instruction for instruction in instructions)
    result = next(message for message in model.requests[3].messages if isinstance(message, ToolMessage))
    assert json.loads(result.content) == {"exited": True, "message": "Plan Mode is inactive"}


async def test_declined_exit_keeps_plan_mode_active():
    model = ScriptedModel(
        proposal(),
        AssistantMessage(content="Planning started"),
        exit_proposal(),
        AssistantMessage(content="I will refine the plan"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("exit-declined"), extensions=[plan_mode])
    await run_with_decision(agent, approved=True)

    events = await run_with_exit_decision(agent, approved=False)

    assert plan_mode.is_plan_mode("exit-declined") is True
    assert all(event.name != PLAN_MODE_EXITED_EVENT_NAME for event in events)
    assert EXIT_PLAN_MODE_TOOL_NAME in {tool.name for tool in model.requests[3].tools}
    instructions = [message.content for message in model.requests[3].messages if isinstance(message, SystemMessage)]
    assert instructions[0].startswith(PLAN_MODE_SYSTEM_PROMPT)
    result = next(message for message in model.requests[3].messages if isinstance(message, ToolMessage))
    assert json.loads(result.content) == {"exited": False, "message": "The user requested more planning"}


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"plan": ""},
        {"plan": "   \n\t"},
        {"plan": "# Valid plan", "unknown": True},
    ],
)
async def test_empty_or_invalid_exit_request_keeps_plan_mode_active(arguments):
    model = ScriptedModel(
        proposal(),
        AssistantMessage(content="Planning started"),
        AssistantMessage(tool_calls=(ToolCall("invalid-exit", EXIT_PLAN_MODE_TOOL_NAME, arguments),)),
        AssistantMessage(content="Continue planning"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("empty-exit"), extensions=[plan_mode])
    await run_with_decision(agent, approved=True)

    events = [event async for event in agent.stream("Submit the plan")]

    assert all(event.name != EXIT_PLAN_MODE_EVENT_NAME for event in events)
    assert all(event.name != PLAN_MODE_EXITED_EVENT_NAME for event in events)
    failed = [event for event in events if event.type == AgentEventType.TOOL_FAILED]
    assert len(failed) == 1
    assert failed[0].message.success is False
    assert plan_mode.is_plan_mode("empty-exit") is True
    assert EXIT_PLAN_MODE_TOOL_NAME in {tool.name for tool in model.requests[3].tools}


async def test_full_lifecycle_restores_empty_prompt_and_empty_original_tool_set():
    model = ScriptedModel(
        proposal(),
        AssistantMessage(content="Plan Mode is ready"),
        exit_proposal(),
        AssistantMessage(content="Normal mode is ready"),
    )
    plan_mode = PlanModeExtension()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("empty-baseline"),
        system_prompt="",
        tools=[],
        extensions=[plan_mode],
    )

    await run_with_decision(agent, approved=True)
    await run_with_exit_decision(agent, approved=True)

    assert plan_mode.is_plan_mode("empty-baseline") is False
    plan_request = model.requests[1]
    assert {tool.name for tool in plan_request.tools} == {
        "read_file",
        "view_image",
        "write_file",
        "replace_in_file",
        "delete_file",
        "glob",
        "grep",
        "run_shell",
        EXIT_PLAN_MODE_TOOL_NAME,
    }
    restored_request = model.requests[3]
    assert {tool.name for tool in restored_request.tools} == {ENTER_PLAN_MODE_TOOL_NAME}
    assert not any(isinstance(message, SystemMessage) for message in restored_request.messages)


async def test_wrong_response_event_cannot_wake_an_enter_confirmation():
    model = ScriptedModel(proposal(), AssistantMessage(content="Continue"))
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("wrong-event"), extensions=[plan_mode])
    ready = asyncio.Event()

    async def consume():
        async for event in agent.stream("Plan"):
            if event.name == ENTER_PLAN_MODE_EVENT_NAME:
                ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), timeout=1)
    assert (
        agent.emit_external_event(
            ExternalEvent(
                name=EXIT_PLAN_MODE_RESPONSE_EVENT_NAME,
                payload={"session_id": "wrong-event", "tool_call_id": "enter-1", "approved": True},
            )
        )
        == []
    )
    assert agent.emit_external_event(
        ExternalEvent(
            name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
            payload={"session_id": "wrong-event", "tool_call_id": "enter-1", "approved": False},
        )
    )
    await asyncio.wait_for(task, timeout=1)
    assert plan_mode.is_plan_mode("wrong-event") is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"tool_call_id": None, "approved": True},
        {"session_id": "empty-response", "tool_call_id": "", "approved": True},
    ],
)
async def test_empty_external_response_does_not_wake_pending_confirmation(payload):
    model = ScriptedModel(proposal(), AssistantMessage(content="Continue"))
    plan_mode = PlanModeExtension()
    agent = await Agent.create(model, config=AgentRunConfig("empty-response"), extensions=[plan_mode])
    ready = asyncio.Event()

    async def consume():
        async for event in agent.stream("Plan"):
            if event.name == ENTER_PLAN_MODE_EVENT_NAME:
                ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), timeout=1)
    assert agent.emit_external_event(ExternalEvent(name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME, payload=payload)) == []
    assert task.done() is False
    assert agent.emit_external_event(
        ExternalEvent(
            name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
            payload={"session_id": "empty-response", "tool_call_id": "enter-1", "approved": False},
        )
    )
    await asyncio.wait_for(task, timeout=1)
    assert plan_mode.is_plan_mode("empty-response") is False


async def test_exit_tool_is_illegal_outside_plan_mode():
    model = ScriptedModel(exit_proposal())
    agent = await Agent.create(
        model,
        config=AgentRunConfig("illegal-exit"),
        extensions=[PlanModeExtension()],
    )

    with pytest.raises(AgentProtocolError, match="while it is not active"):
        await agent.run("Exit")


def test_plan_mode_validates_session_ids_and_custom_prompt():
    extension = PlanModeExtension()
    with pytest.raises(ValueError, match="session_id"):
        extension.is_plan_mode(" ")
    with pytest.raises(ValueError, match="system prompt"):
        PlanModeExtension(system_prompt=" ")
