"""CUSTOM observability events for JIT management and dynamic tool calls."""

from __future__ import annotations

from textwrap import dedent

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.events import AgentEventType
from zett_agent.jit_harness import (
    JIT_TOOL_COMPLETED_EVENT_NAME,
    JIT_TOOL_STARTED_EVENT_NAME,
    JITHarness,
    JITHarnessExtension,
    JITToolCompletedEvent,
    JITToolStartedEvent,
)
from zett_agent.jit_harness.enums import ToolOrigin
from zett_agent.messages import AssistantMessage, ToolCall
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

HELPER_SOURCE = dedent('''\
    @tool(guidelines="Double one integer.")
    async def helper(value: int) -> int:
        """Double the value."""
        return value * 2
''')
SOURCE = dedent('''\
    @tool(guidelines="Double through a composed dependency.")
    async def twice(value: int) -> int:
        """Compose the helper tool."""
        return await harness.call_tool("helper", value=value)
''')
FAILING_SOURCE = SOURCE.replace(
    'return await harness.call_tool("helper", value=value)',
    'await harness.call_tool("helper", value=value)\n    raise ValueError("handler failed")',
)


@tool(guidelines="Use for exact integer addition.")
async def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


class ScriptModel:
    """Emit one scripted batch per model call and record every request."""

    retry = RetryOptions(max_retries=0)

    def __init__(self, *batches):
        self.batches = batches
        self.requests = []

    async def stream(self, request):
        index = len(self.requests)
        self.requests.append(request)
        message = (
            AssistantMessage(tool_calls=self.batches[index])
            if index < len(self.batches)
            else AssistantMessage(content="done")
        )
        yield ModelEvent.completed(ModelResponse(message))


def observed(events):
    """Return only the JIT CUSTOM events, in emission order."""
    return [
        event
        for event in events
        if event.type is AgentEventType.CUSTOM
        and event.name in (JIT_TOOL_STARTED_EVENT_NAME, JIT_TOOL_COMPLETED_EVENT_NAME)
    ]


def summary(event):
    return (
        event.name,
        event.payload["tool_name"],
        event.payload["origin"],
        event.payload.get("success"),
    )


async def collect(batches, *, sources=((HELPER_SOURCE, "helper"), (SOURCE, "twice")), session="events"):
    """Run one scripted request against a fresh session-local harness."""
    template = JITHarness()
    for source, entrypoint in sources:
        template.create_tool(source, entrypoint=entrypoint)
    model = ScriptModel(*batches)
    jit = JITHarnessExtension(template.snapshot())
    agent = await Agent.create(model, tools=[add], extensions=[jit])
    events = [event async for event in agent.stream("go", config=AgentRunConfig(session))]
    return events, model, jit.get_harness(session)


async def test_management_and_dynamic_calls_emit_started_and_completed_events():
    events, _, _ = await collect(
        (
            (ToolCall("base", "add", {"left": 1, "right": 2}),),
            (ToolCall("control", "list_tools", {}),),
            (ToolCall("outer", "twice", {"value": 21}),),
        )
    )
    assert [summary(event) for event in observed(events)] == [
        (JIT_TOOL_STARTED_EVENT_NAME, "list_tools", "management", None),
        (JIT_TOOL_COMPLETED_EVENT_NAME, "list_tools", "management", True),
        (JIT_TOOL_STARTED_EVENT_NAME, "twice", "python", None),
        (JIT_TOOL_STARTED_EVENT_NAME, "helper", "python", None),
        (JIT_TOOL_COMPLETED_EVENT_NAME, "helper", "python", True),
        (JIT_TOOL_COMPLETED_EVENT_NAME, "twice", "python", True),
    ]


async def test_base_tools_and_unknown_names_emit_no_jit_events():
    events, _, harness = await collect(
        (
            (ToolCall("base", "add", {"left": 1, "right": 2}),),
            (ToolCall("ghost", "absent_tool", {}),),
        )
    )
    assert observed(events) == []
    assert harness.origin("absent_tool") is ToolOrigin.UNKNOWN
    assert harness.origin("add") is ToolOrigin.BASE
    assert harness.origin("helper") is ToolOrigin.PYTHON
    assert harness.origin("list_tools") is ToolOrigin.MANAGEMENT


async def test_event_payload_identifies_the_call_and_joins_to_profiling():
    events, _, harness = await collect(((ToolCall("outer", "twice", {"value": 21}),),))
    custom = observed(events)
    outer = [event for event in custom if event.payload["tool_name"] == "twice"]
    assert [event.name for event in outer] == [JIT_TOOL_STARTED_EVENT_NAME, JIT_TOOL_COMPLETED_EVENT_NAME]
    assert {event.payload["tool_call_id"] for event in outer} == {"outer"}
    assert all(event.payload["session_id"] == "events" for event in custom)
    assert all(event.tool_calls[0].name == event.payload["tool_name"] for event in custom)
    # The event ID is the key the profile uses, so sizes and timing join cleanly.
    records = {call.call_id: call for call in harness.profile().calls}
    assert records["outer"].tool_name == "twice" and records["outer"].status == "completed"
    nested = next(call for call in harness.profile().calls if call.caller_tool_name == "twice")
    assert nested.tool_name == "helper"
    assert nested.call_id in {event.payload["tool_call_id"] for event in custom}


async def test_failed_dynamic_call_reports_failure_without_repeating_the_message():
    events, _, harness = await collect(
        ((ToolCall("outer", "twice", {"value": 21}),),),
        sources=((HELPER_SOURCE, "helper"), (FAILING_SOURCE, "twice")),
    )
    completed = [event for event in observed(events) if event.name == JIT_TOOL_COMPLETED_EVENT_NAME]
    assert [summary(event) for event in completed] == [
        (JIT_TOOL_COMPLETED_EVENT_NAME, "helper", "python", True),
        (JIT_TOOL_COMPLETED_EVENT_NAME, "twice", "python", False),
    ]
    assert completed[-1].payload["error_type"] == "ValueError"
    assert "handler failed" not in repr(completed[-1].payload)
    assert harness.tool_errors("twice").errors[0].message == "handler failed"


async def test_payload_keys_are_stable_and_precede_runtime_tool_events():
    events, _, _ = await collect(((ToolCall("control", "list_tools", {}),),))
    custom = observed(events)
    assert [event.name for event in custom] == [JIT_TOOL_STARTED_EVENT_NAME, JIT_TOOL_COMPLETED_EVENT_NAME]
    assert set(custom[0].payload) == {"session_id", "tool_call_id", "tool_name", "origin"}
    assert set(custom[1].payload) == {"session_id", "tool_call_id", "tool_name", "origin", "success", "error_type"}
    runtime_types = [
        event.type for event in events if event.type in (AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED)
    ]
    assert runtime_types == [AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED]


def test_event_classes_build_named_custom_events():
    call = ToolCall("call", "helper", {"value": 1})
    started = JITToolStartedEvent("session", call, ToolOrigin.PYTHON)
    completed = JITToolCompletedEvent("session", call, ToolOrigin.PYTHON, success=True, error=None)
    assert started.type is AgentEventType.CUSTOM and started.name == JIT_TOOL_STARTED_EVENT_NAME
    assert completed.type is AgentEventType.CUSTOM and completed.name == JIT_TOOL_COMPLETED_EVENT_NAME
    assert completed.payload["success"] is True and completed.payload["error_type"] is None


async def test_rejected_nested_dispatch_appears_in_profiling_and_errors_not_as_its_own_event():
    template = JITHarness()
    template.create_tool(HELPER_SOURCE, entrypoint="helper")
    template.create_tool(SOURCE, entrypoint="twice")
    template.unload_tool("helper")  # The dependency cannot be dispatched at all.
    jit = JITHarnessExtension(template.snapshot())
    model = ScriptModel((ToolCall("outer", "twice", {"value": 21}),))
    agent = await Agent.create(model, tools=[add], extensions=[jit])
    events = [event async for event in agent.stream("go", config=AgentRunConfig("rejected"))]
    assert [summary(event) for event in observed(events)] == [
        (JIT_TOOL_STARTED_EVENT_NAME, "twice", "python", None),
        (JIT_TOOL_COMPLETED_EVENT_NAME, "twice", "python", False),
    ]
    harness = jit.get_harness("rejected")
    rejected = [call for call in harness.profile().calls if call.status == "rejected"]
    assert [call.tool_name for call in rejected] == ["helper"]
    assert harness.tool_errors("helper").errors[0].message == "Tool 'helper' is unloaded"
