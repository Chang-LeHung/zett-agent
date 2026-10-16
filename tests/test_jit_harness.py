"""Trusted Python tools: composition, mutation, portable state, and isolation."""

from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import replace
from textwrap import dedent
from threading import Event
from types import ModuleType

import httpx
import pytest
from pydantic import ValidationError

from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.events import AgentEventType
from zett_agent.extensions.ask_user import AskUserExtension
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.external import ExternalEvent
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.jit_harness import (
    InspectionToolExposure,
    JITHarness,
    JITHarnessExtension,
    JITHarnessSnapshot,
    PythonToolSource,
)
from zett_agent.jit_harness.extension import _INSPECTION_TOOL_NAMES
from zett_agent.jit_harness.snapshot import _MANAGEMENT_NAMES
from zett_agent.messages import (
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolMessage,
)
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.providers.openai_compatible import chat_tools
from zett_agent.providers.responses import responses_tools
from zett_agent.tools.base import _SYNC_TOOL_RUNNER, ToolExecutionMode, current_tool_call_id, tool

SOURCE = dedent('''\
    @tool(guidelines="Use for composed integer addition.")
    async def twice(value: int) -> int:
        """Add the same integer twice."""
        return await harness.call_tool("add", left=value, right=value)
''')
SYNC_SOURCE = SOURCE.replace("async def", "def").replace("await harness.call_tool", "harness.call_tool_sync")
PURE_SOURCE = SOURCE.replace('await harness.call_tool("add", left=value, right=value)', "value * 2")


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def context(*tools, session="test", extensions=()):
    return AgentRunContext(AgentRunConfig(session), AgentState(), {tool.name: tool for tool in tools}, extensions)


class QuietModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request):
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))


@pytest.mark.parametrize("source", [SOURCE, SYNC_SOURCE])
async def test_create_compose_unload_reload_and_roundtrip(source, tmp_path):
    harness = JITHarness(tools=[add])
    info = harness.create_tool(source, entrypoint="twice")
    assert info == {"name": "twice", "status": "loaded"}
    assert harness.describe("twice")["origin"] == "python"
    assert await harness.call_tool("twice", value=21) == 42
    harness.unload_tool("add")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("twice", value=21)
    assert harness.inspect_tool("twice")["source"] == source
    harness.save(tmp_path / "harness.json")
    snapshot = JITHarnessSnapshot.load(tmp_path / "harness.json")
    assert snapshot == JITHarnessSnapshot.from_json(snapshot.to_json())
    reloaded = JITHarness(snapshot, tools=[add])
    with pytest.raises(ValueError, match="unloaded"):
        await reloaded.call_tool("add", left=1, right=2)
    reloaded.load_tool("add")
    assert await reloaded.call_tool("twice", value=21) == 42
    assert harness.inspect_tool("add")["status"] == "unloaded"
    assert "source" not in reloaded.inspect_tool("add")


async def test_failed_updates_preserve_old_tool_and_unloaded_updates_stay_unloaded():
    harness = JITHarness()
    harness.create_tool(PURE_SOURCE, entrypoint="twice")
    with pytest.raises(SyntaxError):
        harness.update_tool("twice", "invalid python !", entrypoint="twice")
    assert await harness.call_tool("twice", value=3) == 6
    with pytest.raises(ValueError, match="rename"):
        harness.update_tool("twice", PURE_SOURCE.replace("twice", "renamed"), entrypoint="renamed")
    harness.unload_tool("twice")
    harness.update_tool("twice", PURE_SOURCE.replace("value * 2", "value * 3"), entrypoint="twice")
    assert harness.inspect_tool("twice")["status"] == "unloaded"
    harness.load_tool("twice")
    assert await harness.call_tool("twice", value=3) == 9
    with pytest.raises(ValueError, match="already exists"):
        harness.create_tool(PURE_SOURCE, entrypoint="twice")


@pytest.mark.parametrize(
    "name",
    [
        "create_tool",
        "update_tool",
        "unload_tool",
        "load_tool",
        "list_tools",
        "inspect_tool",
        "inspect_tool_errors",
        "read_tool_source",
        "read_tool_schema",
        "inspect_tool_stats",
        "inspect_tool_calls",
    ],
)
def test_management_names_reserved(name):
    harness = JITHarness()
    with pytest.raises(ValueError, match="reserved"):
        harness.create_tool(PURE_SOURCE.replace("twice", name), entrypoint=name)
    for operation in (harness.unload_tool, harness.load_tool, harness.inspect_tool):
        with pytest.raises(ValueError, match="management"):
            operation(name)


def test_management_tool_docstrings_preserve_guidelines_and_serial_execution():
    expected = {
        "create_tool": (
            "Reuse existing capabilities; create only for missing functionality or worthwhile workflow savings.",
            "Put repetitive tool work in one tool that calls others, not in many separate calls.",
        ),
        "update_tool": ("Inspect source before updating; send only the files that changed.",),
        "unload_tool": "Unload tools no longer needed; dependent calls will fail.",
        "load_tool": "Load an unloaded dependency before calling it.",
        "list_tools": "Use to find tool names and loaded/unloaded/missing states.",
        "inspect_tool": "Inspect parameters before composition; use read_tool_source before updating source.",
        "read_tool_source": "Follow next_offset for more source; compare content_id across pages after updates.",
        "read_tool_schema": "Use only when parameters_omitted is true; concatenate pages before parsing JSON.",
        "inspect_tool_stats": "Use to compare frequency, failure counts, latency, and input/output sizes.",
        "inspect_tool_calls": "Read when caller relationships or per-call timing and sizes help; follow next_offset.",
        "inspect_tool_errors": "Inspect recent errors before deciding whether to retry, repair source, or load dependencies.",
    }
    management = JITHarnessExtension()._management_tools(JITHarness())
    assert {registered.name: registered.guidelines for registered in management} == {
        name: (guideline if isinstance(guideline, tuple) else (guideline,)) for name, guideline in expected.items()
    }
    assert all(registered.execution_mode is ToolExecutionMode.SERIAL for registered in management)


@pytest.mark.parametrize("mode", list(InspectionToolExposure))
def test_inspection_exposure_tiers_the_control_tools_without_losing_capability(mode):
    """Deferral follows the protocol; hiding is the only mode that removes tools."""
    extension = JITHarnessExtension(inspection_tools=mode)
    management = {registered.name: registered for registered in extension._management_tools(JITHarness())}
    inspection = set(_INSPECTION_TOOL_NAMES)
    assert inspection < set(_MANAGEMENT_NAMES)
    if mode is InspectionToolExposure.HIDDEN:
        assert set(management) == set(_MANAGEMENT_NAMES) - inspection
    else:
        assert set(management) == set(_MANAGEMENT_NAMES)
    deferred = {name for name, registered in management.items() if registered.deferred}
    assert deferred == (inspection if mode is InspectionToolExposure.DEFERRED else set())


@pytest.mark.parametrize("value", ["hidden", InspectionToolExposure.HIDDEN])
def test_inspection_exposure_accepts_its_value_and_rejects_anything_else(value):
    assert JITHarnessExtension(inspection_tools=value)._inspection_tools is InspectionToolExposure.HIDDEN
    with pytest.raises(ValueError, match="is not a valid InspectionToolExposure"):
        JITHarnessExtension(inspection_tools="occasionally")


async def test_hidden_inspection_tools_stay_application_side_only():
    """HIDDEN withholds definitions and measurements, not the harness API."""
    jit = JITHarnessExtension(inspection_tools=InspectionToolExposure.HIDDEN)
    ctx = context(add)
    await jit.on_tool(ctx)
    try:
        assert "inspect_tool_stats" not in ctx.tools and "read_tool_source" not in ctx.tools
        assert {"create_tool", "list_tools", "inspect_tool"} <= set(ctx.tools)
        harness = jit.get_harness("test")
        assert await ctx.tools["add"]({"left": 21, "right": 21}) == 42
        assert harness.profile_stats("add").call_count == 1
        assert harness.inspect_tool("add")["name"] == "add"
    finally:
        jit._request_finished(ctx)


def test_reserved_management_names_match_the_exposed_management_tools():
    """Every reserved name must be one the extension actually exposes, and back."""
    management = JITHarnessExtension()._management_tools(JITHarness())
    assert {registered.name for registered in management} == set(_MANAGEMENT_NAMES)


@pytest.mark.parametrize("kind", ["base", "python", "text"])
async def test_unload_tool_is_the_only_unload_entrypoint_and_preserves_serialized_state(kind):
    extension = JITHarnessExtension()
    ctx = context(add)
    await extension.on_tool(ctx)
    harness = extension.get_harness("test")
    if kind == "python":
        harness.create_tool(PURE_SOURCE, entrypoint="twice")
    elif kind == "text":
        harness.create_tool("# Add two numbers", kind="text", name="addition_workflow", description="Use for addition.")
    name = {"base": "add", "python": "twice", "text": "addition_workflow"}[kind]
    arguments = {"base": {"left": 21, "right": 21}, "python": {"value": 21}, "text": {}}[kind]
    original = ctx.tools[name]
    before = harness.snapshot()
    try:
        assert not hasattr(harness, "offload_tool")
        assert "offload_tool" not in ctx.tools
        assert ctx.tools["unload_tool"].definition.name == "unload_tool"
        expected = await original(arguments)
        if kind == "text":
            assert expected["content"] == "# Add two numbers"
        else:
            assert expected == 42

        assert await ctx.tools["unload_tool"]({"name": name}) == {"name": name, "status": "unloaded"}
        assert name not in ctx.tools
        saved = harness.snapshot()
        assert saved.tools == before.tools and saved.base_tools == before.base_tools
        assert saved.unloaded_tools == (name,)
        serialized = json.loads(saved.to_json())
        assert serialized["unloaded_tools"] == [name]
        assert "offloaded_tools" not in serialized
        assert not hasattr(saved, "offloaded_tools")
        assert not hasattr(harness, "_offloaded") and not hasattr(harness, "_is_offloaded")
        assert harness.origin("unload_tool").value == "management"
        assert harness.origin("offload_tool").value == "unknown"
        with pytest.raises(ValueError, match="unloaded"):
            await original(arguments)

        assert await ctx.tools["load_tool"]({"name": name}) == {"name": name, "status": "loaded"}
        assert harness.snapshot() == before
        assert await ctx.tools[name](arguments) == expected
        if kind == "python":
            assert harness.source(name) == PURE_SOURCE
        elif kind == "text":
            assert harness.source(name) == "# Add two numbers"
    finally:
        extension._request_finished(ctx)


async def test_management_state_changes_answer_with_a_compact_digest():
    extension = JITHarnessExtension()
    ctx = context(add)
    await extension.on_tool(ctx)
    source = PURE_SOURCE.replace("twice", "summarize") + "# padding for size\n" * 60
    digest_keys = {"name", "status"}
    try:
        created = await ctx.tools["create_tool"]({"source": source, "entrypoint": "summarize"})
        assert set(created) == digest_keys
        assert created["name"] == "summarize"
        assert created["status"] == "loaded"
        assert source not in json.dumps(created)  # The model already sent that source.
        for value in range(8):
            assert await ctx.tools["summarize"]({"value": value}) == value * 2

        updated = await ctx.tools["update_tool"](
            {"name": "summarize", "source": source + "\n# edited\n", "entrypoint": "summarize"}
        )
        assert set(updated) == digest_keys and updated["status"] == "loaded"
        unloaded = await ctx.tools["unload_tool"]({"name": "summarize"})
        assert set(unloaded) == digest_keys and unloaded["status"] == "unloaded"
        loaded = await ctx.tools["load_tool"]({"name": "summarize"})
        assert set(loaded) == digest_keys and loaded["status"] == "loaded"
        assert all(len(json.dumps(result)) < 100 for result in (created, updated, unloaded, loaded))

        # Measurements are still retained, but only queries return them.
        detail = await ctx.tools["inspect_tool"]({"name": "summarize"})
        assert "source" not in detail and "profiling" not in detail and "calls" not in detail
        stats = await ctx.tools["inspect_tool_stats"]({"name": "summarize"})
        assert stats.call_count == 8
        calls = await ctx.tools["inspect_tool_calls"]({"name": "summarize"})
        assert len(calls["calls"]) == 5
        text = await ctx.tools["read_tool_source"]({"name": "summarize"})
        assert text["content"] == source + "\n# edited\n"
        catalog = await ctx.tools["list_tools"]({})
        assert "profiling" not in next(item for item in catalog["tools"] if item["name"] == "summarize")
        assert len(json.dumps(loaded)) < len(json.dumps(detail))
    finally:
        extension._request_finished(ctx)


async def test_direct_state_changes_do_not_query_profiling_and_preserve_update_unload_state(monkeypatch):
    harness = JITHarness(tools=[add])

    def unexpected_summary(name):
        raise AssertionError(f"State changes must not read profiling for {name}")

    with monkeypatch.context() as patch:
        patch.setattr(harness._profiler, "summary", unexpected_summary)
        assert harness.create_tool(PURE_SOURCE, entrypoint="twice") == {"name": "twice", "status": "loaded"}
        assert harness.update_tool("twice", PURE_SOURCE, entrypoint="twice") == {"name": "twice", "status": "loaded"}
        assert harness.unload_tool("twice") == {"name": "twice", "status": "unloaded"}
        assert harness.update_tool("twice", PURE_SOURCE, entrypoint="twice") == {
            "name": "twice",
            "status": "unloaded",
        }
        assert harness.load_tool("twice") == {"name": "twice", "status": "loaded"}
        assert harness.unload_tool("add") == {"name": "add", "status": "unloaded"}
        assert harness.load_tool("add") == {"name": "add", "status": "loaded"}
    assert await harness.call_tool("twice", value=21) == 42
    assert harness.profile("twice").stats[0].completed_count == 1


async def test_detail_tools_default_to_a_small_page_and_page_on_request():
    extension = JITHarnessExtension()
    ctx = context(add)
    await extension.on_tool(ctx)
    extension.get_harness("test").create_tool(PURE_SOURCE, entrypoint="twice")
    try:
        for value in range(9):
            await ctx.tools["twice"]({"value": value})
        page = await ctx.tools["inspect_tool_calls"]({"name": "twice"})
        assert len(page["calls"]) == 5 and page["total_calls"] == 9
        assert page["next_offset"] == 5
        rest = await ctx.tools["inspect_tool_calls"]({"name": "twice", "offset": page["next_offset"]})
        expected = extension.get_harness("test").profile("twice").calls
        assert [call["call_id"] for call in (*page["calls"], *rest["calls"])] == [call.call_id for call in expected]
        assert rest["next_offset"] is None

        for index in range(7):
            with pytest.raises(ValidationError):
                await ctx.tools["twice"]({"value": f"bad-{index}"})
        errors = await ctx.tools["inspect_tool_errors"]({})
        assert len(errors["errors"]) == 3 and errors["total_matches"] == 7
        assert errors["next_offset"] == 3
    finally:
        extension._request_finished(ctx)


@pytest.mark.parametrize(
    ("source", "entrypoint", "error"),
    [
        (PURE_SOURCE, "not an identifier", "identifier"),
        ("twice = 1", "twice", "@tool"),
        (PURE_SOURCE.replace("@tool(", "@tool(deferred=True, "), "twice", "ordinary"),
        (PURE_SOURCE.replace("@tool(", "@tool(local_tool_search=True, "), "twice", "ordinary"),
        (PURE_SOURCE.replace("@tool(", '@tool(name="!", '), "twice", "Tool names"),
        (PURE_SOURCE.replace("value: int", "value"), "twice", "annotation"),
        (PURE_SOURCE.replace('guidelines="Use for composed integer addition."', 'guidelines=""'), "twice", "guideline"),
    ],
)
def test_source_validation_leaves_catalog_unchanged(source, entrypoint, error):
    before = {key for key in sys.modules if key.startswith("zett_jit_")}
    harness = JITHarness()
    with pytest.raises(ValueError, match=error):
        harness.create_tool(source, entrypoint=entrypoint)
    assert harness.list_tools() == []
    assert {key for key in sys.modules if key.startswith("zett_jit_")} == before


async def test_module_helpers_imports_future_annotations_and_entrypoint_not_run():
    source = dedent('''\
        from __future__ import annotations
        from dataclasses import dataclass
        import math
        @dataclass
        class Result:
            value: int
        @tool(guidelines="Return a typed dataclass.")
        def answer(value: int) -> Result:
            """Return the square root."""
            return Result(math.isqrt(value))
    ''')
    harness = JITHarness()
    harness.create_tool(source, entrypoint="answer")
    assert (await harness.call_tool("answer", value=81)).value == 9
    source = PURE_SOURCE.replace("value * 2", "1 / 0")
    harness.create_tool(source.replace("twice", "failing"), entrypoint="failing")
    with pytest.raises(ZeroDivisionError):
        await harness.call_tool("failing", value=3)


@pytest.mark.parametrize("expression", ['harness.unload_tool("add")', 'harness.create_tool("", entrypoint="x")'])
def test_loading_cannot_mutate_catalog_via_harness(expression):
    harness = JITHarness(tools=[add])
    with pytest.raises(RuntimeError, match="loading Python"):
        harness.create_tool(expression + "\n" + SOURCE, entrypoint="twice")
    assert harness.inspect_tool("add")["status"] == "loaded"


async def test_missing_base_bindings_allowed_until_called():
    harness = JITHarness(tools=[add])
    harness.create_tool(SOURCE, entrypoint="twice")
    missing = JITHarness(harness.snapshot())
    assert missing.inspect_tool("add")["status"] == "missing"
    with pytest.raises(ValueError, match="Missing base tool"):
        await missing.call_tool("twice", value=1)
    with pytest.raises(ValueError, match="Missing base tool"):
        missing.load_tool("add")
    with pytest.raises(ValueError, match="Only dynamic"):
        harness.update_tool("add", PURE_SOURCE, entrypoint="twice")
    with pytest.raises(ValueError, match="already exists"):
        harness.create_tool(PURE_SOURCE.replace("twice", "add"), entrypoint="add")
    for operation in (missing.unload_tool, missing.load_tool, missing.inspect_tool):
        with pytest.raises(ValueError, match="Unknown tool"):
            operation("unknown")


@pytest.mark.parametrize(
    "payload",
    [
        {"format_version": 3},
        {
            "format_version": 1,
            "tools": [
                {
                    "name": "x",
                    "source": "value = 1",
                    "entrypoint": "value",
                    "description": "Describe the tool.",
                    "parameters": {},
                    "guidelines": ["Use it."],
                    "files": {"helpers.py": "value = 2"},
                }
            ],
        },
        {"base_tools": ["x", "x"]},
        {"base_tools": ["bad name"]},
        {"unloaded_tools": ["missing"]},
        {"base_tools": ["x"], "unloaded_tools": ["x", "x"]},
        {"unexpected": True},
    ],
)
def test_snapshot_rejects_invalid_state(payload):
    with pytest.raises(ValidationError):
        JITHarnessSnapshot.from_json(json.dumps(payload))


@pytest.mark.parametrize("names", [[], ["x"]])
def test_snapshot_rejects_old_unload_field_without_a_compatibility_alias(names):
    payload = {"base_tools": ["x"], "offloaded_tools": names}
    with pytest.raises(ValidationError, match="offloaded_tools"):
        JITHarnessSnapshot.model_validate(payload)
    with pytest.raises(ValidationError, match="offloaded_tools"):
        JITHarnessSnapshot.from_json(json.dumps(payload))
    assert "offloaded_tools" not in JITHarnessSnapshot.model_fields
    assert "unloaded_tools" in JITHarnessSnapshot.model_json_schema()["properties"]


def test_snapshot_read_never_executes_source_and_failed_save_preserves_file(tmp_path, monkeypatch):
    source = PythonToolSource(
        name="x",
        source='raise RuntimeError("executed")',
        entrypoint="x",
        description="x",
        parameters={},
        guidelines=("x",),
    )
    path = tmp_path / "snapshot.json"
    snapshot = JITHarnessSnapshot(tools=(source,))
    snapshot.save(path)
    assert JITHarnessSnapshot.load(path) == snapshot
    with pytest.raises(RuntimeError, match="executed"):
        JITHarness(snapshot)
    old = path.read_bytes()

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("zett_agent.jit_harness.snapshot.os.replace", fail)
    with pytest.raises(OSError):
        snapshot.save(path)
    assert path.read_bytes() == old and list(tmp_path.iterdir()) == [path]
    with pytest.raises(FileNotFoundError):
        snapshot.save(tmp_path / "absent" / "snapshot.json")
    with pytest.raises(ValidationError, match="identifier"):
        PythonToolSource(**{**source.model_dump(), "entrypoint": "!"})


@pytest.mark.parametrize("source", [SOURCE, SYNC_SOURCE])
async def test_shared_hooks_middleware_raw_results_and_only_outer_events(source):
    observed = []

    class Hooks(AgentExtension):
        async def before_tool(self, context, call):
            observed.append(("before", call.name))

        async def on_tool_call(self, context, call, call_next):
            observed.append(("middleware", call.name, current_tool_call_id()))
            return await call_next()

        async def after_tool(self, context, call, result, error):
            observed.append(("after", call.name, result.success))

    template = JITHarness(tools=[add])
    template.create_tool(source, entrypoint="twice")
    extension = JITHarnessExtension(template.snapshot())

    class Model(QuietModel):
        async def stream(self, request):
            if isinstance(request.messages[-1], ToolMessage):
                assert request.messages[-1].content == "42"
                assert sum(isinstance(item, ToolMessage) for item in request.messages) == 1
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="42")))
            else:
                yield ModelEvent.completed(
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "twice", {"value": 21}),)))
                )

    agent = await Agent.create(Model(), tools=[add], extensions=[Hooks(), extension])
    events = [event async for event in agent.stream("calculate", config=AgentRunConfig("nested"))]
    calls = [event for event in events if event.type in (AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED)]
    assert len(calls) == 2 and all(event.tool_calls[0].name == "twice" for event in calls)
    assert [(record[0], record[1]) for record in observed] == [
        ("before", "twice"),
        ("middleware", "twice"),
        ("before", "add"),
        ("middleware", "add"),
        ("after", "add"),
        ("after", "twice"),
    ]
    assert observed[1][2] == "outer" and observed[3][2] != "outer"
    assert current_tool_call_id() is None
    assert extension.get_harness("nested").active is False


async def test_old_wrapper_rechecks_update_unload_request_and_policy_removal():
    harness = JITHarness()
    harness.create_tool(PURE_SOURCE, entrypoint="twice")
    first = context()
    harness.attach(first)
    wrapper = first.tools["twice"]
    harness.update_tool("twice", PURE_SOURCE.replace("value * 2", "value * 3"), entrypoint="twice")
    assert await wrapper({"value": 2}) == 6
    harness.unload_tool("twice")
    assert "twice" not in first.tools
    with pytest.raises(ValueError, match="unloaded"):
        await wrapper({"value": 2})
    harness.load_tool("twice")
    first.tools.pop("twice")
    assert harness.inspect_tool("twice")["status"] == "unavailable"
    with pytest.raises(ValueError, match="not registered"):
        await wrapper({"value": 2})
    harness.detach(first)
    second = context()
    harness.attach(second)
    with pytest.raises(RuntimeError, match="closed"):
        await wrapper({"value": 2})
    with pytest.raises(RuntimeError, match="active"):
        harness.attach(context())
    with pytest.raises(RuntimeError, match="inside"):
        await harness.call_tool("twice", value=2)
    harness.detach(second)


async def test_nested_failure_hook_transformation_and_recursion_bound():
    class Hooks(AgentExtension):
        async def after_tool(self, context, call, result, error):
            if call.name == "add":
                result.content = "transformed"

    harness = JITHarness(tools=[add], max_call_depth=4)
    harness.create_tool(SOURCE, entrypoint="twice")
    ctx = context(add, extensions=(Hooks(),))
    harness.attach(ctx)
    assert await ctx.tools["twice"]({"value": 2}) == "transformed"
    with pytest.raises(ValidationError):
        await ctx.tools["twice"]({"value": "bad"})
    harness.detach(ctx)
    recursive = PURE_SOURCE.replace("value * 2", 'await harness.call_tool("twice", value=value)')
    harness.update_tool("twice", recursive, entrypoint="twice")
    with pytest.raises(RuntimeError, match="depth exceeded"):
        await harness.call_tool("twice", value=1)
    with pytest.raises(RuntimeError, match="synchronous JIT"):
        harness.call_tool_sync("add", left=1, right=2)
    harness.update_tool(
        "twice", SOURCE.replace("await harness.call_tool", "harness.call_tool_sync"), entrypoint="twice"
    )
    with pytest.raises(RuntimeError, match="event-loop"):
        await harness.call_tool("twice", value=1)


async def test_session_isolation_rebinding_compaction_and_forget():
    extension = JITHarnessExtension()
    agent = await Agent.create(QuietModel(), tools=[add], extensions=[extension])
    with pytest.raises(KeyError):
        extension.get_harness("absent")
    await asyncio.gather(*(agent.run("hi", config=AgentRunConfig(name)) for name in ("one", "two")))
    first = extension.get_harness("one")
    first.create_tool(SOURCE, entrypoint="twice")
    first.unload_tool("add")
    await agent.run("again", config=AgentRunConfig("one"))
    assert extension.get_harness("one") is first
    assert "twice" in first._sources
    assert extension.get_harness("two").inspect_tool("add")["status"] == "missing"
    assert "twice" not in extension.get_harness("two")._sources
    assert [event async for event in agent.compact(config=AgentRunConfig("one"))] == []
    assert first.active is False
    await agent.run("again", config=AgentRunConfig("one"))
    extension.forget_session("one")
    with pytest.raises(KeyError):
        extension.get_harness("one")


async def test_interactive_nested_tool_runs_before_hook_without_nested_messages():
    initial = JITHarness()
    source = dedent('''\
        @tool(guidelines="Ask the user for a decision.")
        async def decide() -> str:
            """Ask one question."""
            answer = await harness.call_tool("ask_user", question="Proceed?")
            return answer.payload["answer"]
    ''')
    initial.create_tool(source, entrypoint="decide")

    class Model(QuietModel):
        async def stream(self, request):
            if isinstance(request.messages[-1], ToolMessage):
                assert request.messages[-1].content == "yes"
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))
            else:
                yield ModelEvent.completed(
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "decide", {}),)))
                )

    jit = JITHarnessExtension(initial.snapshot())
    agent = await Agent.create(Model(), extensions=[AskUserExtension(), jit])
    async with aclosing(agent.stream("decide", config=AgentRunConfig("interactive"))) as events:
        async for event in events:
            if event.type == AgentEventType.CUSTOM and event.name == "ask_user":
                call_id = event.payload["tool_call_id"]
                assert call_id != "outer"
                agent.emit_external_event(
                    ExternalEvent("ask_user_response", {"tool_call_id": call_id, "answer": "yes"}),
                    config=AgentRunConfig("interactive"),
                )
    assert jit.get_harness("interactive").active is False


@pytest.mark.parametrize("responses", [False, True])
async def test_both_protocols_create_call_update_unload_load_and_replay(responses):
    steps = [
        ("create_tool", {"source": SOURCE, "entrypoint": "twice"}),
        ("twice", {"value": 21}),
        ("inspect_tool", {"name": "twice"}),
        ("unload_tool", {"name": "twice"}),
        ("list_tools", {}),
        (
            "update_tool",
            {"name": "twice", "source": SOURCE.replace("Add the same", "Combine the same"), "entrypoint": "twice"},
        ),
        ("load_tool", {"name": "twice"}),
        ("twice", {"value": 21}),
    ]
    payloads = []

    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        index = len(payloads) - 1
        # The Responses payload may also declare its hosted tool_search endpoint,
        # which is not a function and carries no name.
        names = {item["function"]["name"] if "function" in item else item.get("name") for item in payload["tools"]}
        assert ("twice" in names) == (index in (1, 2, 3, 7, 8, 9))
        assert request.url.path.endswith("/responses" if responses else "/chat/completions")
        if index >= len(steps):
            if responses:
                output = [
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "42", "annotations": []}],
                    }
                ]
            else:
                delta = {"content": "42"}
        else:
            name, arguments = steps[index]
            if responses:
                output = [
                    {
                        "type": "function_call",
                        "id": f"fc{index}",
                        "call_id": f"call{index}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                        "status": "completed",
                    }
                ]
            else:
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                }
        if responses:
            event = {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": f"r{index}",
                    "object": "response",
                    "created_at": 0,
                    "model": "test",
                    "status": "completed",
                    "output": output,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            }
            sse = f"event: response.completed\ndata: {json.dumps(event)}\n\n"
        else:
            event = {
                "id": f"r{index}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            sse = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    provider = OpenAIProvider(
        "test", "test", response=responses, transport=httpx.MockTransport(respond), retry=RetryOptions(max_retries=0)
    )
    extension = JITHarnessExtension()
    memory = InMemoryMessageAccumulator()
    try:
        agent = await Agent.create(
            provider,
            tools=[add],
            config=AgentRunConfig("wire"),
            extensions=[memory, ToolGuidelinesExtension(), extension],
        )
        assert (await agent.run("create and use tools")).content == "42"
        assert (await agent.run("reuse tools")).content == "42"
    finally:
        await provider.aclose()
    assert len(payloads) == 10
    assert all(
        tool.execution_mode is ToolExecutionMode.SERIAL
        for tool in extension._management_tools(extension.get_harness("wire"))
    )
    assert all(not isinstance(message, SystemMessage) for message in memory.messages("wire"))


async def test_management_tool_wrappers_serve_every_exposed_reader():
    """Each control tool is callable through the request registry it mounts in."""
    jit = JITHarnessExtension()
    ctx = context(add)
    await jit.on_tool(ctx)
    try:
        harness = jit.get_harness("test")
        harness.create_tool(PURE_SOURCE, entrypoint="twice")
        source = await ctx.tools["read_tool_source"]({"name": "twice"})
        assert source["content"] == PURE_SOURCE and source["next_offset"] is None
        schema = await ctx.tools["read_tool_schema"]({"name": "twice"})
        assert json.loads(schema["content"])["type"] == "object"
        assert (await ctx.tools["inspect_tool"]({"name": "twice"}))["entrypoint"] == "twice"
        assert (await ctx.tools["list_tools"]({}))["total_tools"] == 2
        assert (await ctx.tools["inspect_tool_stats"]({"name": "add"})).call_count == 0
        assert (await ctx.tools["inspect_tool_calls"]({"name": "add"}))["calls"] == []
        assert (await ctx.tools["inspect_tool_errors"]({}))["errors"] == []
    finally:
        jit._request_finished(ctx)


async def test_observation_stays_silent_for_a_session_without_a_harness():
    """A defensive guard: no harness means no JIT event, and no failure."""
    jit = JITHarnessExtension()
    ctx = context(add, session="unbound")
    call = ToolCall("call", "add", {"left": 1, "right": 1})
    result = ToolMessage(tool_call_id="call", name="add", content="2")
    assert jit._observed_origin(ctx, "add") is None
    await jit.before_tool(ctx, call)
    await jit.after_tool(ctx, call, result, None)
    assert jit._observed_origin(ctx, "add") is None


@pytest.mark.parametrize("mode", list(InspectionToolExposure))
def test_protocols_expose_what_they_support_without_losing_inspection_capability(mode):
    """Only Responses can defer; every other protocol still receives all definitions."""
    extension = JITHarnessExtension(inspection_tools=mode)
    management = extension._management_tools(JITHarness())
    definitions = tuple(registered.definition for registered in management) + (add.definition,)
    names = {item.name for item in definitions}
    inspection = set(_INSPECTION_TOOL_NAMES) & names
    deferred = {registered.name for registered in management if registered.deferred}
    assert deferred == (inspection if mode is InspectionToolExposure.DEFERRED else set())
    rendered = responses_tools(definitions, ())
    chat = chat_tools(definitions)
    # Chat Completions and every other non-Responses protocol ignore the flag.
    assert [item["function"]["name"] for item in chat] == [item.name for item in definitions]
    assert {item["name"] for item in rendered if "name" in item} == names
    assert {item["name"] for item in rendered if item.get("defer_loading")} == deferred
    assert ({"type": "tool_search"} in rendered) is bool(deferred)


@pytest.mark.parametrize("factory", [JITHarness, JITHarnessExtension])
def test_depth_configuration_is_validated(factory):
    with pytest.raises(ValueError, match="positive"):
        factory(max_call_depth=0)


def test_constructor_collisions_and_snapshot_source_name_mismatch():
    harness = JITHarness()
    harness.create_tool(PURE_SOURCE, entrypoint="twice")
    snapshot = harness.snapshot()
    with pytest.raises(ValueError, match="Duplicate"):
        JITHarness(tools=[add, add])
    with pytest.raises(ValueError, match="Duplicate"):
        JITHarness(snapshot, tools=[replace(add, name="twice")])
    with pytest.raises(ValueError, match="client-side"):
        JITHarness(tools=[replace(add, local_tool_search=True)])
    mismatched = snapshot.model_copy(update={"tools": (snapshot.tools[0].model_copy(update={"name": "other"}),)})
    with pytest.raises(ValueError, match="stored tool name"):
        JITHarness(mismatched)
    for candidate in (replace(add, name="twice"), replace(add, local_tool_search=True)):
        with pytest.raises(ValueError, match="conflicts|client-side"):
            harness.attach(context(candidate))
    assert harness.active is False


async def test_nested_exceptions_and_multimodal_results_keep_hooks_and_raw_values():
    seen = []

    class Hooks(AgentExtension):
        async def after_tool(self, context, call, result, error):
            seen.append((call.name, result.success, error))

    @tool(guidelines="Return a value for tests.")
    async def result(kind: str):
        """Return different result types."""
        if kind == "failure":
            raise ValueError("nested failure")
        if kind == "text":
            return TextContent("text")
        return [TextContent("text"), ImageContent(ImageBytesSource(b"image", "image/png"))]

    harness = JITHarness(tools=[result])
    source = dedent('''\
        @tool(guidelines="Return a dependency result.")
        async def proxy(kind: str):
            """Proxy the result tool."""
            return await harness.call_tool("result", kind=kind)
    ''')
    harness.create_tool(source, entrypoint="proxy")
    ctx = context(result, extensions=(Hooks(),))
    harness.attach(ctx)
    assert await ctx.tools["proxy"]({"kind": "text"}) == TextContent("text")
    assert len(await ctx.tools["proxy"]({"kind": "parts"})) == 2
    with pytest.raises(ValueError, match="nested failure"):
        await ctx.tools["proxy"]({"kind": "failure"})
    assert seen[-1][0:2] == ("result", False) and isinstance(seen[-1][2], ValueError)
    assert not harness._nested_tasks and current_tool_call_id() is None
    harness.detach(ctx)


async def test_sync_composition_does_not_depend_on_available_default_pool_workers():
    # A single default worker would deadlock a pooling parent→child bridge.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    harness = JITHarness(tools=[add])
    harness.create_tool(SYNC_SOURCE, entrypoint="twice")
    source = SYNC_SOURCE.replace("twice", "four_times").replace(
        '"add", left=value, right=value', '"twice", value=value * 2'
    )
    harness.create_tool(source, entrypoint="four_times")
    assert await asyncio.wait_for(harness.call_tool("four_times", value=3), timeout=2) == 12
    assert _SYNC_TOOL_RUNNER.get() is None


async def test_inflight_execution_keeps_old_implementation_during_update_and_unload():
    entered, finish = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Wait for controlled tests.")
    async def wait() -> int:
        """Wait and then return."""
        entered.set()
        await finish.wait()
        return 2

    harness = JITHarness(tools=[wait])
    source = PURE_SOURCE.replace("value * 2", '(await harness.call_tool("wait")) * value')
    harness.create_tool(source, entrypoint="twice")
    old = asyncio.create_task(harness.call_tool("twice", value=3))
    await entered.wait()
    harness.update_tool("twice", PURE_SOURCE.replace("value * 2", "value * 3"), entrypoint="twice")
    assert await harness.call_tool("twice", value=3) == 9
    harness.unload_tool("twice")
    finish.set()
    assert await old == 6
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("twice", value=3)


async def test_binding_cleanup_after_model_failure_and_registration_conflict():
    extension = JITHarnessExtension()

    class Fail(QuietModel):
        async def stream(self, request):
            raise RuntimeError("model failed")
            yield

    agent = await Agent.create(Fail(), tools=[add], config=AgentRunConfig("failure"), extensions=[extension])
    with pytest.raises(RuntimeError, match="model failed"):
        await agent.run("fail")
    assert extension.get_harness("failure").active is False
    await agent.run("recover", model=QuietModel())
    reserved = replace(add, name="create_tool")
    conflict = await Agent.create(
        QuietModel(), tools=[reserved], config=AgentRunConfig("conflict"), extensions=[extension]
    )
    with pytest.raises(ValueError, match="reserved"):
        await conflict.run("fail")
    assert extension.get_harness("conflict").active is False
    extension._release(context(session="never started"))


async def test_cancelled_sync_tool_cannot_call_or_mutate_a_later_request(monkeypatch):
    entered, release, finished = Event(), Event(), Event()
    errors = []

    @tool(guidelines="Wait for the test to release the worker.")
    def wait() -> None:
        """Block outside the loop until the test releases the thread."""
        entered.set()
        release.wait(5)

    @tool(guidelines="Collect a worker's late errors.")
    def report() -> None:
        """Mark the late worker as finished."""
        finished.set()

    state = ModuleType("zett_jit_test_state")
    state.__dict__.update(entered=entered, release=release, finished=finished, errors=errors)
    monkeypatch.setitem(sys.modules, state.__name__, state)
    source = dedent('''\
        import zett_jit_test_state as state
        @tool(guidelines="Exercise cancellation.")
        def late() -> None:
            """Wait, then attempt calls after request cancellation."""
            state.entered.set()
            state.release.wait(5)
            for operation in (
                lambda: harness.call_tool_sync("report"),
                lambda: harness.unload_tool("report"),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    state.errors.append(str(error))
            state.finished.set()
    ''')
    initial = JITHarness()
    initial.create_tool(source, entrypoint="late")
    extension = JITHarnessExtension(initial.snapshot())

    class Model(QuietModel):
        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "late", {}),))))

    agent = await Agent.create(
        Model(), tools=[wait, report], config=AgentRunConfig("cancelled"), extensions=[extension]
    )
    running = asyncio.create_task(agent.run("wait"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        harness = extension.get_harness("cancelled")
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert harness.active is False
        new = context(wait, report, session="cancelled")
        harness.attach(new)
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        assert len(errors) == 2 and all("closed" in error for error in errors)
        assert harness.inspect_tool("report")["status"] == "loaded"
        harness.detach(new)
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_forget_active_session_and_policy_filtered_guidance():
    extension = JITHarnessExtension()
    ctx = context(add)
    await extension.on_tool(ctx)
    with pytest.raises(RuntimeError, match="active"):
        extension.forget_session("test")
    harness = extension.get_harness("test")
    harness.unload_tool("add")
    request = ModelRequest(
        messages=(SystemMessage(content="stale", attributes={"tool_guidelines": True}),),
        tools=tuple(tool.definition for tool in ctx.tools.values()),
    )
    recorded = []

    async def next_(request):
        recorded.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    assert [event async for event in extension.on_model_request(ctx, request, next_)]
    assert "stale" not in "\n".join(item.content for item in recorded[-1].messages)
    assert "## add" not in recorded[-1].messages[0].content
    filtered = replace(request, tools=())
    assert [event async for event in extension.on_model_request(ctx, filtered, next_)]
    assert recorded[-1] is filtered
    harness.detach(ctx)
    extension.forget_session("test")


async def test_management_wrappers_reject_old_request_references():
    extension = JITHarnessExtension()
    first = context(add)
    await extension.on_tool(first)
    controls = {name: first.tools[name] for name in ("list_tools", "unload_tool")}
    harness = extension.get_harness("test")
    harness.detach(first)
    for name, control in controls.items():
        with pytest.raises(RuntimeError, match="closed"):
            await control({} if name == "list_tools" else {"name": "add"})
    second = context(add)
    await extension.on_tool(second)
    for name, control in controls.items():
        with pytest.raises(RuntimeError, match="closed"):
            await control({} if name == "list_tools" else {"name": "add"})
    assert harness.inspect_tool("add")["status"] == "loaded"
    assert await second.tools["list_tools"]({})
    current = second.tools.pop("list_tools")
    with pytest.raises(ValueError, match="not registered"):
        await current({})
    harness.detach(second)


async def test_binding_cleanup_survives_earlier_terminal_hook_failures():
    class BrokenHooks(AgentExtension):
        async def on_event(self, context, event):
            if context.state.phase.value in ("failed", "cancelled"):
                raise RuntimeError("terminal event failed")

        async def on_error(self, context, error):
            raise RuntimeError("error hook failed")

    class Fail(QuietModel):
        async def stream(self, request):
            raise ValueError("model failed")
            yield

    jit = JITHarnessExtension()
    agent = await Agent.create(Fail(), tools=[add], extensions=[BrokenHooks(), jit])
    with pytest.raises(ValueError, match="model failed"):
        await agent.run("fail", config=AgentRunConfig("failure"))
    assert jit.get_harness("failure").active is False
    assert not jit.pending
    assert (await agent.run("recover", config=AgentRunConfig("failure"), model=QuietModel())).content == "ok"


async def test_cancelled_sync_bridge_releases_nested_task_and_worker():
    entered, cancelled = asyncio.Event(), asyncio.Event()
    finished = Event()
    jit = JITHarnessExtension()

    @tool(guidelines="Wait until the test cancels the request.")
    async def wait() -> None:
        """Wait for request cancellation."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    @tool(guidelines="Compose the waiting dependency from a sync worker.")
    def parent() -> None:
        """Bridge to the request loop and always signal worker completion."""
        try:
            jit.get_harness("bridge").call_tool_sync("wait")
        finally:
            finished.set()

    class Model(QuietModel):
        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "parent", {}),))))

    agent = await Agent.create(Model(), tools=[wait, parent], extensions=[jit])
    running = asyncio.create_task(agent.run("wait", config=AgentRunConfig("bridge")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        assert await asyncio.to_thread(finished.wait, 2)
        harness = jit.get_harness("bridge")
        assert harness.active is False and not harness._nested_tasks
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
