"""Optional base-tool ownership: lifecycle, late registration, policies, and profiling."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from textwrap import dedent

import pytest

from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.plan_mode import PlanModeExtension
from zett_agent.jit_harness import JITHarness, JITHarnessExtension, JITHarnessSnapshot, ToolOrigin
from zett_agent.jit_harness.runtime import _BINDING, _CALL_STACK, _PROFILE_CALL
from zett_agent.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.tools.base import _SYNC_TOOL_RUNNER, AgentTool, tool

SOURCE = dedent('''\
    @tool(guidelines="Compose the application's addition tool.")
    async def twice(value: int) -> int:
        """Add an integer to itself."""
        return await harness.call_tool("add", left=value, right=value)
''')
SYNC_SOURCE = SOURCE.replace("async def", "def").replace("await harness.call_tool", "harness.call_tool_sync")


@tool(guidelines="Use for exact integer addition.")
async def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def context(*tools, session="base-tools", extensions=()):
    return AgentRunContext(
        AgentRunConfig(session, request_id="request"),
        AgentState(),
        {registered.name: registered for registered in tools},
        extensions,
    )


class ScriptModel:
    retry = RetryOptions(max_retries=0)

    def __init__(self, *calls):
        self.calls = calls
        self.requests = []

    async def stream(self, request):
        index = len(self.requests)
        self.requests.append(request)
        message = (
            AssistantMessage(tool_calls=(self.calls[index],))
            if index < len(self.calls)
            else AssistantMessage(content="done")
        )
        yield ModelEvent.completed(ModelResponse(message))


def assert_unbound():
    assert _BINDING.get() is None
    assert _CALL_STACK.get() == ()
    assert _PROFILE_CALL.get() is None
    assert _SYNC_TOOL_RUNNER.get() is None


@pytest.mark.parametrize("factory", [JITHarness, JITHarnessExtension])
@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_base_management_requires_an_explicit_boolean(factory, value):
    with pytest.raises(ValueError, match="manage_base_tools must be a boolean"):
        factory(manage_base_tools=value)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("source", [SOURCE, SYNC_SOURCE], ids=["async-parent", "sync-parent"])
async def test_direct_and_composed_base_calls_follow_option_while_dynamic_calls_stay_managed(managed, source):
    initial = JITHarness()
    initial.create_tool(source, entrypoint="twice")
    jit = JITHarnessExtension(initial.snapshot(), manage_base_tools=managed)
    ctx = context(add, extensions=(jit,))
    await jit.on_tool(ctx)
    harness = jit.get_harness("base-tools")
    try:
        assert harness.manage_base_tools is managed
        assert (ctx.tools["add"].handler is add.handler) is not managed
        assert await ctx.tools["add"]({"left": 20, "right": 22}) == 42
        assert await ctx.tools["twice"]({"value": 21}) == 42
        assert harness.profile_stats("add").call_count == (2 if managed else 0)
        assert harness.profile_stats("twice").call_count == 1
        assert harness.origin("add") is ToolOrigin.BASE
        assert harness.definition("add")["parameters"] == add.parameters
        assert harness.definition("add").get("source") is None
        assert not harness._nested_tasks
        assert_unbound()

        before = harness.snapshot()
        if managed:
            assert await ctx.tools["unload_tool"]({"name": "add"}) == {"name": "add", "status": "unloaded"}
            assert "add" not in ctx.tools
            with pytest.raises(ValueError, match="unloaded"):
                await ctx.tools["twice"]({"value": 21})
            assert harness.profile_stats("add").rejected_count == 1
            assert await ctx.tools["load_tool"]({"name": "add"}) == {"name": "add", "status": "loaded"}
        else:
            for operation in ("unload_tool", "load_tool"):
                with pytest.raises(ValueError, match="Base tool management is disabled"):
                    await ctx.tools[operation]({"name": "add"})
                assert harness.snapshot() == before
                assert ctx.tools["add"].handler is add.handler
            assert harness.profile_stats("add").call_count == 0

        # This switch never removes dynamic-tool evolution.
        assert harness.unload_tool("twice")["status"] == "unloaded"
        assert "twice" not in ctx.tools
        assert harness.load_tool("twice")["status"] == "loaded"
        assert await ctx.tools["twice"]({"value": 21}) == 42
        assert harness.snapshot() == before
        assert_unbound()
    finally:
        jit._request_finished(ctx)
    assert not harness.active and not harness._wrapper_handlers


@pytest.mark.parametrize("managed", [False, True])
async def test_standalone_harness_applies_option_without_extension(managed):
    harness = JITHarness(tools=[add], manage_base_tools=managed)
    harness.create_tool(SYNC_SOURCE, entrypoint="twice")
    assert await harness.call_tool("add", left=1, right=2) == 3
    assert await harness.call_tool("twice", value=21) == 42
    assert harness.profile_stats("add").call_count == (2 if managed else 0)
    assert harness.profile_stats("twice").call_count == 1
    if not managed:
        with pytest.raises(ValueError, match="disabled"):
            harness.unload_tool("add")
    assert_unbound()


@pytest.mark.parametrize("registration", ["constructor", "early-extension", "late-extension", "before-model"])
@pytest.mark.parametrize("managed", [False, True])
async def test_real_model_call_profiles_and_unloads_every_base_registration_route(registration, managed):
    class Register(AgentExtension):
        priority = 950_000 if registration in ("late-extension", "before-model") else 100

        async def on_tool(self, ctx):
            if registration != "before-model":
                ctx.register_tool(add)

        async def before_model(self, ctx, request):
            if registration == "before-model":
                # Deliberately re-register every iteration. Enabled management
                # must not resurrect an unloaded base tool.
                ctx.tools.pop("add", None)
                ctx.register_tool(add)

    model = ScriptModel(
        ToolCall("base", "add", {"left": 21, "right": 21}),
        ToolCall("stats", "inspect_tool_stats", {"name": "add"}),
        ToolCall("unload", "unload_tool", {"name": "add"}),
        ToolCall("stale", "add", {"left": 1, "right": 1}),
        ToolCall("load", "load_tool", {"name": "add"}),
        ToolCall("again", "add", {"left": 2, "right": 2}),
    )
    jit = JITHarnessExtension(manage_base_tools=managed)
    extensions = [jit] if registration == "constructor" else [Register(), jit]
    agent = await Agent.create(model, tools=[add] if registration == "constructor" else [], extensions=extensions)
    assert (await agent.run("exercise base lifecycle", config=AgentRunConfig("routes"))).content == "done"
    results = {
        message.tool_call_id: message for message in model.requests[-1].messages if isinstance(message, ToolMessage)
    }
    assert results["base"].content == "42"
    assert json.loads(results["stats"].content)["call_count"] == (1 if managed else 0)
    assert results["unload"].success is managed
    assert results["stale"].success is not managed
    assert results["load"].success is managed
    assert results["again"].content == "4" and results["again"].success
    assert ("add" in {entry.name for entry in model.requests[3].tools}) is not managed
    if not managed:
        assert "disabled" in results["unload"].content
    harness = jit.get_harness("routes")
    assert harness.profile_stats("add").completed_count == (2 if managed else 0)
    assert not harness.active
    assert_unbound()


@pytest.mark.parametrize("managed", [False, True])
async def test_base_replaced_after_before_tool_runs_current_implementation_with_correct_profile(managed):
    executed = []

    @tool(guidelines="Use the replacement implementation only.")
    async def replacement(left: int, right: int) -> int:
        """Return a sentinel identifying the late implementation."""
        executed.append(True)
        return 99

    class ReplaceAfterJIT(AgentExtension):
        priority = 950_000

        async def before_tool(self, ctx, call):
            if call.name == "add":
                ctx.tools["add"] = replace(replacement, name="add")

    model = ScriptModel(ToolCall("base", "add", {"left": 1, "right": 2}))
    jit = JITHarnessExtension(manage_base_tools=managed)
    agent = await Agent.create(model, tools=[add], extensions=[jit, ReplaceAfterJIT()])
    await agent.run("replace", config=AgentRunConfig("replace"))
    result = next(message for message in model.requests[-1].messages if isinstance(message, ToolMessage))
    assert result.content == "99" and executed == [True]
    assert jit.get_harness("replace").profile_stats("add").call_count == (1 if managed else 0)
    assert_unbound()


async def test_sync_is_idempotent_and_does_not_re_adopt_wrappers_or_restore_removed_tools():
    harness = JITHarness(tools=[add])
    ctx = context(add)
    harness.attach(ctx)
    original = ctx.tools["add"]
    try:
        for _ in range(8):
            harness.sync_tools(ctx)
            assert ctx.tools["add"] is original
        assert await original({"left": 1, "right": 2}) == 3
        assert harness.profile_stats("add").call_count == 1

        ctx.tools.pop("add")
        harness.sync_tools(ctx)
        assert "add" not in ctx.tools and harness.describe("add")["status"] == "unavailable"
        ctx.tools["add"] = original  # Simulate a policy baseline restoring an old wrapper.
        harness.sync_tools(ctx)
        assert await ctx.tools["add"]({"left": 2, "right": 2}) == 4
        assert harness.profile_stats("add").call_count == 2

        harness.unload_tool("add")
        ctx.tools["add"] = original
        harness.sync_tools(ctx)
        assert "add" not in ctx.tools
        ctx.tools["add"] = add
        harness.sync_tools(ctx)
        assert "add" not in ctx.tools  # Even fresh raw registrations remain unloaded.
        harness.load_tool("add")
        assert await ctx.tools["add"]({"left": 21, "right": 21}) == 42
        assert harness.profile_stats("add").call_count == 3
    finally:
        harness.detach(ctx)
    assert not harness._wrapper_handlers
    with pytest.raises(RuntimeError, match="closed"):
        harness.sync_tools(ctx)


@pytest.mark.parametrize("invalid", ["key-mismatch", "dynamic-collision", "search", "reserved-name"])
async def test_sync_rejects_invalid_late_registration_without_half_adopting_earlier_entries(invalid):
    harness = JITHarness()
    harness.create_tool(SOURCE, entrypoint="twice")
    ctx = context()
    harness.attach(ctx)
    before = harness.snapshot()
    try:
        ctx.tools["good"] = replace(add, name="good")
        if invalid == "key-mismatch":
            ctx.tools["bad"] = add
        elif invalid == "dynamic-collision":
            ctx.tools["twice"] = replace(add, name="twice")
        elif invalid == "search":
            ctx.tools["search"] = replace(add, name="search", local_tool_search=True)
        else:
            ctx.tools["create_tool"] = replace(add, name="create_tool")
        with pytest.raises(ValueError):
            harness.sync_tools(ctx)
        assert harness.snapshot() == before
        assert harness.origin("good") is ToolOrigin.UNKNOWN
    finally:
        harness.detach(ctx)


@pytest.mark.parametrize("managed", [False, True])
async def test_plan_mode_replacements_are_wrapped_only_when_enabled_without_re_exposing_control_tools(managed):
    plan_mode = PlanModeExtension()
    jit = JITHarnessExtension(manage_base_tools=managed)
    # Plan Mode replaces registrations in a higher-priority before_model hook.
    ctx = context(session="plan")
    await jit.on_tool(ctx)
    await plan_mode.on_tool(ctx)
    original = {**ctx.tools}
    plan_mode._active_sessions.add("plan")
    await plan_mode.before_model(ctx, ModelRequest(messages=()))
    raw = ctx.tools["glob"]
    request = ModelRequest(messages=(), tools=tuple(registered.definition for registered in ctx.tools.values()))
    captured = []

    async def capture(current):
        captured.append(current)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    try:
        assert [event async for event in jit.on_model_request(ctx, request, capture)]
        assert "create_tool" not in ctx.tools
        assert "create_tool" not in {entry.name for entry in captured[0].tools}
        assert (ctx.tools["glob"].handler is raw.handler) is not managed
        assert jit.get_harness("plan").origin("glob") is ToolOrigin.BASE
        assert ctx.tools["exit_plan_mode"] is not original.get("exit_plan_mode")
        if managed:
            harness = jit.get_harness("plan")
            harness.unload_tool("glob")
            await plan_mode.before_model(ctx, request)  # Policy re-registers it.
            refreshed = ModelRequest(
                messages=(), tools=tuple(registered.definition for registered in ctx.tools.values())
            )
            assert [event async for event in jit.on_model_request(ctx, refreshed, capture)]
            assert "glob" not in ctx.tools and "glob" not in {entry.name for entry in captured[-1].tools}
            assert "create_tool" not in ctx.tools
    finally:
        plan_mode._active_sessions.discard("plan")
        jit._request_finished(ctx)


@pytest.mark.parametrize("managed", [False, True])
async def test_base_deferral_survives_harness_wrapping(managed):
    """Provider deferral is a base-tool property; management must not clear it."""
    jit = JITHarnessExtension(manage_base_tools=managed)
    ctx = context(replace(add, name="deferred_add", deferred=True))
    await jit.on_tool(ctx)
    try:
        assert ctx.tools["deferred_add"].deferred is True
        assert ctx.tools["deferred_add"].definition.deferred is True
        assert jit.get_harness("base-tools").origin("deferred_add") is ToolOrigin.BASE
    finally:
        jit._request_finished(ctx)


async def test_request_definitions_owned_by_other_extensions_are_preserved():
    """Only harness-owned names are filtered; foreign request definitions stay."""
    jit = JITHarnessExtension()
    ctx = context(add)
    await jit.on_tool(ctx)
    captured = []

    async def capture(request):
        captured.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    try:
        registered = tuple(item.definition for item in ctx.tools.values())
        request = ModelRequest(messages=(), tools=(*registered, replace(add, name="foreign").definition))
        assert [event async for event in jit.on_model_request(ctx, request, capture)]
        offered = {entry.name for entry in captured[0].tools}
        assert "foreign" in offered and {"add", "create_tool"} <= offered
        # A harness-owned name the request dropped is not re-exposed.
        dropped = ModelRequest(messages=(), tools=tuple(item for item in request.tools if item.name != "add"))
        assert [event async for event in jit.on_model_request(ctx, dropped, capture)]
        remaining = {entry.name for entry in captured[1].tools}
        assert "add" not in remaining and "foreign" in remaining
    finally:
        jit._request_finished(ctx)


@pytest.mark.parametrize("managed", [False, True])
async def test_base_failure_and_cancellation_record_only_when_managed(managed):
    entered, release = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Return, raise, or wait as directed by the test.")
    async def sample(kind: str) -> int:
        """Produce controlled base-tool outcomes."""
        if kind == "failure":
            raise ValueError("base failed")
        if kind == "wait":
            entered.set()
            await release.wait()
        return 42

    harness = JITHarness(tools=[sample], manage_base_tools=managed)
    with pytest.raises(ValueError, match="base failed"):
        await harness.call_tool("sample", kind="failure")
    running = asyncio.create_task(harness.call_tool("sample", kind="wait"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert await harness.call_tool("sample", kind="ok") == 42
        summary = harness.profile_stats("sample")
        assert summary.failed_count == (1 if managed else 0)
        assert summary.cancelled_count == (1 if managed else 0)
        assert summary.completed_count == (1 if managed else 0)
        assert len(harness.tool_errors("sample").errors) == (1 if managed else 0)
        assert_unbound()
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_disabled_mode_preserves_snapshot_unload_marks_without_applying_them_to_base_tools():
    template = JITHarness(tools=[add])
    template.create_tool(SOURCE, entrypoint="twice")
    template.unload_tool("add")
    template.unload_tool("twice")
    snapshot = JITHarnessSnapshot.from_json(template.snapshot().to_json())
    protected = JITHarness(snapshot, tools=[add], manage_base_tools=False)
    assert protected.describe("add")["status"] == "loaded"
    assert protected.describe("twice")["status"] == "unloaded"
    assert await protected.call_tool("add", left=1, right=2) == 3
    protected.load_tool("twice")
    assert await protected.call_tool("twice", value=21) == 42
    assert protected.profile_stats("add").call_count == 0
    assert protected.snapshot().unloaded_tools == ("add",)
    assert "manage_base_tools" not in protected.snapshot().to_json()
    managed = JITHarness(protected.snapshot(), tools=[add])
    with pytest.raises(ValueError, match="unloaded"):
        await managed.call_tool("twice", value=21)
    managed.load_tool("add")
    assert await managed.call_tool("twice", value=21) == 42


async def test_same_session_retains_unload_state_other_session_remains_independent_and_raw_tools_are_not_mutated():
    jit = JITHarnessExtension()
    history = InMemoryMessageAccumulator()
    agent = await Agent.create(ScriptModel(), tools=[add], extensions=[history, jit])
    await agent.run("one", config=AgentRunConfig("one"))
    first = jit.get_harness("one")
    first.unload_tool("add")
    one_model = ScriptModel(ToolCall("stale", "add", {"left": 1, "right": 2}))
    two_model = ScriptModel(ToolCall("base", "add", {"left": 1, "right": 2}))
    await asyncio.gather(
        agent.run("one", config=AgentRunConfig("one"), model=one_model),
        agent.run("two", config=AgentRunConfig("two"), model=two_model),
    )
    assert "add" not in {entry.name for entry in one_model.requests[0].tools}
    assert "add" in {entry.name for entry in two_model.requests[0].tools}
    assert jit.get_harness("two").profile_stats("add").completed_count == 1
    assert first.snapshot().unloaded_tools == ("add",)
    assert not first.active and not jit.get_harness("two").active
    assert await add({"left": 1, "right": 2}) == 3  # The application callable was not changed.


@pytest.mark.parametrize("managed", [False, True])
async def test_later_request_middleware_tool_subset_is_not_expanded(managed):
    jit = JITHarnessExtension(manage_base_tools=managed)
    ctx = context(add)
    await jit.on_tool(ctx)
    captured = []

    async def capture(request):
        captured.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    try:
        request = ModelRequest(
            tools=tuple(registered.definition for name, registered in ctx.tools.items() if name != "add"),
            messages=(SystemMessage(content="system"),),
        )
        assert [event async for event in jit.on_model_request(ctx, request, capture)]
        assert "add" not in {entry.name for entry in captured[0].tools}
        text = "\n".join(message.content for message in captured[0].messages if isinstance(message, SystemMessage))
        assert ("Base tool management is enabled" in text) is managed
        assert ("Base tool management is disabled" in text) is not managed
        assert "## add" not in text
    finally:
        jit._request_finished(ctx)


async def test_unhashable_custom_base_handler_is_accepted_by_registry_sync():
    class Handler:
        __hash__ = None

        async def __call__(self, **arguments):
            return 42

    raw = AgentTool("custom", "Return a value.", {}, Handler(), guidelines=("Use a custom handler.",))
    harness = JITHarness()
    ctx = context()
    harness.attach(ctx)
    try:
        ctx.register_tool(raw)
        harness.sync_tools(ctx)
        assert await ctx.tools["custom"]({}) == 42
        assert harness.profile_stats("custom").call_count == 1
    finally:
        harness.detach(ctx)
