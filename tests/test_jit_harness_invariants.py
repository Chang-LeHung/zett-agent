"""Adversarial JIT lifecycle checks against an independent catalog/state oracle."""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
from contextlib import suppress
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest
from pydantic import ValidationError

from jit_support import HEADER, put_tool, seed_repo, write_module
from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.events import AgentEventType
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.jit_harness.repo import MANAGEMENT_NAMES
from zett_agent.jit_harness.runtime import _BINDING, _CALL_STACK
from zett_agent.messages import AssistantMessage, SystemMessage, TextContent, ToolCall, ToolMessage, UserMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.tools.base import _SYNC_TOOL_RUNNER, current_tool_call_id, tool

SOURCE = dedent('''\
    @tool(guidelines="Compose exact addition.")
    async def twice(value: int) -> int:
        """Add an integer to itself."""
        return await harness.call_tool("add", left=value, right=value)
''')
PURE_SOURCE = SOURCE.replace('await harness.call_tool("add", left=value, right=value)', "value * 2")


@tool(guidelines="Use for exact integer addition.")
async def add(left: int, right: int) -> int:
    """Add two integers without a thread or network connection."""
    return left + right


@tool(guidelines="Write one file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


def context(*tools, session="invariants", extensions=()):
    return AgentRunContext(AgentRunConfig(session), AgentState(), {item.name: item for item in tools}, extensions)


def assert_unbound():
    """Invocation-local state must never leak into the caller's task."""
    assert _BINDING.get() is None
    assert _CALL_STACK.get() == ()
    assert _SYNC_TOOL_RUNNER.get() is None
    assert current_tool_call_id() is None


def jit_modules():
    """Every module the repository import hook has created."""
    return [name for name in sys.modules if name.startswith("zett_jit_")]


def jit_finders():
    return [finder for finder in sys.meta_path if type(finder).__name__ == "RepoFinder"]


@pytest.fixture(autouse=True)
def no_residue():
    """Every test must leave the interpreter's import state as it found it."""
    before = (set(jit_modules()), len(jit_finders()))
    yield
    assert (set(jit_modules()), len(jit_finders())) == before


@pytest.fixture
def make_harness(tmp_path):
    """Build harnesses in the test's temporary directory and close every one at the end."""
    created: list[JITHarness] = []

    def factory(directory=None, **options):
        item = JITHarness(Path(directory) if directory is not None else tmp_path / f"harness{len(created)}", **options)
        created.append(item)
        return item

    yield factory
    for item in created:
        item.close()


def close_sessions(jit, *sessions):
    """Drop each session's modules; a harness still serving a request is left alone."""
    for session in sessions:
        with suppress(KeyError):
            if not jit.get_harness(session).active:
                jit.forget_session(session)


class ScriptModel:
    """Emit predetermined batches and capture all actual model-facing requests."""

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


OPERATIONS = ("unload_twice", "load_twice", "unload_add", "load_add", "update", "invalid_update", "restart")


@pytest.mark.parametrize("operations", itertools.product(OPERATIONS, repeat=3), ids=lambda value: "/".join(value))
async def test_catalog_state_machine_matches_reference_after_every_transition(operations, make_harness):
    """343 transition sequences; the oracle never reads private harness maps."""
    harness = make_harness(tools=[add])
    put_tool(harness, SOURCE, "twice")
    ctx = context(add)
    harness.attach(ctx)
    unloaded = set()
    pure = False
    try:
        for operation in operations:
            if operation.startswith("unload_"):
                name = operation.removeprefix("unload_")
                harness.unload_tool(name)
                unloaded.add(name)
            elif operation.startswith("load_"):
                name = operation.removeprefix("load_")
                harness.load_tool(name)
                unloaded.discard(name)
            elif operation == "update":
                put_tool(harness, PURE_SOURCE, "twice")
                pure = True
                unloaded.discard("twice")  # A reload registers the new version loaded.
            elif operation == "invalid_update":
                write_module(harness, "twice", "bad python !")
                with pytest.raises(ValueError):
                    harness.load_repo()
                # The failed load keeps the registered version; make the files loadable again.
                write_module(harness, "twice", PURE_SOURCE if pure else SOURCE)
            else:
                # A restart keeps only the files: a new harness adopts and loads them,
                # and the old request's wrappers are dead.
                old = ctx.tools.get("twice")
                directory = harness.repo_path.parent
                harness.detach(ctx)
                harness.close()
                ctx = context(add)
                harness = make_harness(directory, tools=[add])
                harness.attach(ctx)
                unloaded.clear()  # Unload state belongs to a harness; files are all that persist.
                if old is not None:
                    with pytest.raises(RuntimeError, match="closed"):
                        await old({"value": 3})

            expected_names = {"add", "twice"}
            assert set(ctx.tools) == expected_names - unloaded
            catalog = harness.list_tools()
            assert [item["name"] for item in catalog] == ["add", "twice"]
            assert {item["name"]: item["status"] for item in catalog} == {
                name: "unloaded" if name in unloaded else "loaded" for name in expected_names
            }
            if "twice" not in unloaded:
                if "add" in unloaded and not pure:
                    with pytest.raises(ValueError, match="unloaded"):
                        await ctx.tools["twice"]({"value": 3})
                else:
                    assert await ctx.tools["twice"]({"value": 3}) == 6
            assert not harness._nested_tasks
            assert_unbound()
    finally:
        harness.detach(ctx)


BAD_MODULES = (
    ("bad python !", "twice", "SyntaxError"),
    ("raise RuntimeError('module failed')\n" + PURE_SOURCE, "twice", "RuntimeError: module failed"),
    ("import zett_jit_dependency_that_does_not_exist\n" + PURE_SOURCE, "twice", "ModuleNotFoundError"),
    (PURE_SOURCE, "bad identifier!", "not an importable module name"),
    (PURE_SOURCE.replace("value: int", "value"), "twice", "Missing annotation for value"),
    (PURE_SOURCE.replace("@tool(", "@tool(deferred=True, "), "twice", "non-deferred"),
    (PURE_SOURCE.replace("@tool(", "@tool(local_tool_search=True, "), "twice", "non-deferred"),
    (PURE_SOURCE.replace("@tool(", '@tool(name="repo_load", '), "twice", "reserved"),
    (PURE_SOURCE.replace('guidelines="Compose exact addition."', 'guidelines=""'), "twice", "cannot be blank"),
)


@pytest.mark.parametrize("unloaded", [False, True])
@pytest.mark.parametrize(("source", "module", "message"), BAD_MODULES)
async def test_every_failed_load_preserves_registration_and_unload(
    unloaded,
    source,
    module,
    message,
    make_harness,
):
    harness = make_harness()
    put_tool(harness, PURE_SOURCE, "twice")
    ctx = context()
    harness.attach(ctx)
    wrapper = ctx.tools["twice"]
    if unloaded:
        harness.unload_tool("twice")
    registration = dict(ctx.tools)
    modules = set(sys.modules)
    write_module(harness, module, source)
    try:
        with pytest.raises(ValueError, match=message):
            harness.load_repo()
        assert ctx.tools == registration
        assert not harness._compiling
        assert not {name for name in set(sys.modules) - modules if name.startswith("zett_jit_")}
        if unloaded:
            with pytest.raises(ValueError, match="unloaded"):
                await wrapper({"value": 4})
            harness.load_tool("twice")
        assert await wrapper({"value": 4}) == 8
        assert_unbound()
    finally:
        harness.detach(ctx)


@pytest.mark.parametrize("name", ["", "_", "a" * 64, "1-hyphenated"])
async def test_wire_name_override_is_independent_of_the_module_name(name, make_harness):
    harness = make_harness()
    source = PURE_SOURCE.replace("@tool(", f"@tool(name={name!r}, ")
    put_tool(harness, source, "module_entry")
    # The existing @tool contract uses the function name for an empty override.
    wire_name = name or "twice"
    assert await harness.call_tool(wire_name, value=-3) == -6


@pytest.mark.parametrize("name", ["a" * 65, "space name", "中文", "newline\n", "period.name"])
def test_wire_name_boundaries_leave_no_partial_registration(name, make_harness):
    harness = make_harness()
    put_tool(harness, PURE_SOURCE.replace("@tool(", f"@tool(name={name!r}, "), "twice", reload=False)
    with pytest.raises(ValueError, match="Tool names"):
        harness.load_repo()
    assert harness.list_tools() == [] and not jit_modules()


@pytest.mark.parametrize("sync_layers", itertools.product([False, True], repeat=3))
async def test_three_level_mixed_sync_async_composition_at_exact_depth_boundary(sync_layers, make_harness):
    """All eight thread/loop arrangements, including sync→async→sync."""
    harness = make_harness(max_call_depth=3)
    for index, synchronous in reversed(list(enumerate(sync_layers))):
        expression = (
            "value + 1"
            if index == 2
            else f'{"harness.call_tool_sync" if synchronous else "await harness.call_tool"}("level{index + 1}", value=value)'
        )
        source = (
            '@tool(guidelines="Use for layered composition.")\n'
            f"{'def' if synchronous else 'async def'} level{index}(value: int) -> int:\n"
            '    """Compose the next layer."""\n'
            f"    return {expression}\n"
        )
        put_tool(harness, source, f"level{index}")
    ctx = context()
    harness.attach(ctx)
    try:
        assert await asyncio.wait_for(ctx.tools["level0"]({"value": 41}), timeout=3) == 42
        harness.max_call_depth = 2
        with pytest.raises(RuntimeError, match="depth exceeded"):
            await asyncio.wait_for(ctx.tools["level0"]({"value": 41}), timeout=3)
        assert await ctx.tools["level2"]({"value": 1}) == 2
        assert not harness._nested_tasks
        assert_unbound()
    finally:
        harness.detach(ctx)


def test_describe_and_list_tools_are_defensive_copies(make_harness):
    harness = make_harness(tools=[add])
    put_tool(harness, SOURCE, "twice")
    expected = harness.list_tools()
    for entry in (harness.describe("add"), harness.describe("twice")):
        entry["name"] = "changed"
        entry["status"] = "unloaded"
        entry["description"] = "changed"
    harness.list_tools()[0]["name"] = "changed"
    assert harness.list_tools() == expected


async def test_session_harnesses_keep_independent_registrations_and_unload_state(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    seed_repo(tmp_path, "one", {"twice": SOURCE})
    seed_repo(tmp_path, "two", {"twice": SOURCE})
    one, two = context(add, session="one"), context(add, session="two")
    await jit.on_tool(one)
    await jit.on_tool(two)
    try:
        first, second = jit.get_harness("one"), jit.get_harness("two")
        one.tools["twice"].parameters["properties"].clear()
        assert "value" in two.tools["twice"].parameters["properties"]
        assert await one.tools["twice"]({"value": 3}) == 6
        first.unload_tool("add")
        assert await two.tools["twice"]({"value": 3}) == 6
        assert first.describe("add")["status"] == "unloaded"
        assert second.describe("add")["status"] == "loaded"
        assert first.describe("twice")["origin"] == "python"
        assert second.describe("twice")["origin"] == "python"
    finally:
        jit._request_finished(one)
        jit._request_finished(two)
        close_sessions(jit, "one", "two")


async def test_source_order_does_not_require_dependency_order(make_harness):
    harness = make_harness(tools=[add])
    put_tool(harness, SOURCE.replace("twice", "z_dependency"), "z_dependency", reload=False)
    parent = PURE_SOURCE.replace("twice", "a_parent").replace(
        "value * 2", 'await harness.call_tool("z_dependency", value=value)'
    )
    put_tool(harness, parent, "a_parent", reload=False)
    assert harness.load_repo()["tools"] == ["a_parent", "z_dependency"]
    assert await harness.call_tool("a_parent", value=21) == 42
    harness.unload_tool("z_dependency")
    assert harness.describe("a_parent")["status"] == "loaded"
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("a_parent", value=21)


async def test_module_globals_are_session_local_persist_across_requests_and_do_not_survive_a_restart(
    monkeypatch,
    tmp_path,
    make_harness,
):
    state = ModuleType("activation_state")
    state.activations = 0
    monkeypatch.setitem(sys.modules, state.__name__, state)
    source = dedent('''\
        import activation_state as state
        state.activations += 1
        count = 0
        @tool(guidelines="Count calls in this session.")
        async def counter() -> int:
            """Increment a module-local counter."""
            global count
            count += 1
            return count
    ''')
    directory = tmp_path / "local"
    original = make_harness(directory)
    put_tool(original, source, "counter")
    assert state.activations == 1
    assert await original.call_tool("counter") == 1
    original.unload_tool("counter")
    original.load_tool("counter")
    assert await original.call_tool("counter") == 2  # Unloading keeps the imported module.
    jit = JITHarnessExtension(tmp_path / "sessions")
    seed_repo(tmp_path / "sessions", "one", {"counter": source})
    seed_repo(tmp_path / "sessions", "two", {"counter": source})
    first, second = context(session="one"), context(session="two")
    try:
        await jit.on_tool(first)
        await jit.on_tool(second)
        assert state.activations == 3  # Each session imports its own copy.
        assert await first.tools["counter"]({}) == 1
        assert await first.tools["counter"]({}) == 2
        assert await second.tools["counter"]({}) == 1
        jit._request_finished(first)
        next_request = context(session="one")
        await jit.on_tool(next_request)
        try:
            assert state.activations == 3  # A later request reuses the session's modules.
            assert await next_request.tools["counter"]({}) == 3
        finally:
            jit._request_finished(next_request)
        # A fresh harness on the same directory imports the files again, so counters restart.
        restored = make_harness(directory)
        restored.create_repo("repo")
        assert state.activations == 4
        assert await restored.call_tool("counter") == 1
        assert await original.call_tool("counter") == 3
    finally:
        jit._request_finished(first)
        jit._request_finished(second)
        close_sessions(jit, "one", "two")


async def test_unload_and_load_keep_the_imported_module_but_a_changed_file_reimports(monkeypatch, make_harness):
    state = ModuleType("unloaded_activation_state")
    state.activations = 0
    monkeypatch.setitem(sys.modules, state.__name__, state)
    harness = make_harness()
    source = f"import {state.__name__} as state\nstate.activations += 1\n" + PURE_SOURCE
    put_tool(harness, source, "twice")
    assert state.activations == 1
    harness.unload_tool("twice")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("twice", value=1)
    harness.load_tool("twice")
    harness.load_tool("twice")
    assert state.activations == 1  # Restoring a registration does not execute the module again.
    assert await harness.call_tool("twice", value=1) == 2
    harness.load_repo()
    assert state.activations == 1  # Nothing changed on disk, so loading again imports nothing.
    put_tool(harness, source + "# edited\n", "twice")
    assert state.activations == 2  # A changed file is imported afresh.


@pytest.mark.parametrize("arguments", [{}, {"value": "bad"}, {"value": -1}, {"value": 2, "unknown": 3}])
async def test_nested_schema_constraints_defaults_and_extra_arguments(arguments, make_harness):
    source = dedent('''\
        from typing import Annotated
        from pydantic import Field
        @tool(guidelines="Multiply a nonnegative integer by a default scale.")
        async def checked(value: Annotated[int, Field(ge=0)], *, scale: int = 2, label: str | None = None) -> dict:
            """Validate constrained values before executing."""
            return {"value": value * scale, "label": label}
        @tool(guidelines="Forward keyword arguments through the harness validator.")
        async def forward(arguments: dict) -> dict:
            """Delegate to the constrained tool."""
            return await harness.call_tool("checked", **arguments)
    ''')
    harness = make_harness()
    put_tool(harness, source, "tools")
    ctx = context()
    harness.attach(ctx)
    try:
        with pytest.raises(ValidationError):
            await ctx.tools["forward"]({"arguments": arguments})
        assert await ctx.tools["forward"]({"arguments": {"value": 21}}) == {"value": 42, "label": None}
        assert await ctx.tools["forward"]({"arguments": {"value": 2, "scale": 3, "label": "x"}}) == {
            "value": 6,
            "label": "x",
        }
        assert not harness._nested_tasks
        assert_unbound()
    finally:
        harness.detach(ctx)


async def test_old_wrapper_uses_replaced_parameter_validator(make_harness):
    harness = make_harness()
    put_tool(harness, PURE_SOURCE, "twice")
    ctx = context()
    harness.attach(ctx)
    old = ctx.tools["twice"]
    changed = PURE_SOURCE.replace("value: int", "left: int, right: int").replace("value * 2", "left + right")
    try:
        put_tool(harness, changed, "twice")
        assert set(ctx.tools["twice"].parameters["properties"]) == {"left", "right"}
        with pytest.raises(ValidationError):
            await old({"value": 21})
        assert await old({"left": 20, "right": 22}) == 42
        assert_unbound()
    finally:
        harness.detach(ctx)


@pytest.mark.parametrize("stage", ["before", "middleware", "after"])
async def test_nested_hook_failures_unwind_call_id_depth_and_task_tracking(stage, make_harness):
    failure = RuntimeError(f"{stage} hook failed")
    seen = []

    class Hooks(AgentExtension):
        enabled = True

        async def before_tool(self, context, call):
            seen.append(("before", call.name, current_tool_call_id()))
            if self.enabled and stage == "before":
                raise failure

        async def on_tool_call(self, context, call, call_next):
            seen.append(("middleware", call.name, current_tool_call_id()))
            if self.enabled and stage == "middleware":
                raise failure
            return await call_next()

        async def after_tool(self, context, call, result, error):
            seen.append(("after", call.name, current_tool_call_id()))
            if self.enabled and stage == "after":
                raise failure

    hooks = Hooks()
    harness = make_harness(tools=[add], max_call_depth=2)
    put_tool(harness, SOURCE, "twice")
    ctx = context(add, extensions=(hooks,))
    harness.attach(ctx)
    try:
        with pytest.raises(RuntimeError) as raised:
            await ctx.tools["twice"]({"value": 21})
        assert raised.value is failure
        assert not harness._nested_tasks
        assert_unbound()
        hooks.enabled = False
        assert await ctx.tools["twice"]({"value": 21}) == 42
        assert all(record[1] == "add" and record[2] for record in seen)
        assert ctx.state.messages == () and ctx.event_queue.empty
    finally:
        harness.detach(ctx)


@pytest.mark.parametrize("stage", ["before", "middleware"])
async def test_unload_during_hook_is_checked_again_before_the_handler_starts(stage, make_harness):
    called = []

    @tool(guidelines="Observe whether a removed tool ran.")
    async def dependency() -> int:
        """Return only if still registered."""
        called.append(True)
        return 42

    harness = make_harness(tools=[dependency])
    source = PURE_SOURCE.replace("value * 2", 'await harness.call_tool("dependency")')
    put_tool(harness, source, "twice")

    class Hooks(AgentExtension):
        async def before_tool(self, context, call):
            if stage == "before":
                harness.unload_tool(call.name)

        async def on_tool_call(self, context, call, call_next):
            if stage == "middleware":
                harness.unload_tool(call.name)
            return await call_next()

    ctx = context(dependency, extensions=(Hooks(),))
    harness.attach(ctx)
    try:
        with pytest.raises(ValueError, match="unloaded"):
            await ctx.tools["twice"]({"value": 0})
        assert called == [] and "dependency" not in ctx.tools
        assert harness.describe("twice")["status"] == "loaded"
        assert not harness._nested_tasks
        assert_unbound()
    finally:
        harness.detach(ctx)


async def test_self_unload_can_finish_but_recursive_and_later_calls_fail(make_harness):
    source = dedent('''\
        @tool(guidelines="Unload after one successful invocation.")
        async def once() -> str:
            """Finish the current call, rejecting new calls to this tool."""
            harness.unload_tool("once")
            try:
                await harness.call_tool("once")
            except ValueError as error:
                return str(error)
            raise AssertionError("A self-unloaded tool cannot recurse")
    ''')
    harness = make_harness()
    put_tool(harness, source, "once")
    ctx = context()
    harness.attach(ctx)
    old = ctx.tools["once"]
    try:
        assert "unloaded" in await old({})
        assert "once" not in ctx.tools
        with pytest.raises(ValueError, match="unloaded"):
            await old({})
        assert harness.describe("once")["status"] == "unloaded"
        harness.load_tool("once")
        assert "unloaded" in await ctx.tools["once"]({})
        assert_unbound()
    finally:
        harness.detach(ctx)


async def test_parallel_fanout_has_unique_nested_ids_and_restores_each_parent_binding(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    nested_ids, completed, hooks = [], [], []

    @tool(guidelines="Synchronize parallel dependency invocations.")
    async def leaf(value: int) -> int:
        """Wait until all four independent nested calls have arrived."""
        identifier = current_tool_call_id()
        nested_ids.append(identifier)
        if len(nested_ids) == 4:
            entered.set()
        await release.wait()
        assert current_tool_call_id() == identifier
        completed.append(identifier)
        return value

    class Hooks(AgentExtension):
        async def before_tool(self, context, call):
            hooks.append(("before", call.name, call.id))

        async def after_tool(self, context, call, result, error):
            hooks.append(("after", call.name, call.id))
            assert result.success and error is None

    source = dedent('''\
        import asyncio
        from zett_agent.tools.base import current_tool_call_id
        @tool(guidelines="Fan out to two independent leaf calls.")
        async def fanout(value: int) -> int:
            """Preserve the parent invocation while children run concurrently."""
            parent = current_tool_call_id()
            results = await asyncio.gather(
                harness.call_tool("leaf", value=value),
                harness.call_tool("leaf", value=value + 1),
            )
            assert current_tool_call_id() == parent
            return sum(results)
    ''')
    seed_repo(tmp_path, "parallel", {"fanout": source})
    jit, history = JITHarnessExtension(tmp_path), InMemoryMessageAccumulator()
    model = ScriptModel((ToolCall("outer-a", "fanout", {"value": 1}), ToolCall("outer-b", "fanout", {"value": 10})))
    agent = await Agent.create(model, tools=[leaf], extensions=[history, Hooks(), jit])
    running = asyncio.create_task(agent.run("fan out", config=AgentRunConfig("parallel")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        harness = jit.get_harness("parallel")
        assert len(harness._nested_tasks) == 4
        assert len(set(nested_ids)) == 4 and not set(nested_ids) & {"outer-a", "outer-b"}
        assert_unbound()
        release.set()
        assert (await asyncio.wait_for(running, timeout=2)).content == "done"
        assert set(completed) == set(nested_ids)
        assert not harness._nested_tasks and harness.active is False
        results = [message for message in history.messages("parallel") if isinstance(message, ToolMessage)]
        assert {message.tool_call_id: message.content for message in results} == {"outer-a": "3", "outer-b": "21"}
        for identifier in ("outer-a", "outer-b", *nested_ids):
            assert [stage for stage, _, call_id in hooks if call_id == identifier] == ["before", "after"]
        assert_unbound()
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        close_sessions(jit, "parallel")


async def test_management_batches_respect_serial_order_and_keep_complete_outer_protocol(tmp_path):
    root, session = tmp_path / "jit", "batch"
    target = root / session / "kit"
    model = ScriptModel(
        (
            ToolCall("create", "repo_create", {"repo": "kit"}),
            ToolCall("duplicate", "repo_create", {"repo": "other"}),
        ),
        (ToolCall("write", "put", {"path": str(target / "twice.py"), "content": HEADER + PURE_SOURCE}),),
        (ToolCall("load", "repo_load", {}),),
        (
            ToolCall("call", "twice", {"value": 21}),
            ToolCall("unload", "repo_unload", {}),
        ),
        (
            ToolCall("stale", "twice", {"value": 3}),
            ToolCall("reload", "repo_load", {}),
        ),
        (ToolCall("reloaded", "twice", {"value": 21}),),
    )
    history, jit = InMemoryMessageAccumulator(), JITHarnessExtension(root)
    agent = await Agent.create(model, tools=[put], extensions=[history, jit])
    events = [event async for event in agent.stream("evolve", config=AgentRunConfig(session))]
    results = [message for message in history.messages(session) if isinstance(message, ToolMessage)]
    by_id = {message.tool_call_id: message for message in results}
    assert len(results) == 9 and len(by_id) == 9
    assert {identifier for identifier, result in by_id.items() if not result.success} == {"duplicate", "stale"}
    assert by_id["call"].content == by_id["reloaded"].content == "42"
    assert json.loads(by_id["unload"].content)["status"] == "unloaded"
    # Parallel handlers finish before the serial management calls of the same batch.
    assert [result.tool_call_id for result in results][4:6] == ["call", "unload"]
    for index, request in enumerate(model.requests):
        assert ("twice" in {definition.name for definition in request.tools}) == (index in (3, 5, 6))
    terminals = [event for event in events if event.type in (AgentEventType.TOOL_COMPLETED, AgentEventType.TOOL_FAILED)]
    assert len(terminals) == 9
    assert {event.tool_calls[0].id for event in terminals} == set(by_id)
    harness = jit.get_harness(session)
    assert harness.describe("twice")["status"] == "loaded" and harness.describe("twice")["origin"] == "python"
    assert_unbound()
    close_sessions(jit, session)


@pytest.mark.parametrize("name", sorted(MANAGEMENT_NAMES))
async def test_nested_composition_cannot_dispatch_reserved_management_functions(name, tmp_path):
    source = PURE_SOURCE.replace("value * 2", f"await harness.call_tool({name!r})")
    seed_repo(tmp_path, "invariants", {"twice": source})
    jit = JITHarnessExtension(tmp_path)
    ctx = context()
    await jit.on_tool(ctx)
    try:
        with pytest.raises(ValueError, match="management"):
            await ctx.tools["twice"]({"value": 1})
        assert set(ctx.tools) == MANAGEMENT_NAMES | {"twice"}
        assert jit.get_harness("invariants").describe("twice")["status"] == "loaded"
        assert_unbound()
    finally:
        jit._request_finished(ctx)
        close_sessions(jit, "invariants")


async def test_request_scoped_base_rebinding_changes_implementation_without_leaking_other_session(tmp_path):
    class Bases(AgentExtension):
        async def on_tool(self, ctx):
            marker = ctx.input_message.content

            @tool(guidelines="Read this request's resource binding.")
            async def resource() -> str:
                """Return a marker from the current request only."""
                return marker

            ctx.register_tool(resource)

    source = dedent('''\
        @tool(guidelines="Compose the current request's resource.")
        async def read_current() -> str:
            """Never retain a resource from a previous request."""
            return await harness.call_tool("resource")
    ''')
    seed_repo(tmp_path, "one", {"read_current": source})
    seed_repo(tmp_path, "two", {"read_current": source})
    jit, history = JITHarnessExtension(tmp_path), InMemoryMessageAccumulator()

    class Model:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            if isinstance(request.messages[-1], ToolMessage):
                message = AssistantMessage(content=request.messages[-1].content)
            else:
                message = AssistantMessage(tool_calls=(ToolCall("outer", "read_current", {}),))
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(Model(), extensions=[history, Bases(), jit])
    one, two = AgentRunConfig("one"), AgentRunConfig("two")
    results = await asyncio.gather(agent.run("first", config=one), agent.run("other", config=two))
    assert [result.content for result in results] == ["first", "other"]
    first, second = jit.get_harness("one"), jit.get_harness("two")
    assert first.describe("resource")["status"] == second.describe("resource")["status"] == "missing"
    assert (await agent.run("second", config=one)).content == "second"
    first.unload_tool("resource")
    failed = await agent.run("blocked", config=one)
    assert "unloaded" in failed.content
    assert (await agent.run("independent", config=two)).content == "independent"
    # A missing request-scoped tool cannot be loaded while idle.
    with pytest.raises(ValueError, match="Missing base tool"):
        first.load_tool("resource")
    assert first.describe("resource")["status"] == "unloaded"
    assert second.describe("resource")["status"] == "missing"  # Idle and unbound, but never unloaded.
    assert "resource" not in second._unloaded
    assert_unbound()
    close_sessions(jit, "one", "two")


async def test_cancellation_of_one_session_cleans_fanout_and_does_not_cancel_another_session(tmp_path):
    arrived, cancelled = asyncio.Event(), asyncio.Event()
    release_other = asyncio.Event()
    observed = []

    class Bases(AgentExtension):
        async def on_tool(self, ctx):
            session = ctx.config.session_id

            @tool(guidelines="Wait on a session-specific controlled resource.")
            async def wait(index: int) -> int:
                """Keep cancellation independent between sessions."""
                observed.append((session, index))
                if len(observed) == 4:
                    arrived.set()
                try:
                    await (release_other.wait() if session == "keep" else asyncio.Event().wait())
                    return index
                finally:
                    if session == "cancel":
                        cancelled.set()

            ctx.register_tool(wait)

    source = dedent('''\
        import asyncio
        @tool(guidelines="Compose two parallel dependencies.")
        async def pair() -> int:
            """Cancel all children with the outer request."""
            return sum(await asyncio.gather(
                harness.call_tool("wait", index=1), harness.call_tool("wait", index=2)
            ))
    ''')
    seed_repo(tmp_path, "cancel", {"pair": source})
    seed_repo(tmp_path, "keep", {"pair": source})
    jit, history = JITHarnessExtension(tmp_path), InMemoryMessageAccumulator()

    class Model:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            message = (
                AssistantMessage(content="done")
                if isinstance(request.messages[-1], ToolMessage)
                else AssistantMessage(tool_calls=(ToolCall("outer", "pair", {}),))
            )
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(Model(), extensions=[history, Bases(), jit])
    first = asyncio.create_task(agent.run("wait", config=AgentRunConfig("cancel")))
    second = asyncio.create_task(agent.run("wait", config=AgentRunConfig("keep")))
    try:
        await asyncio.wait_for(arrived.wait(), timeout=2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        stopped, kept = jit.get_harness("cancel"), jit.get_harness("keep")
        assert stopped.active is False and not stopped._nested_tasks
        assert kept.active is True and len(kept._nested_tasks) == 2
        assert not second.done()
        release_other.set()
        assert (await asyncio.wait_for(second, timeout=2)).content == "done"
        assert kept.active is False and not kept._nested_tasks
        failed = [item for item in history.messages("cancel") if isinstance(item, ToolMessage)]
        assert len(failed) == 1 and failed[0].tool_call_id == "outer" and not failed[0].success
        # A later request can replay a cancelled outer call/result without dangling nested IDs.
        recovery = ScriptModel()
        await agent.run("recover", config=AgentRunConfig("cancel"), model=recovery)
        replay = [item for item in recovery.requests[0].messages if isinstance(item, ToolMessage)]
        assert len(replay) == 1 and replay[0].tool_call_id == "outer"
        assert_unbound()
    finally:
        release_other.set()
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        close_sessions(jit, "cancel", "keep")


@pytest.mark.parametrize("mutation", ["unload_parent", "unload_dependency", "update_parent"])
async def test_inflight_request_keeps_started_implementation_but_old_wrappers_recheck_mutations(
    mutation,
    make_harness,
):
    entered, release = asyncio.Event(), asyncio.Event()
    invocations = []

    @tool(guidelines="Hold a started dependency at a deterministic boundary.")
    async def gate(value: int) -> int:
        """Let the test mutate the catalog after dispatch, before completion."""
        invocations.append(value)
        entered.set()
        await release.wait()
        return value

    harness = make_harness(tools=[gate])
    source = PURE_SOURCE.replace("value * 2", '(await harness.call_tool("gate", value=value)) * 2')
    put_tool(harness, source, "twice")
    ctx = context(gate)
    harness.attach(ctx)
    old = ctx.tools["twice"]
    running = asyncio.create_task(old({"value": 21}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        if mutation == "update_parent":
            put_tool(harness, PURE_SOURCE.replace("value * 2", "value * 3"), "twice")
            assert await old({"value": 2}) == 6
        else:
            harness.unload_tool("twice" if mutation == "unload_parent" else "gate")
            with pytest.raises(ValueError, match="unloaded"):
                await old({"value": 2})
        release.set()
        assert await asyncio.wait_for(running, timeout=2) == 42
        assert invocations == [21]
        assert not harness._nested_tasks
        assert_unbound()
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        harness.detach(ctx)


async def test_call_waiting_in_before_hook_cannot_enter_a_later_request(make_harness):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Hooks(AgentExtension):
        async def before_tool(self, context, call):
            entered.set()
            await release.wait()

    @tool(guidelines="Only run in the current request.")
    async def dependency() -> int:
        """Record dispatch after the pre-call hook."""
        calls.append(True)
        return 42

    harness = make_harness(tools=[dependency])
    put_tool(harness, PURE_SOURCE.replace("value * 2", 'await harness.call_tool("dependency")'), "twice")
    first = context(dependency, extensions=(Hooks(),))
    harness.attach(first)
    running = asyncio.create_task(first.tools["twice"]({"value": 0}))
    second = context(dependency)
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        harness.detach(first)
        harness.attach(second)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert calls == []
        assert await second.tools["twice"]({"value": 0}) == 42
        assert calls == [True] and not harness._nested_tasks
        harness.detach(first)  # Repeated late cleanup must not detach the new request.
        assert harness.active is True
        assert_unbound()
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        harness.detach(second)


@pytest.mark.parametrize("max_depth", [1, 2, 5])
async def test_mutual_recursion_hits_depth_bound_and_next_independent_call_still_works(max_depth, make_harness):
    harness = make_harness(max_call_depth=max_depth)
    for name, dependency in (("a", "b"), ("b", "a")):
        source = PURE_SOURCE.replace("twice", name).replace(
            "value * 2", f"await harness.call_tool({dependency!r}, value=value)"
        )
        put_tool(harness, source, name)
    put_tool(harness, PURE_SOURCE, "twice")
    ctx = context()
    harness.attach(ctx)
    try:
        with pytest.raises(RuntimeError, match=f"depth exceeded {max_depth}"):
            await ctx.tools["a"]({"value": 0})
        assert await ctx.tools["twice"]({"value": 21}) == 42
        assert not harness._nested_tasks
        assert_unbound()
    finally:
        harness.detach(ctx)


def test_a_module_body_cannot_dispatch_a_tool_while_it_loads(make_harness):
    harness = make_harness(tools=[add])
    before = harness.list_tools()
    source = 'import asyncio\nasyncio.run(harness.call_tool("add", left=1, right=2))\n' + PURE_SOURCE
    with pytest.raises(ValueError, match="Cannot call tools while loading"):
        put_tool(harness, source, "twice")
    assert harness.list_tools() == before and not harness._compiling


async def test_dynamic_guidance_has_only_current_definitions_and_never_mutates_input_request(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    ctx = context(add)
    ctx.add_message(SystemMessage(content="APPLICATION INSTRUCTIONS"))
    await jit.on_tool(ctx)
    await ToolGuidelinesExtension().on_state(ctx)
    ctx.add_message(UserMessage(content="go"))
    harness = jit.get_harness("invariants")
    recorded = []
    source = PURE_SOURCE.replace("Compose exact addition.", "FIRST_VERSION_GUIDANCE")
    put_tool(harness, source, "twice")

    async def next_(request):
        recorded.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    async def render():
        request = ModelRequest(messages=ctx.state.messages, tools=tuple(item.definition for item in ctx.tools.values()))
        before = request.messages
        assert [event async for event in jit.on_model_request(ctx, request, next_)]
        assert request.messages is before and ctx.state.messages == before
        emitted = recorded[-1]
        guidance_positions = [
            index
            for index, item in enumerate(emitted.messages)
            if isinstance(item, SystemMessage) and item.attributes.get("jit_harness:guidance", False)
        ]
        # After every application system message, before the dialogue.
        assert guidance_positions == [1]
        assert emitted.messages[0].content == "APPLICATION INSTRUCTIONS"
        assert isinstance(emitted.messages[2], UserMessage)
        assert not emitted.messages[guidance_positions[0]].persist
        return "\n".join(item.content for item in emitted.messages if isinstance(item, SystemMessage))

    try:
        initial_guidance = await render()
        assert "FIRST_VERSION_GUIDANCE" in initial_guidance
        assert "fewest total tokens and shortest end-to-end" in initial_guidance
        assert "correct, complete, satisfying answer" in initial_guidance
        assert "Never sacrifice" in initial_guidance
        assert "required work, safety, or answer quality" in initial_guidance
        assert "Actively use the JIT repository" in initial_guidance
        assert "writing and validation costs" in initial_guidance
        assert "avoid unnecessary tool-building" in initial_guidance
        assert "Prefer existing tools or a direct solution" in initial_guidance
        put_tool(harness, source.replace("FIRST_VERSION_GUIDANCE", "SECOND_VERSION_GUIDANCE"), "twice")
        updated = await render()
        assert "SECOND_VERSION_GUIDANCE" in updated and "FIRST_VERSION_GUIDANCE" not in updated
        harness.unload_tool("twice")
        unloaded = await render()
        assert "SECOND_VERSION_GUIDANCE" not in unloaded and "## twice" not in unloaded
        harness.load_tool("twice")
        assert "SECOND_VERSION_GUIDANCE" in await render()
        assert "SECOND_VERSION_GUIDANCE" not in "\n".join(item.content for item in ctx.state.messages)
    finally:
        jit._request_finished(ctx)
        close_sessions(jit, "invariants")


async def test_guidance_closes_the_leading_system_run_and_leaves_later_instructions(tmp_path):
    """A system message after the dialogue must not pull the block into the dialogue."""
    jit = JITHarnessExtension(tmp_path)
    ctx = context(add)
    await jit.on_tool(ctx)
    late = SystemMessage(content="LATE INSTRUCTION")
    request = ModelRequest(
        messages=(
            SystemMessage(content="FIRST"),
            SystemMessage(content="SECOND"),
            UserMessage(content="go"),
            late,
            UserMessage(content="again"),
        ),
        tools=tuple(item.definition for item in ctx.tools.values()),
    )
    recorded = []

    async def next_(request):
        recorded.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    try:
        assert [event async for event in jit.on_model_request(ctx, request, next_)]
        messages = recorded[-1].messages
        guidance = [
            index
            for index, item in enumerate(messages)
            if isinstance(item, SystemMessage) and item.attributes.get("jit_harness:guidance")
        ]
        # Counting every system message would insert at 3: after `go`, immediately
        # above `late`, i.e. inside the dialogue.
        assert guidance == [2]
        assert isinstance(messages[3], UserMessage)
        assert messages[3].content == "go"
        assert messages[4] is late
    finally:
        jit._request_finished(ctx)
        close_sessions(jit, "invariants")


@pytest.mark.parametrize("kind", ["plain", "text", "parts"])
async def test_in_place_result_hook_transformations_are_visible_to_composing_parent(kind, make_harness):
    class Redact(AgentExtension):
        async def after_tool(self, context, call, result, error):
            if isinstance(result.content, list):
                result.content[:] = [TextContent("redacted")]
            else:
                result.content = "redacted"

    @tool(guidelines="Return content that must be transformed by the result hook.")
    async def secret():
        """Return different result representations for hook redaction."""
        return "secret" if kind == "plain" else TextContent("secret") if kind == "text" else [TextContent("secret")]

    harness = make_harness(tools=[secret])
    put_tool(harness, PURE_SOURCE.replace("value * 2", 'await harness.call_tool("secret")'), "twice")
    ctx = context(secret, extensions=(Redact(),))
    harness.attach(ctx)
    try:
        result = await ctx.tools["twice"]({"value": 0})
        assert result == ("redacted" if kind == "plain" else [TextContent("redacted")])
        assert_unbound()
    finally:
        harness.detach(ctx)


async def test_concurrent_loads_leave_one_registered_definition_and_no_partial_state(make_harness):
    harness = make_harness()
    put_tool(harness, PURE_SOURCE, "twice")

    outcomes = await asyncio.gather(*(asyncio.to_thread(harness.load_repo) for _ in range(16)))
    assert [item["tools"] for item in outcomes] == [["twice"]] * 16
    assert harness.list_tools() == [harness.describe("twice")]
    assert len(jit_finders()) == 1  # Each reload replaced the previous package; the one shared hook stays.
    assert not harness._compiling
    assert await harness.call_tool("twice", value=21) == 42
    assert_unbound()


async def test_parallel_failed_child_can_be_caught_without_poisoning_siblings_or_parent_binding(make_harness):
    seen = []

    class Hooks(AgentExtension):
        async def after_tool(self, context, call, result, error):
            seen.append((call.id, call.arguments["value"], result.success, error))

    @tool(guidelines="Fail only for negative inputs.")
    async def partial(value: int) -> int:
        """Return or raise without leaking another task's invocation identity."""
        await asyncio.sleep(0)
        if value < 0:
            raise ValueError("negative input")
        return value

    source = dedent('''\
        import asyncio
        from zett_agent.tools.base import current_tool_call_id
        @tool(guidelines="Collect successful dependency results after a partial failure.")
        async def collect() -> list[int]:
            """Catch one child's failure and continue composing."""
            parent = current_tool_call_id()
            values = await asyncio.gather(*(
                harness.call_tool("partial", value=value) for value in (-1, 1, 2)
            ), return_exceptions=True)
            assert isinstance(values[0], ValueError)
            assert current_tool_call_id() == parent
            return [*values[1:], await harness.call_tool("partial", value=3)]
    ''')
    harness = make_harness(tools=[partial], max_call_depth=2)
    put_tool(harness, source, "collect")
    ctx = context(partial, extensions=(Hooks(),))
    harness.attach(ctx)
    try:
        assert await ctx.tools["collect"]({}) == [1, 2, 3]
        assert len(seen) == 4 and len({identifier for identifier, *_ in seen}) == 4
        assert {value: success for _, value, success, _ in seen} == {-1: False, 1: True, 2: True, 3: True}
        assert isinstance(next(error for _, value, _, error in seen if value == -1), ValueError)
        assert not harness._nested_tasks and ctx.state.messages == () and ctx.event_queue.empty
        assert_unbound()
    finally:
        harness.detach(ctx)


@pytest.mark.parametrize("representation", ["none", "empty", "dict", "dataclass", "pydantic"])
async def test_raw_python_result_shape_survives_composition_and_serializes_only_once(representation, tmp_path):
    source = dedent('''\
        from dataclasses import dataclass
        from pydantic import BaseModel
        @dataclass
        class Data:
            value: int
        class Model(BaseModel):
            value: int
        @tool(guidelines="Return a structured Python object.")
        async def structured(kind: str):
            """Return Python, not model-facing JSON text."""
            if kind == "none":
                return None
            if kind == "empty":
                return []
            if kind == "dict":
                return {"value": 42}
            return Data(42) if kind == "dataclass" else Model(value=42)
        @tool(guidelines="Inspect a dependency's raw result before returning it.")
        async def proxy(kind: str):
            """Preserve None, collections, and user-defined objects."""
            result = await harness.call_tool("structured", kind=kind)
            if kind == "none":
                assert result is None
            elif kind == "empty":
                assert result == [] and isinstance(result, list)
            elif kind == "dict":
                assert result["value"] == 42
            else:
                assert result.value == 42
            return result
    ''')
    seed_repo(tmp_path, "structured", {"tools": source})
    jit, history = JITHarnessExtension(tmp_path), InMemoryMessageAccumulator()
    model = ScriptModel((ToolCall("outer", "proxy", {"kind": representation}),))
    agent = await Agent.create(model, extensions=[history, jit])
    await agent.run("compose", config=AgentRunConfig("structured"))
    results = [item for item in history.messages("structured") if isinstance(item, ToolMessage)]
    assert len(results) == 1 and results[0].success
    assert json.loads(results[0].content) == (
        None if representation == "none" else [] if representation == "empty" else {"value": 42}
    )
    assert_unbound()
    close_sessions(jit, "structured")
