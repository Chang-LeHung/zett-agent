"""Failure paths: loading, invocation, long chains, and recovery afterwards."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from jit_support import HEADER, put_tool, seed_repo, write_module
from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.jit_harness.enums import ToolCallStatus
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.tools.base import tool


def link_source(name: str, dependency: str | None = None) -> str:
    """Build one chain link that either returns or composes the next link."""
    body = "return 1" if dependency is None else f'return await harness.call_tool("{dependency}")'
    return dedent(f'''\
        @tool(guidelines="Use as one link in a composed chain.")
        async def {name}() -> int:
            """Chain link."""
            {body}
    ''')


def build_chain(harness: JITHarness, length: int, *, leaf: str | None = None) -> None:
    """Write ``link0`` .. ``linkN`` where each link composes the next one, then load them."""
    for index in range(length):
        is_leaf = index + 1 == length
        if is_leaf and leaf is not None:
            source = leaf
        else:
            source = link_source(f"link{index}", None if is_leaf else f"link{index + 1}")
        write_module(harness, f"link{index}", source)
    harness.load_repo()


FAILING_LEAF = dedent('''\
    @tool(guidelines="Use as one link in a composed chain.")
    async def link2() -> int:
        """Fail at the end of the chain."""
        raise ValueError("leaf exploded")
''')

OPAQUE_SOURCE = dedent('''\
    @tool(guidelines="Return something JSON cannot hold.")
    async def opaque():
        """Return an opaque object."""
        return object()
''')

DEFERRED_SOURCE = link_source("ok").replace("@tool(", "@tool(deferred=True, ")
SEARCH_SOURCE = link_source("ok").replace("@tool(", "@tool(local_tool_search=True, ")
UNANNOTATED_SOURCE = link_source("ok").replace("() -> int:", "(value) -> int:")
GUIDELESS_SOURCE = link_source("ok").replace('guidelines="Use as one link in a composed chain."', "")

#: A module body that must fail the whole load, with the error the caller sees.
#: ``load_repo`` raises ``ValueError`` for everything except an interrupted process.
LOAD_FAILURES = [
    ("syntax", "def broken(", ValueError, "SyntaxError"),
    ("module raise", 'raise RuntimeError("import blew up")\n', ValueError, "RuntimeError: import blew up"),
    ("missing import", "import absent_module_for_test\n", ValueError, "ModuleNotFoundError"),
    ("exit during load", "import sys\nsys.exit(4)\n", SystemExit, None),
    ("deferred tool", DEFERRED_SOURCE, ValueError, "non-deferred"),
    ("search tool", SEARCH_SOURCE, ValueError, "non-deferred"),
    ("missing annotation", UNANNOTATED_SOURCE, ValueError, "Missing annotation"),
    ("no guideline", GUIDELESS_SOURCE, ValueError, "non-empty guideline"),
]


@tool(guidelines="Use for exact integer addition.")
async def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@tool(guidelines="Write one file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


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


def statuses(harness, name) -> list[ToolCallStatus]:
    return [call.status for call in harness.profile(name).calls]


def jit_modules():
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
    """Create harnesses in their own directories and close every one afterwards."""
    created: list[JITHarness] = []

    def create(name: str = "session", **options: Any) -> JITHarness:
        harness = JITHarness(tmp_path / f"{name}-{len(created)}", **options)
        created.append(harness)
        return harness

    yield create
    for harness in created:
        harness.close()


async def test_failed_load_leaves_the_harness_clean_and_usable(make_harness):
    for label, source, error, message in LOAD_FAILURES:
        harness = make_harness(label, tools=[add])
        write_module(harness, "bad", source)
        with pytest.raises(error, match=message):
            harness.load_repo()

        assert [item["name"] for item in harness.list_tools()] == ["add"], label  # Nothing was registered.
        assert not jit_modules(), label  # A failed attempt leaves no imported module behind.
        # A wedged importer or half-written registry would break later operations.
        write_module(harness, "bad", link_source("recovered"))
        harness.load_repo()
        assert await harness.call_tool("recovered") == 1, label
        assert harness.profile("recovered").calls[0].status is ToolCallStatus.COMPLETED, label
        harness.close()


async def test_failed_load_leaves_an_existing_tool_updatable_and_callable(make_harness):
    for label, source, error, message in LOAD_FAILURES:
        harness = make_harness(label)
        put_tool(harness, link_source("target"), "target")
        bad = write_module(harness, "bad", source)
        with pytest.raises(error, match=message):
            harness.load_repo()
        bad.unlink()  # The next attempt fails on the target module itself.
        with pytest.raises(error, match=message):
            put_tool(harness, source, "target")

        assert [item["name"] for item in harness.list_tools()] == ["target"], label
        assert await harness.call_tool("target") == 1, label
        harness.unload_tool("target")
        harness.load_tool("target")
        assert await harness.call_tool("target") == 1, label
        # Direct API failures are not dispatches, so the LRU cache stays empty and
        # no failure is blamed on the tool that survived.
        assert harness.tool_errors(limit=10).errors == (), label
        harness.close()


async def test_module_side_effects_survive_a_failed_load_but_registration_does_not(monkeypatch, make_harness):
    state = ModuleType("jit_failure_state")
    state.ran = []
    monkeypatch.setitem(sys.modules, state.__name__, state)
    source = dedent('''\
        import jit_failure_state as state
        state.ran.append("imported")
        raise RuntimeError("failed after the side effect")

        @tool(guidelines="Never registered.")
        async def unreachable() -> int:
            """Never registered."""
            return 1
    ''')
    harness = make_harness()
    with pytest.raises(ValueError, match="failed after the side effect"):
        put_tool(harness, source, "unreachable")
    assert state.ran == ["imported"]  # Documented: module side effects are not rolled back.
    assert harness.list_tools() == []


async def test_handler_failure_records_every_layer_and_keeps_the_original_error(make_harness):
    harness = make_harness()
    build_chain(harness, 3, leaf=FAILING_LEAF)
    with pytest.raises(ValueError, match="leaf exploded"):
        await harness.call_tool("link0")

    assert [(call.tool_name, call.status, call.error_type) for call in harness.profile().calls] == [
        ("link0", ToolCallStatus.FAILED, "ValueError"),
        ("link1", ToolCallStatus.FAILED, "ValueError"),
        ("link2", ToolCallStatus.FAILED, "ValueError"),
    ]
    top, middle, leaf = (harness.profile(name).calls[0] for name in ("link0", "link1", "link2"))
    assert top.caller_tool_name is None
    assert middle.caller_tool_name == "link0" and leaf.caller_tool_name == "link1"
    # Each link reports its own entry, so a reader can see where the chain broke.
    assert [(error.tool_name, error.message) for error in harness.tool_errors(limit=10).errors] == [
        ("link0", "leaf exploded"),
        ("link1", "leaf exploded"),
        ("link2", "leaf exploded"),
    ]
    assert harness.profile("link0").stats[0].failed_count == 1
    assert not harness._nested_tasks


async def test_depth_limit_stops_a_long_chain_and_names_the_rejected_link(make_harness):
    harness = make_harness(max_call_depth=2)
    build_chain(harness, 4)
    with pytest.raises(RuntimeError, match="depth exceeded 2"):
        await harness.call_tool("link0")

    assert [(call.tool_name, call.status) for call in harness.profile().calls] == [
        ("link0", ToolCallStatus.FAILED),
        ("link1", ToolCallStatus.FAILED),
        ("link2", ToolCallStatus.REJECTED),
    ]
    assert statuses(harness, "link3") == []  # The link past the bound never started.
    # Recency follows completion, so the rejected link is reported first.
    assert [error.status for error in harness.tool_errors(limit=10).errors] == [
        ToolCallStatus.FAILED,
        ToolCallStatus.FAILED,
        ToolCallStatus.REJECTED,
    ]
    put_tool(harness, link_source("shallow"), "shallow")
    assert await harness.call_tool("shallow") == 1  # Depth is per call chain, not per harness.


async def test_long_chain_succeeds_at_the_exact_depth_boundary(make_harness):
    harness = make_harness(max_call_depth=8)
    build_chain(harness, 8)
    assert await harness.call_tool("link0") == 1
    assert statuses(harness, "link0") == [ToolCallStatus.COMPLETED]
    assert [call.status for call in harness.profile().calls] == [ToolCallStatus.COMPLETED] * 8
    assert [call.tool_name for call in harness.profile().calls] == [f"link{index}" for index in range(8)]
    assert not harness._nested_tasks


async def test_chain_one_link_past_the_bound_fails_at_that_link_only(make_harness):
    harness = make_harness(max_call_depth=7)
    build_chain(harness, 8)
    with pytest.raises(RuntimeError, match="depth exceeded 7"):
        await harness.call_tool("link0")
    assert statuses(harness, "link7") == [ToolCallStatus.REJECTED]
    assert statuses(harness, "link6") == [ToolCallStatus.FAILED]
    assert statuses(harness, "link0") == [ToolCallStatus.FAILED]
    assert await harness.call_tool("link7") == 1  # The leaf itself is fine.


async def test_a_middle_link_can_catch_the_leaf_failure_and_finish(make_harness):
    harness = make_harness()
    put_tool(harness, FAILING_LEAF.replace("link2", "leaf"), "leaf")
    put_tool(
        harness,
        dedent('''\
            @tool(guidelines="Use as one link in a composed chain.")
            async def recovery() -> int:
                """Catch the leaf failure and answer anyway."""
                try:
                    await harness.call_tool("leaf")
                except ValueError:
                    return 7
                raise AssertionError("the leaf must fail")
        '''),
        "recovery",
    )
    assert await harness.call_tool("recovery") == 7
    assert statuses(harness, "recovery") == [ToolCallStatus.COMPLETED]
    assert statuses(harness, "leaf") == [ToolCallStatus.FAILED]
    assert [error.tool_name for error in harness.tool_errors(limit=5).errors] == ["leaf"]


async def test_cancelling_a_long_chain_marks_every_running_link_cancelled(monkeypatch, make_harness):
    # JIT modules only see their own globals, so test state arrives as a module.
    state = ModuleType("jit_cancel_state")
    state.entered = asyncio.Event()
    state.release = asyncio.Event()
    monkeypatch.setitem(sys.modules, state.__name__, state)
    harness = make_harness()
    put_tool(
        harness,
        dedent('''\
            import jit_cancel_state as state

            @tool(guidelines="Use as one link in a composed chain.")
            async def leaf() -> int:
                """Wait until the test cancels the chain."""
                state.entered.set()
                await state.release.wait()
                return 1
        '''),
        "leaf",
    )
    put_tool(harness, link_source("middle", "leaf"), "middle")
    put_tool(harness, link_source("top", "middle"), "top")
    running = asyncio.create_task(harness.call_tool("top"))
    try:
        await asyncio.wait_for(state.entered.wait(), timeout=2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert statuses(harness, "top") == [ToolCallStatus.CANCELLED]
        assert statuses(harness, "middle") == [ToolCallStatus.CANCELLED]
        assert statuses(harness, "leaf") == [ToolCallStatus.CANCELLED]
        assert harness.tool_errors(limit=5).errors == ()  # Cancellation is not a cached error.
        assert not harness._nested_tasks
    finally:
        state.release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_nested_validation_failure_stops_the_dependency_before_its_body_runs(make_harness):
    ran = []

    @tool(guidelines="Record that the body ran.")
    async def record(value: int) -> int:
        """Record one call."""
        ran.append(value)
        return value

    harness = make_harness(tools=[record])
    put_tool(
        harness,
        dedent('''\
            @tool(guidelines="Forward the raw argument to the dependency.")
            async def forward(value: str) -> int:
                """Forward the raw argument."""
                return await harness.call_tool("record", value=value)
        '''),
        "forward",
    )
    with pytest.raises(ValidationError):
        await harness.call_tool("forward", value="not-an-int")
    assert ran == []
    assert statuses(harness, "record") == [ToolCallStatus.FAILED]
    assert harness.profile("record").calls[0].error_type == "ValidationError"
    assert statuses(harness, "forward") == [ToolCallStatus.FAILED]


async def test_unserializable_nested_result_fails_the_caller_without_corrupting_state(make_harness):
    harness = make_harness()
    put_tool(harness, OPAQUE_SOURCE, "opaque")
    put_tool(harness, link_source("parent", "opaque"), "parent")
    ctx = AgentRunContext(AgentRunConfig("serialize", request_id="request"), AgentState(), {}, ())
    harness.attach(ctx)
    try:
        with pytest.raises(PydanticSerializationError, match="[Uu]nable to serialize"):
            await ctx.tools["parent"]({})
        # The child handler did return, so its own record stays completed; the
        # caller is the call whose result could not be represented.
        assert statuses(harness, "opaque") == [ToolCallStatus.COMPLETED]
        assert statuses(harness, "parent") == [ToolCallStatus.FAILED]
        assert harness.profile("parent").calls[0].error_type == "PydanticSerializationError"
        assert [error.tool_name for error in harness.tool_errors(limit=5).errors] == ["parent"]
        assert not harness._nested_tasks
        put_tool(harness, link_source("later"), "later")
        assert await ctx.tools["later"]({}) == 1
    finally:
        harness.detach(ctx)


async def test_unserializable_result_without_a_request_reaches_the_caller_raw(make_harness):
    harness = make_harness()
    put_tool(harness, OPAQUE_SOURCE, "opaque")
    # Without a bound request there is no ToolMessage to build, so nothing
    # serializes the result and the raw object travels back to the caller.
    assert await harness.call_tool("opaque") is not None
    assert statuses(harness, "opaque") == [ToolCallStatus.COMPLETED]


async def test_model_call_returning_an_unserializable_value_becomes_a_failed_tool_message(tmp_path):
    seed_repo(tmp_path, "unserializable", {"opaque": OPAQUE_SOURCE})
    jit = JITHarnessExtension(tmp_path)
    model = ScriptModel((ToolCall("bad", "opaque", {}),))
    agent = await Agent.create(model, extensions=[jit])
    await agent.run("go", config=AgentRunConfig("unserializable"))

    results = [message for message in model.requests[-1].messages if isinstance(message, ToolMessage)]
    assert [(result.tool_call_id, result.success) for result in results] == [("bad", False)]
    assert "serialize" in results[0].content.lower()
    # The harness boundary is the handler returning, so its record stays completed;
    # the failure happens afterwards when the runtime builds the ToolMessage.
    assert statuses(jit.get_harness("unserializable"), "opaque") == [ToolCallStatus.COMPLETED]
    jit.forget_session("unserializable")


async def test_base_exception_from_a_handler_propagates_and_leaves_the_harness_reusable(make_harness):
    harness = make_harness()
    put_tool(
        harness,
        dedent('''\
            @tool(guidelines="Interrupt the current call.")
            async def interrupt() -> int:
                """Raise a BaseException the runtime does not convert."""
                raise KeyboardInterrupt("stopped by the operator")
        '''),
        "interrupt",
    )
    put_tool(harness, link_source("afterwards"), "afterwards")
    with pytest.raises(KeyboardInterrupt, match="stopped by the operator"):
        await harness.call_tool("interrupt")
    assert statuses(harness, "interrupt") == [ToolCallStatus.FAILED]
    assert harness.profile("interrupt").calls[0].error_type == "KeyboardInterrupt"
    assert await harness.call_tool("afterwards") == 1  # Lock, stack, and binding were released.
    assert not harness._nested_tasks


async def test_model_sees_a_failed_management_call_and_can_retry_it(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    directory = seed_repo(tmp_path, "retry", {"broken": "def broken("})
    repaired = HEADER + link_source("okay")

    class Model:
        retry = RetryOptions(max_retries=0)

        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            index = len(self.requests)
            self.requests.append(request)
            batches = (
                (ToolCall("bad", "repo_load", {}),),
                (ToolCall("fix", "put", {"path": str(directory / "broken.py"), "content": repaired}),),
                (ToolCall("good", "repo_load", {}),),
                (ToolCall("run", "okay", {}),),
            )
            message = (
                AssistantMessage(tool_calls=batches[index])
                if index < len(batches)
                else AssistantMessage(content="done")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    agent = await Agent.create(model, tools=[put], extensions=[jit])
    assert (await agent.run("go", config=AgentRunConfig("retry"))).content == "done"
    results = [message for message in model.requests[-1].messages if isinstance(message, ToolMessage)]
    assert [(result.tool_call_id, result.success) for result in results] == [
        ("bad", False),
        ("fix", True),
        ("good", True),
        ("run", True),
    ]
    assert "broken.py:2: def broken(" in results[0].content and "SyntaxError" in results[0].content
    assert json.loads(results[2].content)["status"] == "loaded"

    harness = jit.get_harness("retry")
    assert [(error.tool_name, error.error_type) for error in harness.tool_errors(limit=5).errors] == [
        ("repo_load", "ValueError")
    ]
    assert harness.profile("okay").calls[0].status is ToolCallStatus.COMPLETED
    jit.forget_session("retry")


async def test_model_sees_a_failed_dynamic_call_and_the_tool_can_be_repaired(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    directory = seed_repo(tmp_path, "repair", {"flaky": FAILING_LEAF.replace("link2", "flaky")})
    repaired = HEADER + link_source("flaky")

    class Model:
        retry = RetryOptions(max_retries=0)

        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            index = len(self.requests)
            self.requests.append(request)
            batches = (
                (ToolCall("first", "flaky", {}),),
                (ToolCall("repair", "put", {"path": str(directory / "flaky.py"), "content": repaired}),),
                (ToolCall("reload", "repo_load", {}),),
                (ToolCall("second", "flaky", {}),),
            )
            message = (
                AssistantMessage(tool_calls=batches[index])
                if index < len(batches)
                else AssistantMessage(content="repaired")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    agent = await Agent.create(model, tools=[put], extensions=[jit])
    assert (await agent.run("go", config=AgentRunConfig("repair"))).content == "repaired"
    results = [message for message in model.requests[-1].messages if isinstance(message, ToolMessage)]
    assert [(result.tool_call_id, result.success) for result in results] == [
        ("first", False),
        ("repair", True),
        ("reload", True),
        ("second", True),
    ]
    assert "leaf exploded" in results[0].content

    harness = jit.get_harness("repair")
    assert [(error.tool_name, error.error_type) for error in harness.tool_errors(limit=5).errors] == [
        ("flaky", "ValueError")
    ]
    assert statuses(harness, "flaky") == [ToolCallStatus.FAILED, ToolCallStatus.COMPLETED]
    jit.forget_session("repair")
