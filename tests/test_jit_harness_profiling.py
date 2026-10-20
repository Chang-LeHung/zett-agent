"""Profiling correctness, privacy, lifecycle invariants, and model accessibility."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from textwrap import dedent
from threading import Event

import pytest
from pydantic import ValidationError

from jit_support import HEADER, put_tool
from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.extensions.base import AgentExtension
from zett_agent.jit_harness import JITHarness, JITHarnessExtension, JITHarnessProfile
from zett_agent.jit_harness import profiling as profiling_module
from zett_agent.jit_harness.enums import ToolCallStatus, ToolOrigin
from zett_agent.jit_harness.profiling import MAX_MEASURED_CHARS, ToolCallProfile, _characters, _lower_bound
from zett_agent.jit_harness.runtime import _PROFILE_CALL
from zett_agent.messages import AssistantMessage, ImageBytesSource, ImageContent, TextContent, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

SOURCE = dedent('''\
    @tool(guidelines="Double an integer through a dependency.")
    async def twice(value: int) -> int:
        """Compose the base tool."""
        return await harness.call_tool("add", left=value, right=value)
''')
SYNC_SOURCE = SOURCE.replace("async def", "def").replace("await harness.call_tool", "harness.call_tool_sync")
PURE_SOURCE = SOURCE.replace('await harness.call_tool("add", left=value, right=value)', "value * 2")


@tool(guidelines="Use for exact integer addition.")
async def add(left: int, right: int) -> int:
    """Add integers."""
    return left + right


@tool(guidelines="Write one file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


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
def make_harness(tmp_path, no_residue):
    """Give each harness its own session directory and drop its modules afterwards.

    ``no_residue`` is requested so this fixture is torn down first: a loaded
    repository keeps modules in ``sys.modules`` until ``close()``.
    """
    created = []

    def factory(**options):
        harness = JITHarness(tmp_path / f"session-{len(created)}", **options)
        created.append(harness)
        return harness

    yield factory
    for harness in created:
        harness.close()


def context(*tools, extensions=()):
    return AgentRunContext(
        AgentRunConfig("profile", request_id="request"),
        AgentState(),
        {item.name: item for item in tools},
        extensions,
    )


def stats(harness, name):
    return harness.profile(name).stats[0]


def assert_invariants(harness):
    report = harness.profile(limit=1000)
    capacity = harness._profiler.call_capacity
    # Retention bounds records only; aggregates always cover every attempt.
    complete = report.total_calls == sum(summary.call_count for summary in report.stats)
    for summary in report.stats:
        assert (
            summary.call_count
            == summary.completed_count + summary.failed_count + summary.cancelled_count + summary.rejected_count
        )
        records = [call for call in report.calls if call.tool_name == summary.tool_name]
        assert len(records) <= min(summary.call_count, capacity)
        if complete:
            assert len(records) == summary.call_count
            if records:
                assert summary.average_duration_ms == pytest.approx(
                    sum(call.duration_ms for call in records) / len(records)
                )
            inputs = [call.input_chars for call in records if call.input_chars is not None]
            assert summary.average_input_chars == (pytest.approx(sum(inputs) / len(inputs)) if inputs else None)
            outputs = [call.output_chars for call in records if call.output_chars is not None]
            assert summary.average_output_chars == (pytest.approx(sum(outputs) / len(outputs)) if outputs else None)
    assert _PROFILE_CALL.get() is None


@pytest.mark.parametrize("source", [SOURCE, SYNC_SOURCE])
async def test_sync_async_nested_calls_count_once_preserve_links_ids_and_json_sizes(make_harness, source):
    harness = make_harness(tools=[add])
    put_tool(harness, source, "twice")
    ctx = context(add)
    harness.attach(ctx)
    try:
        for value in (21, -3):
            assert await ctx.tools["twice"]({"value": value}) == value * 2
        report = harness.profile()
        assert len(report.calls) == 4
        first, child, second, child2 = report.calls
        assert first.caller_tool_name is None and second.caller_tool_name is None
        assert child.caller_tool_name == "twice" and child2.caller_tool_name == "twice"
        assert len({call.call_id for call in report.calls}) == 4
        assert child.input_chars == len('{"left":21,"right":21}') and child.output_chars == 2
        assert first.input_chars == len('{"value":21}') and first.output_chars == 2
        assert stats(harness, "twice").average_output_chars == 2
        assert first.origin is ToolOrigin.PYTHON and child.origin is ToolOrigin.BASE
        assert_invariants(harness)
    finally:
        harness.detach(ctx)


async def test_monotonic_durations_and_sample_averages_are_exact(monkeypatch, make_harness):
    ticks = iter((10_000_000, 14_000_000, 20_000_000, 28_000_000))
    monkeypatch.setattr("zett_agent.jit_harness.profiling.perf_counter_ns", lambda: next(ticks))
    harness = make_harness(tools=[add])
    assert await harness.call_tool("add", left=1, right=1) == 2
    assert await harness.call_tool("add", left=10, right=10) == 20
    summary = stats(harness, "add")
    assert summary.average_duration_ms == 6
    assert summary.average_output_chars == 1.5
    assert summary.average_input_chars == (len('{"left":1,"right":1}') + len('{"left":10,"right":10}')) / 2
    assert_invariants(harness)


@pytest.mark.parametrize(
    ("value", "size"),
    [
        ("中文🙂", 3),
        ("", 0),
        (None, 4),
        ([], 2),
        ({}, 2),
        ({"中文": "🙂"}, len('{"中文":"🙂"}')),
        (TextContent("中文🙂"), 3),
        (ImageContent(ImageBytesSource(b"private-image", "image/png")), 0),
        ([TextContent("hello"), ImageContent(ImageBytesSource(b"private", "image/png")), TextContent("中文")], 7),
        (object(), None),
    ],
)
async def test_character_counts_are_unicode_not_bytes_or_tokens_and_unsupported_values_stay_unknown(
    make_harness, value, size
):
    @tool(guidelines="Return a profiling sample without changing it.")
    async def result():
        """Return different Python result representations."""
        return value

    harness = make_harness(tools=[result])
    assert await harness.call_tool("result") is value
    assert harness.profile().calls[0].input_chars == 2
    assert harness.profile().calls[0].output_chars == size
    summary = stats(harness, "result")
    assert summary.average_output_chars == size
    assert_invariants(harness)


def test_measurement_stops_at_the_budget_without_serializing_large_payloads(monkeypatch):
    """A serialization-free bound keeps dispatch cheap; the value means "at least"."""
    assert _characters({"left": 21, "right": 21}) == len('{"left":21,"right":21}')
    assert _characters({"content": "x" * 1_000_000}) == MAX_MEASURED_CHARS
    assert _characters(list(range(100_000))) == MAX_MEASURED_CHARS
    # A container wider than the remaining budget is over it without inspection.
    assert _characters({f"k{index}": 1 for index in range(20_000)}) == MAX_MEASURED_CHARS
    # Bytes are measured after JSON conversion, not as their raw length.
    assert _characters({"payload": b"0123456789"}) == len('{"payload":"0123456789"}')
    # Escaping can push the exact length over the budget after the cheap walk.
    assert _lower_bound({"a": "\n" * 4000}) < MAX_MEASURED_CHARS
    assert _characters({"a": "\n" * 4000}) == MAX_MEASURED_CHARS
    # Oversized payloads never reach the serializer, which stays untouched below.
    monkeypatch.setattr(profiling_module, "_JSON", None)
    assert _characters({"content": "x" * 1_000_000}) == MAX_MEASURED_CHARS
    assert _characters({"left": 21, "right": 21}) is None


@pytest.mark.parametrize("value", [0, -1, True, "3", None])
def test_call_capacity_must_be_a_positive_integer(tmp_path, value):
    with pytest.raises(ValueError, match="Call capacity must be a positive integer"):
        JITHarness(tmp_path, call_capacity=value)
    with pytest.raises(ValueError, match="Call capacity must be a positive integer"):
        JITHarnessExtension(tmp_path, call_capacity=value)


async def test_retained_call_records_are_bounded_while_aggregates_stay_complete(make_harness):
    harness = make_harness(tools=[add], call_capacity=3)
    for value in range(6):
        assert await harness.call_tool("add", left=value, right=value) == value * 2
    report = harness.profile()
    assert len(report.calls) == report.total_calls == 3
    assert [call.input_chars for call in report.calls] == [
        len(f'{{"left":{value},"right":{value}}}') for value in (3, 4, 5)
    ]
    assert report.next_offset is None
    # Lifetime aggregates survive eviction, exactly like the error cache.
    assert stats(harness, "add").call_count == 6
    assert stats(harness, "add").completed_count == 6
    page = harness.profile(offset=0, limit=1)
    assert page.calls == report.calls[:1] and page.next_offset == 1
    assert_invariants(harness)


async def test_unknown_input_size_and_secret_payloads_do_not_break_or_leak_into_profile(make_harness):
    @tool(guidelines="Accept arbitrary local objects.")
    async def local(value: object):
        """Return a private string."""
        return "TOP_SECRET_PAYLOAD"

    harness = make_harness(tools=[local])
    await harness.call_tool("local", value=object())
    await harness.call_tool("local", value="PASSWORD_NEVER_LOG")
    report = harness.profile()
    assert report.calls[0].input_chars is None
    assert stats(harness, "local").average_input_chars == len('{"value":"PASSWORD_NEVER_LOG"}')
    exported = report.model_dump_json()
    assert "TOP_SECRET_PAYLOAD" not in exported and "PASSWORD_NEVER_LOG" not in exported
    assert "arguments" not in exported and "result" not in exported
    assert JITHarnessProfile.model_validate_json(exported) == report
    assert_invariants(harness)


async def test_failures_rejections_and_validation_errors_have_separate_counts_and_cleanup(make_harness):
    harness = make_harness(tools=[add])
    put_tool(harness, SOURCE, "twice")
    with pytest.raises(ValidationError):
        await harness.call_tool("twice", value="bad")
    harness.unload_tool("add")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("twice", value=1)
    assert stats(harness, "twice").failed_count == 2
    assert stats(harness, "add").rejected_count == 1
    record = harness.profile("add").calls[0]
    assert record.caller_tool_name == "twice"
    assert record.error_type == "ValueError" and record.output_chars == len("Tool 'add' is unloaded")
    assert_invariants(harness)
    harness.load_tool("add")
    assert await harness.call_tool("twice", value=1) == 2
    assert_invariants(harness)


async def test_profile_keeps_update_and_unload_history(make_harness):
    harness = make_harness()
    put_tool(harness, PURE_SOURCE, "twice")
    await harness.call_tool("twice", value=1)
    changed = PURE_SOURCE.replace("value * 2", "value * 3")
    put_tool(harness, changed, "twice")
    await harness.call_tool("twice", value=1)
    harness.unload_tool("twice")
    with pytest.raises(ValueError):
        await harness.call_tool("twice", value=1)
    harness.load_tool("twice")
    await harness.call_tool("twice", value=1)
    records = harness.profile().calls
    assert [record.status for record in records] == [
        ToolCallStatus.COMPLETED,
        ToolCallStatus.COMPLETED,
        ToolCallStatus.REJECTED,
        ToolCallStatus.COMPLETED,
    ]
    assert stats(harness, "twice").call_count == 4
    assert "profiling" not in harness.list_tools()[0]
    assert_invariants(harness)


async def test_parallel_completion_order_and_cancellation_balance_totals(make_harness):
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    @tool(guidelines="Wait at a controlled profiling boundary.")
    async def wait(value: int):
        """Hold two concurrent requests until released or cancelled."""
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
        await release.wait()
        return value

    harness = make_harness(tools=[wait])
    first = asyncio.create_task(harness.call_tool("wait", value=1))
    second = asyncio.create_task(harness.call_tool("wait", value=2))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        before = harness.profile()
        assert before.calls == () and stats(harness, "wait").call_count == 0
        assert stats(harness, "wait").average_duration_ms is None
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert stats(harness, "wait").cancelled_count == 1 and stats(harness, "wait").call_count == 1
        release.set()
        assert await first == 1
        report = harness.profile()
        assert [call.status for call in report.calls] == [ToolCallStatus.COMPLETED, ToolCallStatus.CANCELLED]
        assert report.calls[1].output_chars is None
        assert_invariants(harness)
    finally:
        release.set()
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize(("offset", "limit"), [(-1, 1), (0, 0), (0, 1001)])
def test_pagination_invalid_bounds_rejected(tmp_path, offset, limit):
    with pytest.raises(ValueError, match="Profiling"):
        JITHarness(tmp_path).profile(offset=offset, limit=limit)


def test_tool_origin_is_an_enum_with_stable_json_values():
    assert [member.value for member in ToolOrigin] == ["base", "python", "management", "unknown"]
    assert ToolCallProfile.model_fields["origin"].annotation is ToolOrigin
    assert ToolCallProfile.model_fields["status"].annotation is ToolCallStatus
    # String comparison and JSON export keep the pre-enum contract.
    assert ToolOrigin.BASE == "base"
    assert str(ToolOrigin.PYTHON) == "python"
    assert json.dumps({"origin": ToolOrigin.MANAGEMENT}) == '{"origin": "management"}'
    with pytest.raises(ValidationError):
        ToolCallProfile(
            call_id="call",
            tool_name="tool",
            origin="not-a-registry",
            duration_ms=0,
            status="completed",
        )


def test_tool_call_status_is_an_enum_with_stable_json_values():
    assert [member.value for member in ToolCallStatus] == ["completed", "failed", "cancelled", "rejected"]
    assert ToolCallProfile.model_fields["status"].annotation is ToolCallStatus
    assert ToolCallStatus.FAILED == "failed"
    assert str(ToolCallStatus.REJECTED) == "rejected"
    assert json.dumps({"status": ToolCallStatus.CANCELLED}) == '{"status": "cancelled"}'
    with pytest.raises(ValidationError):
        ToolCallProfile(
            call_id="call",
            tool_name="tool",
            origin=ToolOrigin.BASE,
            duration_ms=0,
            status="succeeded",
        )


async def test_tool_call_status_classifies_every_dispatch_outcome(make_harness):
    entered, release = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Produce each profiling outcome for a controlled test.")
    async def outcome(kind: str) -> int:
        """Return, fail, or wait depending on the requested outcome."""
        if kind == "failed":
            raise ValueError("handler failed")
        if kind == "cancelled":
            entered.set()
            await release.wait()
        return 7

    harness = make_harness(tools=[outcome], max_call_depth=1)
    assert await harness.call_tool("outcome", kind="completed") == 7
    with pytest.raises(ValueError, match="handler failed"):
        await harness.call_tool("outcome", kind="failed")
    with pytest.raises(ValidationError):
        await harness.call_tool("outcome", wrong="argument")
    put_tool(
        harness,
        '@tool(guidelines="Compose a rejected dependency.")\nasync def nested() -> int:\n'
        '    """Call another tool past the depth limit."""\n'
        '    return await harness.call_tool("outcome", kind="completed")\n',
        "nested",
    )
    with pytest.raises(RuntimeError, match="depth exceeded"):
        await harness.call_tool("nested")
    harness.unload_tool("outcome")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("outcome", kind="completed")
    harness.load_tool("outcome")

    running = asyncio.create_task(harness.call_tool("outcome", kind="cancelled"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    assert [call.status for call in harness.profile("outcome").calls] == [
        ToolCallStatus.COMPLETED,
        ToolCallStatus.FAILED,
        ToolCallStatus.FAILED,
        ToolCallStatus.REJECTED,
        ToolCallStatus.REJECTED,
        ToolCallStatus.CANCELLED,
    ]
    # The depth-limit error propagates out of `nested`, so its own record is failed.
    assert harness.profile("nested").calls[0].status is ToolCallStatus.FAILED
    summary = stats(harness, "outcome")
    assert (summary.completed_count, summary.failed_count) == (1, 2)
    assert (summary.cancelled_count, summary.rejected_count) == (1, 2)
    assert summary.call_count == 6
    assert {call["status"] for call in harness.profile().model_dump(mode="json")["calls"]} == {
        "completed",
        "failed",
        "rejected",
        "cancelled",
    }
    assert_invariants(harness)


async def test_tool_origin_classifies_base_python_and_management_dispatch(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    ctx = context(add)
    await jit.on_tool(ctx)
    try:
        harness = jit.get_harness("profile")
        assert await ctx.tools["add"]({"left": 1, "right": 1}) == 2
        put_tool(harness, PURE_SOURCE, "twice")
        twice = ctx.tools["twice"]
        assert await twice({"value": 2}) == 4
        await ctx.tools["repo_load"]({})
        report = harness.profile(limit=100)
        assert {call.tool_name: call.origin for call in report.calls} == {
            "add": ToolOrigin.BASE,
            "twice": ToolOrigin.PYTHON,
            "repo_load": ToolOrigin.MANAGEMENT,
        }
        assert all(isinstance(call.origin, ToolOrigin) for call in report.calls)
        assert {call["origin"] for call in report.model_dump(mode="json")["calls"]} == {
            "base",
            "python",
            "management",
        }
        # Unloading is a rejected call on a dynamic tool, not a new origin.
        harness.unload_tool("twice")
        with pytest.raises(ValueError, match="unloaded"):
            await twice({"value": 2})
        rejected = harness.profile("twice").calls[-1]
        assert rejected.origin is ToolOrigin.PYTHON and rejected.status == "rejected"
        assert_invariants(harness)
    finally:
        jit._request_finished(ctx)
        jit.forget_session("profile")


async def test_describe_reports_a_compact_catalog_and_profile_paginates_losslessly(make_harness):
    harness = make_harness(tools=[add])
    for value in range(7):
        await harness.call_tool("add", left=value, right=1)
    info = harness.describe("add")
    assert "profiling" not in info and "calls" not in info and "source" not in info
    assert harness.profile_stats("add").call_count == 7
    assert set(info) == {
        "name",
        "origin",
        "status",
        "description",
    }
    offset = 0
    seen = []
    while True:
        page = harness.profile("add", offset=offset, limit=2)
        assert page.stats[0].call_count == 7
        seen.extend(call.call_id for call in page.calls)
        if page.next_offset is None:
            break
        offset = page.next_offset
    assert seen == [call.call_id for call in harness.profile("add", limit=100).calls]
    assert harness.profile(offset=99).calls == () and harness.profile(offset=99).next_offset is None
    info["description"] = "changed"
    assert stats(harness, "add").call_count == 7
    assert_invariants(harness)


async def test_model_creates_a_composed_tool_in_its_repository_and_second_session_starts_fresh(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    target = jit.session_directory("one") / "kit"
    write = ToolCall("write", "put", {"path": str(target / "twice.py"), "content": HEADER + SOURCE})
    observed = []

    class Model:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            index = len(observed)
            if index == 0:
                message = AssistantMessage(tool_calls=(ToolCall("base", "add", {"left": 21, "right": 21}),))
            elif index == 1:
                message = AssistantMessage(tool_calls=(ToolCall("create", "repo_create", {"repo": "kit"}),))
            elif index == 2:
                message = AssistantMessage(tool_calls=(write,))
            elif index == 3:
                message = AssistantMessage(tool_calls=(ToolCall("load", "repo_load", {}),))
            elif index == 4:
                message = AssistantMessage(tool_calls=(ToolCall("outer", "twice", {"value": 21}),))
            else:
                message = AssistantMessage(content="done")
            observed.append(request)
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(Model(), tools=[add, put], extensions=[jit])
    await agent.run("profile then compose", config=AgentRunConfig("one", request_id="r1"))
    harness = jit.get_harness("one")
    assert stats(harness, "add").call_count == 2
    assert stats(harness, "twice").call_count == 1
    records = harness.profile().calls
    assert next(call for call in records if call.tool_name == "twice").call_id == "outer"
    child = next(call for call in records if call.caller_tool_name == "twice")
    assert child.tool_name == "add"
    assert len([message for message in observed[-1].messages if isinstance(message, ToolMessage)]) == 5
    await agent.run("new session", config=AgentRunConfig("two"))
    assert jit.get_harness("two").profile().calls == ()
    assert stats(harness, "add").call_count == 2
    assert_invariants(harness)
    jit.forget_session("one")
    jit.forget_session("two")


async def test_middleware_short_circuit_not_counted_and_raw_output_size_excludes_result_hooks(make_harness):
    class Hooks(AgentExtension):
        skip = True

        async def on_tool_call(self, context, call, call_next):
            if self.skip and call.name == "add":
                return "blocked"
            return await call_next()

        async def after_tool(self, context, call, result, error):
            if call.name == "add":
                result.content = "transformed output"

    hooks = Hooks()
    harness = make_harness(tools=[add])
    put_tool(harness, SOURCE, "twice")
    ctx = context(add, extensions=(hooks,))
    harness.attach(ctx)
    try:
        assert await ctx.tools["twice"]({"value": 1}) == "transformed output"
        assert stats(harness, "add").call_count == 0
        hooks.skip = False
        assert await ctx.tools["twice"]({"value": 1}) == "transformed output"
        assert harness.profile("add").calls[0].output_chars == 1
        assert_invariants(harness)
    finally:
        harness.detach(ctx)


def test_failed_character_measurements_never_raise():
    assert _characters({"value": object()}) is None
    # Match Pydantic's JSON-compatible normalization (NaN becomes null).
    assert _characters({"value": float("nan")}) == len('{"value":null}')


async def test_sync_worker_cancellation_finishes_profile_once_and_never_leaks_into_next_request(make_harness):
    entered, release, finished = Event(), Event(), Event()
    errors = []
    harness = make_harness()

    @tool(guidelines="Wait for a controlled synchronous cancellation.")
    def blocked() -> None:
        """Do not let late worker completion append a second profile."""
        entered.set()
        release.wait(3)
        try:
            harness.call_tool_sync("blocked")
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            finished.set()

    first = context(blocked)
    harness.attach(first)
    old = first.tools["blocked"]
    running = asyncio.create_task(old({}))
    second = context(blocked)
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert stats(harness, "blocked").cancelled_count == 1
        assert len(harness.profile("blocked").calls) == 1
        harness.detach(first)
        harness.attach(second)
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        assert errors and "closed" in errors[0]
        before = harness.profile()
        with pytest.raises(RuntimeError, match="closed"):
            await old({})
        assert harness.profile() == before
        assert stats(harness, "blocked").call_count == 1
        assert_invariants(harness)
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        harness.detach(first)
        harness.detach(second)


async def test_inflight_update_and_unload_error_is_execution_failure_not_rejection(make_harness):
    entered, release = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Hold one old implementation in flight.")
    async def gate():
        """Let newer calls dispatch before this one finishes."""
        entered.set()
        await release.wait()

    harness = make_harness(tools=[gate])
    old = PURE_SOURCE.replace("return value * 2", 'await harness.call_tool("gate")\n    return value * 2')
    put_tool(harness, old, "twice")
    running = asyncio.create_task(harness.call_tool("twice", value=1))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        put_tool(harness, PURE_SOURCE, "twice")
        assert await harness.call_tool("twice", value=2) == 4
        harness.unload_tool("twice")
        release.set()
        assert await running == 2
        calls = harness.profile("twice").calls
        assert len(calls) == 2
        harness.load_tool("twice")
        failed = PURE_SOURCE.replace(
            "return value * 2", 'harness.unload_tool("twice")\n    raise ValueError("handler failed")'
        )
        put_tool(harness, failed, "twice")
        with pytest.raises(ValueError, match="handler failed"):
            await harness.call_tool("twice", value=1)
        assert harness.profile("twice").calls[-1].status == "failed"
        assert_invariants(harness)
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_missing_policy_removed_and_depth_rejections_are_tracked_and_bad_source_is_not(make_harness):
    harness = make_harness(tools=[add], max_call_depth=1)
    put_tool(harness, SOURCE, "twice")
    with pytest.raises(RuntimeError, match="depth"):
        await harness.call_tool("twice", value=1)
    assert stats(harness, "add").rejected_count == 1
    restored = make_harness()
    put_tool(restored, SOURCE, "twice")
    # A finished request leaves 'add' a known base name with no bound implementation,
    # so the composed call is rejected as missing instead of unknown.
    previous = context(add)
    restored.attach(previous)
    restored.detach(previous)
    with pytest.raises(ValueError, match="Missing base"):
        await restored.call_tool("twice", value=1)
    assert stats(restored, "add").rejected_count == 1
    ctx = context(add)
    harness.max_call_depth = 3
    harness.attach(ctx)
    try:
        ctx.tools.pop("add")
        with pytest.raises(ValueError, match="not registered"):
            await ctx.tools["twice"]({"value": 1})
        assert stats(harness, "add").rejected_count == 2
        with pytest.raises(ValueError, match="SyntaxError"):
            put_tool(harness, "invalid !", "twice")
        before = restored.profile()
        with pytest.raises(ValueError, match="Unknown"):
            await restored.call_tool("unknown")
        assert restored.profile() == before
        assert_invariants(harness)
        assert_invariants(restored)
    finally:
        harness.detach(ctx)


async def test_handler_exception_with_broken_str_does_not_break_profiling(make_harness):
    class Broken(ValueError):
        def __str__(self):
            raise RuntimeError("cannot format")

    error = Broken()

    @tool(guidelines="Exercise telemetry's error measurement fallback.")
    async def fail():
        """Propagate the original error unchanged."""
        raise error

    harness = make_harness(tools=[fail])
    with pytest.raises(Broken) as raised:
        await harness.call_tool("fail")
    assert raised.value is error
    record = harness.profile().calls[0]
    assert record.status == "failed" and record.output_chars is None and record.error_type == "Broken"
    assert_invariants(harness)
