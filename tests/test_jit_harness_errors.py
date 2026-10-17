"""LRU diagnostics stay bounded, session-local, read-only, and useful to the model."""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from jit_support import HEADER, put_tool, seed_repo, write_module
from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.jit_harness import JITHarness, JITHarnessExtension, ToolCallStatus, ToolErrorRecord, ToolErrorReport
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

SOURCE = dedent('''\
    @tool(guidelines="Exercise repair after a controlled failure.")
    async def value() -> int:
        """Return an integer or fail."""
        raise ValueError("missing normalization")
''')


@tool(guidelines="Raise requested diagnostics without external side effects.")
async def fail(message: str, kind: str = "value") -> None:
    """Raise two exception types to test deduplication boundaries."""
    raise (ValueError(message) if kind == "value" else TypeError(message))


async def fail_with(harness, message, name="fail", kind="value"):
    with pytest.raises((ValueError, TypeError)):
        await harness.call_tool(name, message=message, kind=kind)


@tool(guidelines="Write one file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


def context(session="errors"):
    return AgentRunContext(AgentRunConfig(session, request_id=f"{session}-request"), AgentState(), {})


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


async def test_lru_deduplicates_refreshes_on_reoccurrence_and_evicts_least_recent_error(tmp_path):
    harness = JITHarness(tmp_path, tools=[fail], error_capacity=2)
    await fail_with(harness, "A")
    first = harness.tool_errors().errors[0]
    await fail_with(harness, "B")
    assert [error.message for error in harness.tool_errors().errors] == ["B", "A"]
    await fail_with(harness, "A")
    refreshed = harness.tool_errors().errors[0]
    assert refreshed.occurrences == 2 and refreshed.message == first.message
    assert refreshed.last_seen_at >= first.last_seen_at
    assert refreshed.last_call_id != first.last_call_id
    await fail_with(harness, "C")
    assert [error.message for error in harness.tool_errors().errors] == ["C", "A"]
    await fail_with(harness, "B")  # Evicted errors start a new residence.
    report = harness.tool_errors()
    assert [error.message for error in report.errors] == ["B", "C"]
    assert report.errors[0].occurrences == 1
    assert report.capacity == report.retained_errors == report.total_matches == 2
    assert harness.profile("fail").stats[0].failed_count == 5
    assert len(harness.profile().calls) == 5
    assert all("message" not in call.model_dump() for call in harness.profile().calls)


def test_error_records_only_accept_the_two_failure_statuses_as_an_enum():
    payload = {
        "tool_name": "tool",
        "error_type": "ValueError",
        "message": "boom",
        "message_truncated": False,
        "occurrences": 1,
        "last_seen_at": "2026-01-01T00:00:00Z",
        "last_call_id": "call",
    }
    failed = ToolErrorRecord.model_validate({**payload, "status": "failed"})
    assert failed.status is ToolCallStatus.FAILED
    assert ToolErrorRecord.model_validate({**payload, "status": "rejected"}).status is ToolCallStatus.REJECTED
    assert failed.model_dump(mode="json")["status"] == "failed"
    for status in ("completed", "cancelled"):
        with pytest.raises(ValidationError, match="failed or rejected"):
            ToolErrorRecord.model_validate({**payload, "status": status})
    with pytest.raises(ValidationError, match="Input should be"):
        ToolErrorRecord.model_validate({**payload, "status": "unknown"})


async def test_reads_never_refresh_lru_and_same_text_does_not_merge_tools_or_error_types(tmp_path):
    other = replace(fail, name="other")
    harness = JITHarness(tmp_path, tools=[fail, other], error_capacity=3)
    await fail_with(harness, "same")
    await fail_with(harness, "same", name="other")
    await fail_with(harness, "same", kind="type")
    before = harness.tool_errors()
    assert len({(error.tool_name, error.error_type, error.message) for error in before.errors}) == 3
    assert len(harness.tool_errors("fail").errors) == 2
    assert len(harness.tool_errors("other").errors) == 1
    harness.tool_errors("fail", offset=1, limit=1)  # Read oldest repeatedly.
    harness.tool_errors("fail", offset=1, limit=1)
    assert harness.tool_errors() == before
    await fail_with(harness, "new")
    assert not any(
        error.tool_name == "fail" and error.error_type == "ValueError" and error.message == "same"
        for error in harness.tool_errors().errors
    )


async def test_messages_are_character_bounded_and_full_text_not_prefix_controls_deduplication(tmp_path):
    harness = JITHarness(tmp_path, tools=[fail], error_capacity=2, error_message_chars=3)
    await fail_with(harness, "中文🙂A")
    await fail_with(harness, "中文🙂B")
    errors = harness.tool_errors().errors
    # Truncated prefixes are identical; the full text still separates the records.
    assert len(errors) == 2
    assert all(error.message == "中文🙂" and error.message_truncated for error in errors)
    await fail_with(harness, "中文🙂A")
    assert harness.tool_errors().errors[0].occurrences == 2
    assert sum(len(error.message) for error in harness.tool_errors().errors) <= 6


@pytest.mark.parametrize("message", ["", "中文", "x" * 4000])
async def test_empty_and_at_limit_messages_not_marked_truncated(tmp_path, message):
    harness = JITHarness(tmp_path, tools=[fail])
    await fail_with(harness, message)
    error = harness.tool_errors().errors[0]
    assert error.message == message
    assert not error.message_truncated and error.error_type == "ValueError"


@pytest.mark.parametrize("factory", [JITHarness, JITHarnessExtension])
@pytest.mark.parametrize("option", ["error_capacity", "error_message_chars"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_error_config_is_validated_at_construction_before_any_session(tmp_path, factory, option, value):
    with pytest.raises(ValueError, match="positive integers"):
        factory(tmp_path, **{option: value})


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, 1001), (True, 1), (0, True), (0.5, 1), (0, 1.5)],
)
def test_error_pagination_rejects_invalid_bounds(tmp_path, offset, limit):
    with pytest.raises(ValueError, match="Error offset"):
        JITHarness(tmp_path).tool_errors(offset=offset, limit=limit)


async def test_filter_pagination_immutable_export_and_empty_report_are_read_only(tmp_path):
    harness = JITHarness(tmp_path, tools=[fail], error_capacity=5)
    assert harness.tool_errors().errors == ()
    for message in ("A", "B", "C", "D", "E"):
        await fail_with(harness, message)
    original = harness.tool_errors()
    page1 = harness.tool_errors("fail", limit=2)
    page2 = harness.tool_errors("fail", offset=page1.next_offset, limit=2)
    page3 = harness.tool_errors("fail", offset=page2.next_offset, limit=2)
    assert [error.message for error in (*page1.errors, *page2.errors, *page3.errors)] == ["E", "D", "C", "B", "A"]
    assert page3.next_offset is None and page3.total_matches == 5
    assert harness.tool_errors(offset=99).errors == () and harness.tool_errors(offset=99).next_offset is None
    assert harness.tool_errors("never-called").total_matches == 0
    with pytest.raises(ValidationError):
        original.errors[0].message = "changed"
    exported = original.model_dump(mode="json")
    exported["errors"][0]["message"] = "changed"
    assert harness.tool_errors() == original
    assert ToolErrorReport.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize("synchronous", [False, True])
async def test_nested_failure_preserves_context_and_parent_can_catch_then_continue(tmp_path, synchronous):
    header = "def" if synchronous else "async def"
    invoke = "harness.call_tool_sync" if synchronous else "await harness.call_tool"
    source = f'''\
@tool(guidelines="Handle an expected dependency failure.")
{header} parent() -> str:
    """Catch the child exception without losing diagnostic context."""
    try:
        {invoke}("fail", message="child failure")
    except ValueError:
        return "recovered"
    raise AssertionError("dependency should fail")
'''
    seed_repo(tmp_path, "errors", {"parent": source})
    jit = JITHarnessExtension(tmp_path, error_capacity=1)
    ctx = context()
    ctx.register_tool(fail)
    await jit.on_tool(ctx)
    harness = jit.get_harness("errors")
    try:
        assert await ctx.tools["parent"]({}) == "recovered"
        error = harness.tool_errors().errors[0]
        assert error.tool_name == "fail" and error.caller_tool_name == "parent"
        assert error.status == "failed" and error.last_call_id
        assert harness.profile("parent").stats[0].completed_count == 1
        assert not harness._nested_tasks
    finally:
        jit._request_finished(ctx)
        jit.forget_session("errors")


async def test_errors_survive_a_reload_and_unload_but_never_reach_the_profile(tmp_path):
    harness = JITHarness(tmp_path, error_capacity=2)
    put_tool(harness, SOURCE, "value")
    with pytest.raises(ValueError):
        await harness.call_tool("value")
    original = harness.tool_errors().errors[0]
    changed = SOURCE.replace('raise ValueError("missing normalization")', "return 42")
    put_tool(harness, changed, "value")
    assert await harness.call_tool("value") == 42
    assert harness.tool_errors().errors == (original,)  # Successful repair does not refresh or delete evidence.
    harness.unload_tool("value")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("value")
    error = harness.tool_errors().errors[0]
    assert error.status == "rejected" and error.last_call_id != original.last_call_id
    assert all(
        error.message is None or error.message not in harness.profile().model_dump_json()
        for error in harness.tool_errors().errors
    )
    harness.close()


async def test_one_hundred_distinct_errors_stay_within_global_capacity_and_character_limit(tmp_path):
    harness = JITHarness(tmp_path, tools=[fail], error_capacity=4, error_message_chars=8)
    for index in range(100):
        await fail_with(harness, f"failure-{index:03}")
        assert harness.tool_errors().retained_errors <= 4
    report = harness.tool_errors()
    assert len(report.errors) == 4 and sum(len(error.message) for error in report.errors) <= 32
    newest_first = [call.call_id for call in reversed(harness.profile("fail").calls)]
    assert [error.last_call_id for error in report.errors] == newest_first[:4]
    assert harness.profile("fail").stats[0].failed_count == 100


async def test_parallel_duplicates_are_atomic_and_completion_recency_controls_lru(tmp_path):
    harness = JITHarness(tmp_path, tools=[fail], error_capacity=2)

    async def attempt():
        await fail_with(harness, "duplicate")

    await asyncio.gather(*(attempt() for _ in range(30)))
    assert harness.tool_errors().errors[0].occurrences == 30
    assert harness.tool_errors().retained_errors == 1
    entered, release = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Finish an older invocation later.")
    async def delayed() -> None:
        """Recency follows actual failure, not dispatch sequence."""
        entered.set()
        await release.wait()
        raise ValueError("late completion")

    second = JITHarness(tmp_path, tools=[fail, delayed], error_capacity=2)
    running = asyncio.create_task(second.call_tool("delayed"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await fail_with(second, "early completion")
        release.set()
        with pytest.raises(ValueError):
            await running
        errors = second.tool_errors().errors
        assert [error.message for error in errors] == ["late completion", "early completion"]
        assert errors[0].last_seen_at > errors[1].last_seen_at
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def test_cancelled_dispatches_and_broken_message_formatting_remain_safe(tmp_path):
    entered = asyncio.Event()

    @tool(guidelines="Wait for cancellation.")
    async def wait() -> None:
        """Cancellation is not a cached tool error."""
        entered.set()
        await asyncio.Event().wait()

    harness = JITHarness(tmp_path, tools=[wait])
    running = asyncio.create_task(harness.call_tool("wait"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert harness.tool_errors().errors == ()
        assert harness.profile("wait").stats[0].cancelled_count == 1
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    class Broken(ValueError):
        def __str__(self):
            raise RuntimeError("cannot format")

    @tool(guidelines="Fail with an unformattable error.")
    async def broken() -> None:
        """Do not let diagnostics mask this exception."""
        raise Broken()

    harness = JITHarness(tmp_path, tools=[broken])
    for _ in range(2):
        with pytest.raises(Broken):
            await harness.call_tool("broken")
    error = harness.tool_errors().errors[0]
    assert error.message is None and not error.message_truncated
    assert error.occurrences == 2 and error.error_type == "Broken"


async def test_cache_does_not_retain_exception_objects_or_traceback_frames(tmp_path):
    references = []

    class TemporaryError(Exception):
        pass

    @tool(guidelines="Raise a fresh temporary exception.")
    async def transient() -> None:
        """Only text and metadata should survive."""
        error = TemporaryError("temporary")
        references.append(weakref.ref(error))
        raise error

    harness = JITHarness(tmp_path, tools=[transient])

    async def invoke():
        try:
            await harness.call_tool("transient")
        except TemporaryError:
            pass

    await invoke()
    gc.collect()
    assert references[0]() is None
    assert harness.tool_errors().errors[0].message == "temporary"


async def test_model_diagnoses_a_failure_and_repairs_the_repository_without_autonomous_retry(tmp_path):
    jit = JITHarnessExtension(tmp_path, error_capacity=3, error_message_chars=100)
    directory = seed_repo(tmp_path, "one", {"value": SOURCE})
    steps = []
    repaired = SOURCE.replace('raise ValueError("missing normalization")', "return 42")

    class Model:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            index = len(steps)
            offered = {definition.name for definition in request.tools}
            assert {"repo_create", "repo_load", "repo_unload", "put", "value"} <= offered
            assert "inspect_tool_errors" not in offered and "update_tool" not in offered
            assert "repo_load" in request.messages[0].content
            if index == 0:
                message = AssistantMessage(tool_calls=(ToolCall("bad", "value", {}),))
            elif index == 1:
                assert isinstance(request.messages[-1], ToolMessage) and not request.messages[-1].success
                assert request.messages[-1].content == "missing normalization"
                repair = {"path": str(directory / "value.py"), "content": HEADER + repaired}
                message = AssistantMessage(tool_calls=(ToolCall("repair", "put", repair),))
            elif index == 2:
                message = AssistantMessage(tool_calls=(ToolCall("load", "repo_load", {}),))
            elif index == 3:
                assert json.loads(request.messages[-1].content)["status"] == "loaded"
                message = AssistantMessage(tool_calls=(ToolCall("good", "value", {}),))
            else:
                assert request.messages[-1].content == "42"
                message = AssistantMessage(content="fixed")
            steps.append(request)
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(Model(), tools=[put], extensions=[jit])
    assert (await agent.run("repair the failed tool", config=AgentRunConfig("one"))).content == "fixed"
    harness = jit.get_harness("one")
    assert harness.profile("value").stats[0].call_count == 2
    assert harness.profile("value").stats[0].failed_count == 1  # One failure, no hidden retry.
    assert harness.tool_errors().errors[0].occurrences == 1
    assert (directory / "value.py").read_text() == HEADER + repaired
    jit.forget_session("one")
    ctx = context("two")
    await jit.on_tool(ctx)
    try:
        assert jit.get_harness("two").tool_errors().errors == ()
        assert jit.get_harness("two").repo is None  # A new session starts without a repository.
    finally:
        jit._request_finished(ctx)
        jit.forget_session("two")


async def test_management_tool_load_failure_is_visible_in_recent_errors(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    ctx = context()
    await jit.on_tool(ctx)
    try:
        harness = jit.get_harness("errors")
        write_module(harness, "bad", "invalid python !\n", header=False)
        with pytest.raises(ValueError, match="SyntaxError"):
            await ctx.tools["repo_load"]({})
        report = harness.tool_errors()
        assert report.errors[0].tool_name == "repo_load"
        assert report.errors[0].error_type == "ValueError"
        assert harness.list_tools() == []  # The failed load registered nothing.
    finally:
        jit._request_finished(ctx)
        jit.forget_session("errors")


async def test_management_wrappers_cannot_act_on_a_new_request_through_old_references(tmp_path):
    jit = JITHarnessExtension(tmp_path)
    first = context()
    await jit.on_tool(first)
    harness = jit.get_harness("errors")
    old = first.tools["repo_load"]
    write_module(harness, "bad", "invalid !\n", header=False)
    with pytest.raises(ValueError, match="SyntaxError"):
        await old({})
    before = harness.tool_errors()
    jit._request_finished(first)
    with pytest.raises(RuntimeError, match="closed"):
        await old({})
    second = context()
    await jit.on_tool(second)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            await old({})
        assert harness.tool_errors() == before
        with pytest.raises(ValueError, match="SyntaxError"):
            await second.tools["repo_load"]({})
        current = second.tools.pop("repo_load")
        with pytest.raises(ValueError, match="not registered"):
            await current({})
        assert harness.tool_errors().errors[0].tool_name == "repo_load"
    finally:
        jit._request_finished(second)
        jit.forget_session("errors")


async def test_reload_failure_retains_old_implementation_and_latest_diagnostic(tmp_path):
    working = SOURCE.replace('raise ValueError("missing normalization")', "return 42")
    seed_repo(tmp_path, "errors", {"value": working})
    jit = JITHarnessExtension(tmp_path, error_capacity=2)
    ctx = context()
    await jit.on_tool(ctx)
    harness = jit.get_harness("errors")
    try:
        write_module(harness, "value", "invalid !\n", header=False)
        with pytest.raises(ValueError, match="SyntaxError"):
            await ctx.tools["repo_load"]({})
        assert await ctx.tools["value"]({}) == 42
        report = harness.tool_errors("repo_load", limit=1)
        assert report.errors[0].error_type == "ValueError"
        assert "invalid" in report.errors[0].message
        assert harness.tool_errors("value").errors == ()
    finally:
        jit._request_finished(ctx)
        jit.forget_session("errors")
