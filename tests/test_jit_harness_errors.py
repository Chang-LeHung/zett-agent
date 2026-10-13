"""LRU diagnostics stay bounded, session-local, read-only, and useful to the model."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from dataclasses import replace
from textwrap import dedent

import pytest
from pydantic import ValidationError

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


def context(session="errors"):
    return AgentRunContext(AgentRunConfig(session, request_id=f"{session}-request"), AgentState(), {})


async def test_lru_deduplicates_refreshes_on_reoccurrence_and_evicts_least_recent_error():
    harness = JITHarness(tools=[fail], error_capacity=2)
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


async def test_reads_never_refresh_lru_and_same_text_does_not_merge_tools_or_error_types():
    other = replace(fail, name="other")
    harness = JITHarness(tools=[fail, other], error_capacity=3)
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


async def test_messages_are_character_bounded_and_full_text_not_prefix_controls_deduplication():
    harness = JITHarness(tools=[fail], error_capacity=2, error_message_chars=3)
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
async def test_empty_and_at_limit_messages_not_marked_truncated(message):
    harness = JITHarness(tools=[fail])
    await fail_with(harness, message)
    error = harness.tool_errors().errors[0]
    assert error.message == message
    assert not error.message_truncated and error.error_type == "ValueError"


@pytest.mark.parametrize("factory", [JITHarness, JITHarnessExtension])
@pytest.mark.parametrize("option", ["error_capacity", "error_message_chars"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_error_config_is_validated_before_source_activation_or_first_session(factory, option, value):
    with pytest.raises(ValueError, match="positive integers"):
        factory(**{option: value})


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, 1001), (True, 1), (0, True), (0.5, 1), (0, 1.5)],
)
def test_error_pagination_rejects_invalid_bounds(offset, limit):
    with pytest.raises(ValueError, match="Error offset"):
        JITHarness().tool_errors(offset=offset, limit=limit)


async def test_filter_pagination_immutable_export_and_empty_report_are_read_only():
    harness = JITHarness(tools=[fail], error_capacity=5)
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
async def test_nested_failure_preserves_context_and_parent_can_catch_then_continue(synchronous):
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
    initial = JITHarness()
    initial.create_tool(source, entrypoint="parent")
    jit = JITHarnessExtension(initial.snapshot(), error_capacity=1)
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


async def test_errors_survive_updates_and_unload_but_never_serialize_with_executable_snapshot():
    harness = JITHarness(error_capacity=2)
    harness.create_tool(SOURCE, entrypoint="value")
    with pytest.raises(ValueError):
        await harness.call_tool("value")
    original = harness.tool_errors().errors[0]
    changed = SOURCE.replace('raise ValueError("missing normalization")', "return 42")
    harness.update_tool("value", changed, entrypoint="value")
    assert await harness.call_tool("value") == 42
    assert harness.tool_errors().errors == (original,)  # Successful repair does not refresh or delete evidence.
    harness.unload_tool("value")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("value")
    error = harness.tool_errors().errors[0]
    assert error.status == "rejected" and error.last_call_id != original.last_call_id
    snapshot = harness.snapshot()
    restored = JITHarness(snapshot, error_capacity=1)
    assert restored.tool_errors().errors == ()
    assert "occurrences" not in snapshot.to_json()
    assert all(
        error.message is None or error.message not in harness.profile().model_dump_json()
        for error in harness.tool_errors().errors
    )


async def test_one_hundred_distinct_errors_stay_within_global_capacity_and_character_limit():
    harness = JITHarness(tools=[fail], error_capacity=4, error_message_chars=8)
    for index in range(100):
        await fail_with(harness, f"failure-{index:03}")
        assert harness.tool_errors().retained_errors <= 4
    report = harness.tool_errors()
    assert len(report.errors) == 4 and sum(len(error.message) for error in report.errors) <= 32
    newest_first = [call.call_id for call in reversed(harness.profile("fail").calls)]
    assert [error.last_call_id for error in report.errors] == newest_first[:4]
    assert harness.profile("fail").stats[0].failed_count == 100


async def test_parallel_duplicates_are_atomic_and_completion_recency_controls_lru():
    harness = JITHarness(tools=[fail], error_capacity=2)

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

    second = JITHarness(tools=[fail, delayed], error_capacity=2)
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


async def test_cancelled_dispatches_and_broken_message_formatting_remain_safe():
    entered = asyncio.Event()

    @tool(guidelines="Wait for cancellation.")
    async def wait() -> None:
        """Cancellation is not a cached tool error."""
        entered.set()
        await asyncio.Event().wait()

    harness = JITHarness(tools=[wait])
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

    harness = JITHarness(tools=[broken])
    for _ in range(2):
        with pytest.raises(Broken):
            await harness.call_tool("broken")
    error = harness.tool_errors().errors[0]
    assert error.message is None and not error.message_truncated
    assert error.occurrences == 2 and error.error_type == "Broken"


async def test_cache_does_not_retain_exception_objects_or_traceback_frames():
    references = []

    class TemporaryError(Exception):
        pass

    @tool(guidelines="Raise a fresh temporary exception.")
    async def transient() -> None:
        """Only text and metadata should survive."""
        error = TemporaryError("temporary")
        references.append(weakref.ref(error))
        raise error

    harness = JITHarness(tools=[transient])

    async def invoke():
        try:
            await harness.call_tool("transient")
        except TemporaryError:
            pass

    await invoke()
    gc.collect()
    assert references[0]() is None
    assert harness.tool_errors().errors[0].message == "temporary"


async def test_model_queries_errors_and_explicitly_repairs_source_without_autonomous_retry():
    initial = JITHarness()
    initial.create_tool(SOURCE, entrypoint="value")
    jit = JITHarnessExtension(initial.snapshot(), error_capacity=3, error_message_chars=100)
    steps = []
    repaired = SOURCE.replace('raise ValueError("missing normalization")', "return 42")

    class Model:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            index = len(steps)
            offered = {definition.name for definition in request.tools}
            assert "inspect_tool_errors" in offered
            assert "untrusted diagnostic data" in request.messages[0].content
            if index == 0:
                message = AssistantMessage(tool_calls=(ToolCall("bad", "value", {}),))
            elif index == 1:
                assert isinstance(request.messages[-1], ToolMessage) and not request.messages[-1].success
                message = AssistantMessage(tool_calls=(ToolCall("errors", "inspect_tool_errors", {"name": "value"}),))
            elif index == 2:
                report = json.loads(request.messages[-1].content)
                assert "capacity" not in report and report["total_matches"] == 1
                assert report["errors"][0]["message"] == "missing normalization"
                assert report["errors"][0]["last_call_id"] == "bad"
                assert report["errors"][0]["occurrences"] == 1
                message = AssistantMessage(
                    tool_calls=(
                        ToolCall("repair", "update_tool", {"name": "value", "source": repaired, "entrypoint": "value"}),
                    )
                )
            elif index == 3:
                message = AssistantMessage(tool_calls=(ToolCall("good", "value", {}),))
            else:
                assert request.messages[-1].content == "42"
                message = AssistantMessage(content="fixed")
            steps.append(request)
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(Model(), extensions=[jit])
    assert (await agent.run("inspect then repair", config=AgentRunConfig("one"))).content == "fixed"
    harness = jit.get_harness("one")
    assert harness.profile("value").stats[0].call_count == 2
    assert harness.tool_errors().errors[0].occurrences == 1
    assert harness.inspect_tool("value")["source"] == repaired
    ctx = context("two")
    await jit.on_tool(ctx)
    try:
        assert jit.get_harness("two").tool_errors().errors == ()
        assert jit.get_harness("two").inspect_tool("value")["source"] == SOURCE
    finally:
        jit._request_finished(ctx)


async def test_management_tool_creation_failure_is_visible_in_recent_errors():
    jit = JITHarnessExtension()
    ctx = context()
    await jit.on_tool(ctx)
    try:
        with pytest.raises(SyntaxError):
            await ctx.tools["create_tool"]({"source": "invalid python !", "entrypoint": "bad"})
        report = await ctx.tools["inspect_tool_errors"]({})
        assert report["errors"][0]["tool_name"] == "create_tool"
        assert report["errors"][0]["error_type"] == "SyntaxError"
        assert jit.get_harness("errors").snapshot().tools == ()
    finally:
        jit._request_finished(ctx)


async def test_error_query_wrappers_cannot_read_new_request_through_old_references():
    jit = JITHarnessExtension()
    first = context()
    await jit.on_tool(first)
    old = first.tools["inspect_tool_errors"]
    with pytest.raises(SyntaxError):
        await first.tools["create_tool"]({"source": "invalid !", "entrypoint": "bad"})
    before = jit.get_harness("errors").tool_errors()
    jit._request_finished(first)
    with pytest.raises(RuntimeError, match="closed"):
        await old({})
    second = context()
    await jit.on_tool(second)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            await old({})
        assert jit.get_harness("errors").tool_errors() == before
        assert (await second.tools["inspect_tool_errors"]({}))["errors"][0]["error_type"] == "SyntaxError"
        current = second.tools.pop("inspect_tool_errors")
        with pytest.raises(ValueError, match="not registered"):
            await current({})
        assert jit.get_harness("errors").tool_errors().errors[0].tool_name == "inspect_tool_errors"
    finally:
        jit._request_finished(second)


async def test_source_update_failure_retains_old_implementation_and_latest_diagnostic():
    initial = JITHarness()
    initial.create_tool(SOURCE.replace('raise ValueError("missing normalization")', "return 42"), entrypoint="value")
    jit = JITHarnessExtension(initial.snapshot(), error_capacity=2)
    ctx = context()
    await jit.on_tool(ctx)
    try:
        with pytest.raises(SyntaxError):
            await ctx.tools["update_tool"]({"name": "value", "source": "invalid !", "entrypoint": "value"})
        assert await ctx.tools["value"]({}) == 42
        report = await ctx.tools["inspect_tool_errors"]({"name": "update_tool", "offset": 0, "limit": 1})
        assert report["errors"][0]["error_type"] == "SyntaxError"
        assert "invalid" in report["errors"][0]["message"]
        assert jit.get_harness("errors").tool_errors("value").errors == ()
    finally:
        jit._request_finished(ctx)
