"""Model queries return only the requested data and stay within a JSON budget."""

from __future__ import annotations

import json
from dataclasses import replace
from textwrap import dedent

import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.jit_harness.queries import MAX_RESPONSE_CHARS, ToolQueries
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.tools.base import _bind_tool_call, _reset_tool_call, tool

SOURCE = dedent('''\
    @tool(guidelines="Use to double an integer.")
    async def twice(value: int) -> int:
        """Double an integer."""
        return value * 2
''')


@tool(guidelines="Use for exact integer addition.")
async def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def assert_small(value):
    # Match real runtime serialization, not a repr or unescaped text length.
    assert len(add.serialize_result(value)) <= MAX_RESPONSE_CHARS


async def test_queries_are_purpose_specific_and_do_not_repeat_unrequested_data():
    harness = JITHarness(tools=[add])
    harness.create_tool(SOURCE, entrypoint="twice")
    await harness.call_tool("twice", value=21)
    queries = ToolQueries(harness)
    listing = queries.list_tools()
    assert set(listing) == {"tools", "total_tools", "next_offset"}
    assert set(listing["tools"][0]) == {"name", "origin", "status", "description"}
    definition = queries.inspect_tool("twice")
    assert set(definition) == {"name", "origin", "status", "description", "entrypoint", "parameters"}
    text = queries.read_source("twice")
    assert text["content"] == SOURCE
    assert set(text) == {"name", "path", "content", "content_id", "total_chars", "next_offset"}
    assert text["path"] == "__init__.py"
    stats = queries.stats("twice")
    assert stats.call_count == 1 and stats.average_output_chars == 2
    calls = queries.calls("twice")
    assert set(calls) == {"name", "calls", "total_calls", "next_offset"}
    assert "source" not in calls and "stats" not in calls
    assert set(calls["calls"][0]) == {"call_id", "status", "duration_ms", "input_chars", "output_chars"}
    for value in (listing, definition, text, stats, calls, queries.errors()):
        assert_small(value)
    # The full application view carries aggregates, while call records live in profile().
    full = harness.inspect_tool("twice")
    assert full["source"] == SOURCE and "profiling" not in full
    assert harness.profile_stats("twice").call_count == 1
    assert harness.profile("twice").calls[0].call_id


async def test_large_catalog_is_bounded_paged_sorted_and_reading_does_not_fetch_profiling(monkeypatch):
    tools = [replace(add, name=f"base_{index:03}", description="\x01" * 10000) for index in range(110)]
    harness = JITHarness(tools=tools)
    queries = ToolQueries(harness)

    def forbidden(name):
        raise AssertionError("Discovery must not fetch measurements")

    monkeypatch.setattr(harness, "profile_stats", forbidden)
    offset = 0
    names = []
    while True:
        page = queries.list_tools(offset, limit=100)
        assert_small(page)
        assert page["tools"] and page["total_tools"] == 110
        assert len(page["tools"]) < 100  # The character budget limits entries too.
        for entry in page["tools"]:
            assert len(entry["description"]) == 200 and entry["description"].endswith("…")
            assert "profiling" not in entry
            names.append(entry["name"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert names == [f"base_{index:03}" for index in range(110)]
    assert queries.list_tools(999)["tools"] == []
    assert queries.list_tools(999)["next_offset"] is None
    assert len(harness.describe("base_000")["description"]) == 10000  # Preview did not alter state.


@pytest.mark.parametrize("padding", ["中文🙂", "\x01", '"\\\n'])
def test_source_fragments_preserve_exact_text_even_when_json_escaping_magnifies_size(padding):
    # A long comment can include control characters without executing code.
    source = SOURCE + "# " + padding.replace("\n", "\n# ") * 5000 + "\n"
    harness = JITHarness()
    harness.create_tool(source, entrypoint="twice")
    queries = ToolQueries(harness)
    offset, pieces, ids = 0, [], set()
    while True:
        page = queries.read_source("twice", offset=offset, max_chars=4000)
        assert_small(page)
        assert len(page["content"]) <= 4000
        assert page["content"] == source[offset : offset + len(page["content"])]
        pieces.append(page["content"])
        ids.add(page["content_id"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert "".join(pieces) == source and len(ids) == 1
    assert queries.read_source("twice", offset=len(source))["content"] == ""
    assert queries.read_source("twice", offset=len(source) + 1)["next_offset"] is None
    harness.update_tool("twice", source + "# change\n", entrypoint="twice")
    assert queries.read_source("twice")["content_id"] not in ids


def test_large_schema_is_explicitly_omitted_then_losslessly_readable_without_unrelated_fields():
    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": [f"{index} 中文\x01" * 4 for index in range(900)]}},
    }
    harness = JITHarness(tools=[replace(add, parameters=schema, description="long description " * 1000)])
    queries = ToolQueries(harness)
    view = queries.inspect_tool("add")
    assert_small(view)
    assert view["parameters_omitted"] is True and "parameters" not in view
    assert len(view["description"]) == 500
    assert "guidelines" not in view and "source" not in view
    offset, fragments = 0, []
    while True:
        page = queries.read_schema("add", offset, max_chars=4000)
        assert_small(page)
        fragments.append(page["content"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    content = "".join(fragments)
    assert json.loads(content) == schema
    assert view["parameters_chars"] == len(content)
    assert harness.definition("add")["parameters"] == schema


async def test_calls_page_budget_does_not_drop_records_and_optional_links_are_preserved():
    harness = JITHarness()
    harness.create_tool(SOURCE, entrypoint="twice")
    harness.create_tool(
        SOURCE.replace("twice", "parent").replace("value * 2", 'await harness.call_tool("twice", value=value)'),
        entrypoint="parent",
    )
    for value in range(70):
        await harness.call_tool("parent", value=value)
    queries = ToolQueries(harness)
    offset, seen = 0, []
    while True:
        page = queries.calls("twice", offset, limit=100)
        assert_small(page)
        assert page["calls"]
        assert page["total_calls"] == 70
        for call in page["calls"]:
            assert call["caller_tool_name"] == "parent"
            assert "tool_name" not in call and "sequence" not in call and "error_type" not in call
            seen.append(call["call_id"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert seen == [call.call_id for call in harness.profile("twice").calls]
    assert queries.calls("twice", 999)["next_offset"] is None


async def test_recent_error_preview_and_pages_bound_escaped_messages_without_mutating_lru():
    @tool(guidelines="Produce a diagnostic sample.")
    async def fail(index: int) -> None:
        """Raise long and heavily escaped error messages."""
        raise ValueError(f"failure {index}: " + "\x01" * 5000)

    harness = JITHarness(tools=[fail])
    for index in range(20):
        with pytest.raises(ValueError):
            await harness.call_tool("fail", index=index)
    before = harness.tool_errors(limit=100)
    queries = ToolQueries(harness)
    defaults = queries.errors()
    assert len(defaults["errors"]) <= 3 and defaults["next_offset"] is not None
    assert all(len(error["message"]) <= 512 and error["message_truncated"] for error in defaults["errors"])
    offset, call_ids = 0, []
    while True:
        page = queries.errors("fail", offset=offset, limit=100, max_chars=4000)
        assert_small(page)
        assert page["errors"] and "capacity" not in page and "retained_errors" not in page
        for error in page["errors"]:
            assert set(error) == {
                "tool_name",
                "error_type",
                "message",
                "occurrences",
                "last_call_id",
                "message_truncated",
            }
            assert error["message_truncated"]
            call_ids.append(error["last_call_id"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert call_ids == [error.last_call_id for error in before.errors]
    assert harness.tool_errors(limit=100) == before
    assert queries.errors("unknown")["errors"] == []
    assert queries.errors(offset=999)["next_offset"] is None


@pytest.mark.parametrize("offset", [-1, True, 0.5])
def test_negative_or_non_integer_offsets_fail_cleanly(offset):
    queries = ToolQueries(JITHarness(tools=[add]))
    for operation in (queries.list_tools, lambda **kwargs: queries.calls("add", **kwargs), queries.errors):
        with pytest.raises(ValueError, match="offset"):
            operation(offset=offset)
    with pytest.raises(ValueError, match="offset"):
        queries.read_schema("add", offset=offset)


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_invalid_page_limits_fail_cleanly(limit):
    queries = ToolQueries(JITHarness(tools=[add]))
    for operation in (queries.list_tools, lambda **kwargs: queries.calls("add", **kwargs), queries.errors):
        with pytest.raises(ValueError, match="limit"):
            operation(limit=limit)


@pytest.mark.parametrize("max_chars", [0, 4001, True, 1.5])
def test_invalid_text_limits_fail_cleanly(max_chars):
    queries = ToolQueries(JITHarness(tools=[add]))
    with pytest.raises(ValueError, match="max_chars"):
        queries.read_schema("add", max_chars=max_chars)
    with pytest.raises(ValueError, match="max_chars"):
        queries.errors(max_chars=max_chars)


def test_missing_base_bindings_and_unknown_names_return_errors_not_massive_metadata():
    harness = JITHarness(tools=[add])
    restored = JITHarness(harness.snapshot())
    queries = ToolQueries(restored)
    assert queries.inspect_tool("add")["status"] == "missing"
    assert "parameters" not in queries.inspect_tool("add")
    with pytest.raises(ValueError, match="Missing base tool schema"):
        queries.read_schema("add")
    with pytest.raises(ValueError, match="no retained"):
        queries.read_source("add")
    for operation in (queries.stats, queries.calls):
        with pytest.raises(ValueError, match="Unknown tool"):
            operation("x" * 10000)
    assert restored.snapshot() == harness.snapshot()


async def test_unbounded_call_metadata_is_rejected_instead_of_an_empty_nonadvancing_page():
    harness = JITHarness(tools=[add])
    token = _bind_tool_call("x" * 10000)
    try:
        await harness.call_tool("add", left=1, right=2)
    finally:
        _reset_tool_call(token)
    with pytest.raises(ValueError, match="Call metadata"):
        ToolQueries(harness).calls("add")


def test_long_entrypoint_is_rejected_by_definition_query_but_source_remains_readable():
    entrypoint = "a" * 10000
    source = SOURCE + f"\n{entrypoint} = twice\n"
    harness = JITHarness()
    harness.create_tool(source, entrypoint=entrypoint)
    queries = ToolQueries(harness)
    with pytest.raises(ValueError, match="definition metadata"):
        queries.inspect_tool("twice")
    assert_small(queries.read_source("twice"))
    assert harness.source("twice") == source


def test_text_page_rejects_metadata_that_leaves_no_room_for_any_content():
    with pytest.raises(ValueError, match="Text metadata"):
        ToolQueries._text_page("x" * 10000, "content", offset=0, max_chars=100)


async def test_calls_and_errors_keep_failure_identifiers_but_not_full_request_metadata():
    harness = JITHarness()
    harness.create_tool(SOURCE.replace("return value * 2", 'raise ValueError("bad input")'), entrypoint="twice")
    with pytest.raises(ValueError, match="bad input"):
        await harness.call_tool("twice", value=1)
    queries = ToolQueries(harness)
    call = queries.calls("twice")["calls"][0]
    assert call["error_type"] == "ValueError"
    error = queries.errors("twice")["errors"][0]
    assert error["last_call_id"] == call["call_id"]
    assert error["message"] == "bad input" and not error["message_truncated"]
    assert_small(queries.calls("twice"))
    assert_small(queries.errors("twice"))


async def test_error_queries_retain_none_when_exception_text_cannot_be_formatted():
    class BrokenError(ValueError):
        def __str__(self):
            raise RuntimeError("format failed")

    @tool(guidelines="Raise an error with unformattable text.")
    async def broken() -> None:
        """Expose safe diagnostic fallback."""
        raise BrokenError()

    harness = JITHarness(tools=[broken])
    with pytest.raises(BrokenError):
        await harness.call_tool("broken")
    page = ToolQueries(harness).errors()
    assert page["errors"][0]["message"] is None
    assert not page["errors"][0]["message_truncated"]
    assert page["errors"][0]["error_type"] == "BrokenError"
    assert_small(page)


async def test_error_metadata_that_alone_exceeds_budget_fails_with_a_small_error():
    @tool(guidelines="Produce a failure linked to a very long ID.")
    async def fail() -> None:
        """Fail without exposing huge IDs to the model."""
        raise ValueError("failure")

    harness = JITHarness(tools=[fail])
    token = _bind_tool_call("x" * 10000)
    try:
        with pytest.raises(ValueError):
            await harness.call_tool("fail")
    finally:
        _reset_tool_call(token)
    with pytest.raises(ValueError, match="Error metadata") as raised:
        ToolQueries(harness).errors()
    assert len(str(raised.value)) < 100


async def test_query_split_is_wired_end_to_end_for_model_driven_read_then_update():
    template = JITHarness()
    template.create_tool(SOURCE, entrypoint="twice")
    batches = [
        ("list_tools", {}),
        ("inspect_tool", {"name": "twice"}),
        ("read_tool_source", {"name": "twice"}),
        ("twice", {"value": 21}),
        ("inspect_tool_stats", {"name": "twice"}),
        ("inspect_tool_calls", {"name": "twice"}),
        ("update_tool", {"name": "twice", "source": SOURCE.replace("value * 2", "value * 3"), "entrypoint": "twice"}),
        ("twice", {"value": 14}),
    ]
    requests = []

    class Model:
        retry = RetryOptions(max_retries=0)

        async def stream(self, request):
            index = len(requests)
            requests.append(request)
            if index < len(batches):
                name, arguments = batches[index]
                message = AssistantMessage(tool_calls=(ToolCall(f"call-{index}", name, arguments),))
            else:
                message = AssistantMessage(content="done")
            yield ModelEvent.completed(ModelResponse(message))

    jit = JITHarnessExtension(template.snapshot())
    agent = await Agent.create(Model(), extensions=[jit])
    assert (await agent.run("read then improve", config=AgentRunConfig("query-split"))).content == "done"
    results = [message for message in requests[-1].messages if isinstance(message, ToolMessage)]
    assert len(results) == len(batches) and all(message.success for message in results)
    assert all(len(message.content) <= MAX_RESPONSE_CHARS for message in results)
    listing, definition, source, _, stats, calls, updated, _ = [json.loads(message.content) for message in results]
    assert "profiling" not in listing["tools"][0]
    assert "source" not in definition and "calls" not in definition
    assert source["content"] == SOURCE
    assert stats["call_count"] == 1 and "calls" not in stats
    assert calls["calls"][0]["call_id"] == "call-3"
    assert updated == {"name": "twice", "status": "loaded"}
    assert jit.get_harness("query-split").profile_stats("twice").call_count == 2
