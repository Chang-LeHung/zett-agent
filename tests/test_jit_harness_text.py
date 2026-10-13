"""Reusable workflows share lifecycle guarantees without executing their text."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from pydantic import ValidationError

from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.events import AgentEventType
from zett_agent.jit_harness import (
    JITHarness,
    JITHarnessExtension,
    JITHarnessSnapshot,
    TextToolSource,
    ToolCallStatus,
    ToolKind,
    ToolOrigin,
)
from zett_agent.jit_harness.pages import MAX_RESPONSE_CHARS, text_page
from zett_agent.jit_harness.queries import ToolQueries
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

WORKFLOW = "# Diagnose\n\n1. Reproduce.\n2. Inspect relevant evidence.\n3. Verify the result.\n"
SOURCE = '''\
@tool(guidelines="Double one integer.")
async def twice(value: int) -> int:
    """Double an integer."""
    return value * 2
'''


def create_text(harness, *, name="diagnose", source=WORKFLOW, description="Investigate a failure."):
    return harness.create_tool(source, kind=ToolKind.TEXT, name=name, description=description)


def context(*tools, session="text", extensions=()):
    return AgentRunContext(AgentRunConfig(session), AgentState(), {item.name: item for item in tools}, extensions)


class ScriptModel:
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


async def test_text_is_inert_and_mutations_return_only_a_small_acknowledgement():
    harness = JITHarness()
    inert = 'raise RuntimeError("Markdown must never execute")\n\n' + WORKFLOW
    assert create_text(harness, source=inert) == {"name": "diagnose", "status": "loaded"}
    assert harness.origin("diagnose") is ToolOrigin.TEXT
    before = harness.snapshot()
    result = await harness.call_tool("diagnose")
    assert result["content"] == inert
    assert result["content_id"] == sha256(inert.encode()).hexdigest()
    assert result["next_offset"] is None
    assert harness.snapshot() == before  # Reading neither mutates nor executes source.
    assert harness.definition("diagnose")["parameters"]["properties"].keys() == {"offset", "max_chars"}
    assert "entrypoint" not in harness.definition("diagnose")
    assert harness.source("diagnose") == inert
    full = harness.inspect_tool("diagnose")
    assert full["source"] == inert and "profiling" not in full
    assert harness.profile_stats("diagnose").completed_count == 1
    queries = ToolQueries(harness)
    assert queries.read_source("diagnose") == result
    definition = queries.inspect_tool("diagnose")
    assert definition["origin"] is ToolOrigin.TEXT
    assert "source" not in definition and "profiling" not in definition
    for operation in (
        lambda: harness.update_tool("diagnose", inert + "\nNew check."),
        lambda: harness.unload_tool("diagnose"),
        lambda: harness.load_tool("diagnose"),
    ):
        assert set(operation()) == {"name", "status"}


@pytest.mark.parametrize("managed", [True, False])
async def test_text_unload_update_load_stale_wrappers_and_policy_removals(managed):
    extension = JITHarnessExtension(manage_base_tools=managed)
    ctx = context()
    await extension.on_tool(ctx)
    harness = extension.get_harness("text")
    try:
        create_text(harness)
        old = ctx.tools["diagnose"]
        assert (await old({}))["content"] == WORKFLOW
        harness.unload_tool("diagnose")
        assert "diagnose" not in ctx.tools
        with pytest.raises(ValueError, match="unloaded"):
            await old({})
        assert harness.update_tool("diagnose", "# Better path", description="New applicability.") == {
            "name": "diagnose",
            "status": "unloaded",
        }
        harness.sync_tools(ctx)
        assert "diagnose" not in ctx.tools
        assert harness.describe("diagnose")["description"] == "Read workflow: New applicability."
        assert ToolQueries(harness).read_source("diagnose")["content"] == "# Better path"
        harness.load_tool("diagnose")
        assert (await old({}))["content"] == "# Better path"  # Old wrapper resolves the live definition.
        ctx.tools.pop("diagnose")
        harness.sync_tools(ctx)
        assert harness.describe("diagnose")["status"] == "unavailable"
        with pytest.raises(ValueError, match="not registered"):
            await old({})
        # Cleanup must invalidate request-bound wrappers.
        harness.detach(ctx)
        with pytest.raises(RuntimeError, match="closed"):
            await old({})
    finally:
        extension._request_finished(ctx)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": None},
        {"name": "bad name"},
        {"name": "load_tool"},
        {"name": "n" * 65},
        {"description": None},
        {"description": ""},
        {"description": " \n"},
        {"description": "x" * 513},
        {"description": 1},
        {"source": ""},
        {"source": " \n"},
        {"source": 1},
        {"entrypoint": "python_function"},
        {"kind": "unknown"},
    ],
)
def test_invalid_text_creation_is_atomic(kwargs):
    harness = JITHarness()
    create_text(harness, name="kept")
    before = harness.snapshot()
    arguments = {"kind": ToolKind.TEXT, "name": "diagnose", "description": "Use for failures.", "source": WORKFLOW}
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        harness.create_tool(**arguments)
    assert harness.snapshot() == before
    assert harness.origin("diagnose") is ToolOrigin.UNKNOWN


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": ""},
        {"source": " \n"},
        {"description": ""},
        {"description": " \n"},
        {"description": "x" * 513},
        {"entrypoint": "diagnose"},
    ],
)
async def test_failed_text_updates_preserve_old_source_registration_and_unload_state(kwargs):
    extension = JITHarnessExtension()
    ctx = context()
    await extension.on_tool(ctx)
    harness = extension.get_harness("text")
    try:
        create_text(harness)
        for unloaded in (False, True):
            if unloaded:
                harness.unload_tool("diagnose")
            before = harness.snapshot()
            registration = ctx.tools.get("diagnose")
            arguments = {"source": "# Replacement", **kwargs}
            with pytest.raises(ValueError):
                harness.update_tool("diagnose", **arguments)
            assert harness.snapshot() == before
            assert ctx.tools.get("diagnose") is registration
            assert harness.source("diagnose") == WORKFLOW
        harness.load_tool("diagnose")
        assert (await ctx.tools["diagnose"]({}))["content"] == WORKFLOW
    finally:
        extension._request_finished(ctx)


@pytest.mark.parametrize("owner", ["base", "python", "text"])
def test_text_names_share_namespace_with_every_other_kind(owner):
    @tool
    async def diagnose() -> str:
        """Diagnose a failure.

        Guidelines:
            - Use relevant evidence.
        """
        return "ok"

    harness = JITHarness(tools=[diagnose] if owner == "base" else [])
    if owner == "python":
        harness.create_tool(SOURCE.replace("twice", "diagnose"), entrypoint="diagnose")
    elif owner == "text":
        create_text(harness)
    before = harness.snapshot()
    with pytest.raises(ValueError, match="already exists"):
        create_text(harness)
    with pytest.raises(ValueError, match="already exists"):
        harness.create_tool(SOURCE.replace("twice", "diagnose"), entrypoint="diagnose")
    assert harness.snapshot() == before


@pytest.mark.parametrize("late", [False, True])
async def test_text_tools_cannot_be_shadowed_by_base_registrations(late):
    @tool(guidelines="Use to reproduce a namespace conflict.")
    async def diagnose() -> str:
        """Application diagnostic."""
        return "base"

    harness = JITHarness()
    create_text(harness)
    before = harness.snapshot()
    with pytest.raises(ValueError, match="Duplicate"):
        JITHarness(before, tools=[diagnose])
    ctx = context(*([] if late else [diagnose]))
    if late:
        harness.attach(ctx)
        ctx.tools["diagnose"] = diagnose
    try:
        with pytest.raises(ValueError, match="conflicts"):
            harness.sync_tools(ctx) if late else harness.attach(ctx)
        assert harness.snapshot() == before
    finally:
        harness.detach(ctx)


async def test_updates_preserve_kind_and_description_and_allow_markdown_code_blocks():
    harness = JITHarness()
    create_text(harness)
    original_description = harness.describe("diagnose")["description"]
    harness.update_tool("diagnose", "```python\nraise RuntimeError('example only')\n```")
    assert harness.origin("diagnose") is ToolOrigin.TEXT
    assert harness.describe("diagnose")["description"] == original_description
    assert "raise RuntimeError" in (await harness.call_tool("diagnose"))["content"]
    harness.create_tool(SOURCE, entrypoint="twice")
    before = harness.snapshot()
    for kwargs in ({"description": "Convert to text"}, {"entrypoint": "twice", "description": "Override"}):
        with pytest.raises(ValueError):
            harness.update_tool("twice", WORKFLOW, **kwargs)
    with pytest.raises(ValueError, match="entrypoint"):
        harness.create_tool(SOURCE)
    with pytest.raises(ValueError, match="come from"):
        harness.create_tool(SOURCE, entrypoint="twice", name="other")
    assert harness.snapshot() == before
    assert await harness.call_tool("twice", value=21) == 42


@pytest.mark.parametrize("padding", ["中文🙂", "\x01", '"\\\n'])
async def test_long_text_reads_are_lossless_bounded_and_versioned(padding):
    harness = JITHarness()
    workflow = WORKFLOW + padding * 5000
    create_text(harness, source=workflow)
    offset, pieces, ids = 0, [], set()
    while True:
        page = await harness.call_tool("diagnose", offset=offset, max_chars=4000)
        assert len(json.dumps(page, ensure_ascii=False)) <= MAX_RESPONSE_CHARS
        assert page["content"] == workflow[offset : offset + len(page["content"])]
        assert 0 < len(page["content"]) <= 4000
        assert page["total_chars"] == len(workflow)
        pieces.append(page["content"])
        ids.add(page["content_id"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert "".join(pieces) == workflow
    assert ids == {sha256(workflow.encode()).hexdigest()}
    assert (await harness.call_tool("diagnose", offset=len(workflow)))["content"] == ""
    assert (await harness.call_tool("diagnose", offset=len(workflow) + 20))["next_offset"] is None
    harness.update_tool("diagnose", WORKFLOW)
    page = await harness.call_tool("diagnose")
    assert page["content_id"] not in ids
    calls = harness.profile("diagnose").calls
    assert calls and all(call.origin is ToolOrigin.TEXT for call in calls)


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": -1},
        {"offset": True},
        {"offset": 0.5},
        {"max_chars": 0},
        {"max_chars": 4001},
        {"max_chars": True},
        {"max_chars": 1.5},
        {"unexpected": "not allowed"},
    ],
)
async def test_bad_reader_arguments_are_profiled_and_retained_as_recent_errors(arguments):
    harness = JITHarness(error_capacity=1, error_message_chars=100)
    create_text(harness)
    with pytest.raises(ValueError):
        await harness.call_tool("diagnose", **arguments)
    assert harness.profile_stats("diagnose").failed_count == 1
    error = harness.tool_errors("diagnose").errors[0]
    assert error.tool_name == "diagnose"
    assert harness.profile("diagnose").calls[0].origin is ToolOrigin.TEXT
    assert len(error.message) <= 100
    assert (await harness.call_tool("diagnose"))["content"] == WORKFLOW


def test_empty_pages_still_enforce_output_metadata_budget():
    for offset in (0, 100):
        with pytest.raises(ValueError, match="metadata"):
            text_page("x" * 10000, "", offset, max_chars=100)


async def test_mixed_snapshots_roundtrip_disk_unload_state_and_independent_activation(tmp_path):
    harness = JITHarness()
    harness.create_tool(SOURCE, entrypoint="twice")
    create_text(harness, source="```python\nraise RuntimeError('inert')\n```\n" + WORKFLOW)
    create_text(harness, name="other")
    harness.unload_tool("diagnose")
    saved = harness.snapshot()
    assert [record.name for record in saved.tools] == ["twice"]
    assert [record.name for record in saved.text_tools] == ["diagnose", "other"]
    path = tmp_path / "harness.json"
    harness.save(path)
    assert JITHarnessSnapshot.load(path) == saved
    assert JITHarnessSnapshot.from_json(saved.to_json()) == saved
    left, right = JITHarness(saved), JITHarness(saved)
    assert left.profile().calls == () and left.tool_errors().errors == ()
    with pytest.raises(ValueError, match="unloaded"):
        await left.call_tool("diagnose")
    left.load_tool("diagnose")
    assert (await left.call_tool("diagnose"))["content"] == harness.source("diagnose")
    left.update_tool("other", "# Changed")
    assert right.source("other") == WORKFLOW
    assert right.describe("diagnose")["status"] == "unloaded"
    assert await right.call_tool("twice", value=21) == 42
    assert JITHarnessSnapshot.load(path) == saved
    # Python-only v1 files written before this feature still parse normally.
    old = json.loads(JITHarness(tools=[]).snapshot().to_json())
    old.pop("text_tools")
    assert JITHarnessSnapshot.from_json(json.dumps(old)).text_tools == ()


@pytest.mark.parametrize(
    "payload",
    [
        {"text_tools": [{"name": "x", "source": WORKFLOW, "description": "x", "entrypoint": "x"}]},
        {"text_tools": [{"name": "load_tool", "source": WORKFLOW, "description": "x"}]},
        {"text_tools": [{"name": "x", "source": " ", "description": "x"}]},
        {"text_tools": [{"name": "x", "source": WORKFLOW, "description": " "}]},
        {"base_tools": ["x"], "text_tools": [{"name": "x", "source": WORKFLOW, "description": "x"}]},
        {"unloaded_tools": ["x"]},
    ],
)
def test_invalid_text_snapshots_are_rejected_without_activation(payload):
    with pytest.raises(ValidationError):
        JITHarnessSnapshot.from_json(json.dumps(payload))


def test_cross_kind_and_duplicate_text_snapshot_names_are_rejected():
    harness = JITHarness()
    create_text(harness, name="twice")
    text = harness.snapshot().text_tools[0]
    harness = JITHarness()
    harness.create_tool(SOURCE, entrypoint="twice")
    with pytest.raises(ValidationError, match="unique"):
        JITHarnessSnapshot(tools=harness.snapshot().tools, text_tools=(text,))
    with pytest.raises(ValidationError, match="unique"):
        JITHarnessSnapshot(text_tools=(text, text))
    with pytest.raises(ValidationError):
        TextToolSource(name="x", source=WORKFLOW, description="x" * 513)


@pytest.mark.parametrize("sync", [False, True])
async def test_python_can_compose_a_loaded_text_reader_and_rejected_calls_keep_origin(sync):
    harness = JITHarness()
    create_text(harness)
    source = '''\
@tool(guidelines="Read diagnostic steps.")
async def consult() -> str:
    """Read the diagnostic workflow through a composed call."""
    return (await harness.call_tool("diagnose", max_chars=4000))["content"]
'''
    if sync:
        source = source.replace("async def", "def").replace("await harness.call_tool", "harness.call_tool_sync")
    harness.create_tool(source, entrypoint="consult")
    assert await harness.call_tool("consult") == WORKFLOW
    child = harness.profile("diagnose").calls[0]
    assert child.origin is ToolOrigin.TEXT and child.caller_tool_name == "consult"
    harness.unload_tool("diagnose")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("consult")
    rejected = harness.profile("diagnose").calls[-1]
    assert rejected.status is ToolCallStatus.REJECTED and rejected.origin is ToolOrigin.TEXT
    assert rejected.caller_tool_name == "consult"
    harness.load_tool("diagnose")
    assert await harness.call_tool("consult") == WORKFLOW


async def test_management_model_loop_events_profiling_and_text_context_stay_compact():
    revised = "# Revised shortest verified path\n\nCheck the focused test."
    model = ScriptModel(
        (
            ToolCall(
                "create",
                "create_tool",
                {"kind": "text", "name": "diagnose", "description": "For failures.", "source": WORKFLOW},
            ),
        ),
        (ToolCall("read", "diagnose", {}),),
        (ToolCall("unload", "unload_tool", {"name": "diagnose"}),),
        (ToolCall("inspect", "read_tool_source", {"name": "diagnose"}),),
        (ToolCall("edit", "update_tool", {"name": "diagnose", "source": revised}),),
        (ToolCall("load", "load_tool", {"name": "diagnose"}),),
        (ToolCall("read_again", "diagnose", {}),),
    )
    extension = JITHarnessExtension()
    agent = await Agent.create(model, extensions=[extension])
    events = [event async for event in agent.stream("Learn from this task.", config=AgentRunConfig("text"))]
    messages = [message for message in model.requests[-1].messages if isinstance(message, ToolMessage)]
    assert all(message.success for message in messages)
    for message in messages:
        assert len(message.content) <= MAX_RESPONSE_CHARS
        if message.name in {"create_tool", "update_tool", "load_tool", "unload_tool"}:
            assert set(json.loads(message.content)) == {"name", "status"}
    assert (
        json.loads(next(message for message in messages if message.tool_call_id == "read").content)["content"]
        == WORKFLOW
    )
    assert json.loads(messages[-1].content)["content"] == revised
    assert "diagnose" not in {definition.name for definition in model.requests[3].tools}
    assert "diagnose" in {definition.name for definition in model.requests[-1].tools}
    # No entire workflow body in the ephemeral system instructions.
    instructions = model.requests[1].messages[0].content
    assert WORKFLOW not in instructions
    assert "substantial time, tokens" in instructions and "SKILL" in instructions
    custom = [event for event in events if event.type is AgentEventType.CUSTOM]
    text_events = [event for event in custom if event.payload.get("origin") == "text"]
    assert [event.name for event in text_events] == [
        "jit_tool_started",
        "jit_tool_completed",
        "jit_tool_started",
        "jit_tool_completed",
    ]
    harness = extension.get_harness("text")
    assert harness.profile_stats("diagnose").completed_count == 2
    assert all(call.origin is ToolOrigin.TEXT for call in harness.profile("diagnose").calls)
    assert harness.snapshot().text_tools[0].source == revised


async def test_template_workflows_are_session_local_and_only_reader_metadata_is_offered():
    template = JITHarness()
    create_text(template, source=WORKFLOW * 500)
    extension = JITHarnessExtension(template.snapshot())
    model = ScriptModel()
    agent = await Agent.create(model, extensions=[extension])
    await agent.run("Read only if relevant.", config=AgentRunConfig("one"))
    first = extension.get_harness("one")
    first.unload_tool("diagnose")
    first.update_tool("diagnose", "# Changed")
    await agent.run("Use an independent workflow.", config=AgentRunConfig("two"))
    second = extension.get_harness("two")
    assert second.source("diagnose") == WORKFLOW * 500
    assert second.describe("diagnose")["status"] == "loaded"
    assert template.source("diagnose") == WORKFLOW * 500
    for request in model.requests:
        definition = next(item for item in request.tools if item.name == "diagnose")
        assert len(definition.description) < 100
        assert WORKFLOW not in request.messages[0].content


async def test_policy_subsets_do_not_leak_text_reader_definitions():
    template = JITHarness()
    create_text(template)
    extension = JITHarnessExtension(template.snapshot())
    ctx = context()
    await extension.on_tool(ctx)
    seen = []

    async def capture(request):
        seen.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    try:
        request = ModelRequest(messages=(), tools=tuple(item.definition for item in ctx.tools.values()))
        request = replace(request, tools=tuple(item for item in request.tools if item.name != "diagnose"))
        original = request
        _ = [event async for event in extension.on_model_request(ctx, request, capture)]
        assert request is original and request.messages == ()
        assert "diagnose" not in {item.name for item in seen[0].tools}
        assert "## diagnose" not in seen[0].messages[0].content
    finally:
        extension._request_finished(ctx)


@pytest.mark.parametrize(
    "expression",
    [
        'harness.create_tool("# Unexpected", kind="text", name="intruder", description="Unexpected")',
        'harness.update_tool("diagnose", "# Unexpected")',
        'harness.unload_tool("diagnose")',
        'harness.load_tool("diagnose")',
    ],
)
def test_python_module_activation_cannot_mutate_text_workflows(expression):
    harness = JITHarness()
    create_text(harness)
    before = harness.snapshot()
    with pytest.raises(RuntimeError, match="while loading"):
        harness.create_tool(expression + "\n" + SOURCE, entrypoint="twice")
    assert harness.snapshot() == before
    assert not harness._compiling


async def test_failed_management_creation_and_update_are_small_and_recoverable():
    model = ScriptModel(
        (ToolCall("bad_create", "create_tool", {"kind": "text", "source": WORKFLOW}),),
        (
            ToolCall(
                "create",
                "create_tool",
                {"kind": "text", "name": "diagnose", "description": "For failures.", "source": WORKFLOW},
            ),
        ),
        (ToolCall("bad_update", "update_tool", {"name": "diagnose", "source": " "}),),
        (ToolCall("read", "diagnose", {}),),
    )
    extension = JITHarnessExtension()
    agent = await Agent.create(model, extensions=[extension])
    await agent.run("Learn a workflow.", config=AgentRunConfig("text"))
    messages = [message for message in model.requests[-1].messages if isinstance(message, ToolMessage)]
    assert [message.success for message in messages] == [False, True, False, True]
    assert all(len(message.content) < 2000 for message in messages)
    harness = extension.get_harness("text")
    assert harness.source("diagnose") == WORKFLOW
    assert harness.profile_stats("create_tool").failed_count == 1
    assert harness.profile_stats("update_tool").failed_count == 1
    assert harness.tool_errors("create_tool").errors and harness.tool_errors("update_tool").errors
