"""Subclass hooks for the control tools and the prompt they contribute."""

from __future__ import annotations

from dataclasses import replace

import pytest

from jit_support import put_resource, put_tool, seed_repo
from zett_agent.agent import AgentRunConfig, AgentRunContext, AgentState
from zett_agent.jit_harness import (
    MAX_NOTE_CATALOG_CHARS,
    JITHarness,
    JITHarnessExtension,
    TextKind,
    render_note_catalog,
)
from zett_agent.jit_harness.enums import ToolCallStatus, ToolOrigin
from zett_agent.jit_harness.extension import _INSTRUCTIONS
from zett_agent.jit_harness.repo import MANAGEMENT_NAMES
from zett_agent.messages import AssistantMessage, SystemMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse
from zett_agent.tools.base import ToolExecutionMode, tool

SHADOW_SOURCE = '''\
@tool(guidelines="Shadow a management name.")
def cache_report() -> int:
    """Shadow the management tool."""
    return 1
'''


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@tool(execution_mode=ToolExecutionMode.PARALLEL, guidelines="Use for cache statistics.")
def cache_report() -> dict[str, int]:
    """Report one cache hit count."""
    return {"hits": 1}


def context(*tools, session="api"):
    return AgentRunContext(AgentRunConfig(session), AgentState(), {tool.name: tool for tool in tools}, ())


class CustomJIT(JITHarnessExtension):
    def management_tools(self, harness):
        tools = [tool for tool in super().management_tools(harness) if tool.name != "profile_tools"]
        tools.append(cache_report)
        return tuple(tools)


def test_default_management_surface_is_unchanged(tmp_path):
    extension = JITHarnessExtension(tmp_path)
    harness = JITHarness(tmp_path)
    tools = tuple(extension.management_tools(harness))
    assert [tool.name for tool in tools] == [
        "repo_create",
        "repo_load",
        "repo_unload",
        "profile_tools",
    ]
    assert {tool.name for tool in tools} == set(MANAGEMENT_NAMES)
    assert all(tool.execution_mode is ToolExecutionMode.SERIAL for tool in tools)
    assert extension.instructions(harness) == _INSTRUCTIONS
    assert extension.guidance_tool == "repo_create"


def test_origin_tracks_attach_state_instead_of_the_global_reservation(tmp_path):
    harness = JITHarness(tmp_path)
    assert harness.management_names == frozenset()
    assert harness.origin("repo_load") is ToolOrigin.UNKNOWN

    ctx = context()
    harness.attach(ctx, tuple(JITHarnessExtension(tmp_path).management_tools(harness)))
    try:
        assert harness.origin("repo_load") is ToolOrigin.MANAGEMENT
        assert harness.management_names == MANAGEMENT_NAMES
    finally:
        harness.detach(ctx)
    assert harness.management_names == frozenset()
    assert harness.origin("repo_load") is ToolOrigin.UNKNOWN
    harness.close()


async def test_a_subclass_can_remove_add_and_reword_control_tools(tmp_path):
    extension = CustomJIT(tmp_path)
    ctx = context(session="custom")
    await extension.on_tool(ctx)
    harness = extension.get_harness("custom")
    try:
        assert "profile_tools" not in ctx.tools
        assert "cache_report" in ctx.tools
        assert harness.management_names == frozenset({"repo_create", "repo_load", "repo_unload", "cache_report"})
        assert harness.origin("profile_tools") is ToolOrigin.UNKNOWN
        assert harness.origin("cache_report") is ToolOrigin.MANAGEMENT
        assert ctx.tools["cache_report"].execution_mode is ToolExecutionMode.SERIAL

        assert await ctx.tools["cache_report"]({}) == {"hits": 1}
        record = harness.profile("cache_report").calls[0]
        assert record.origin is ToolOrigin.MANAGEMENT
        assert record.status is ToolCallStatus.COMPLETED
        with pytest.raises(ValueError, match="control tools are not measured"):
            harness.profile_tools(["cache_report"])
    finally:
        extension._request_finished(ctx)
        extension.forget_session("custom")


async def test_guidance_tool_none_uses_any_registered_management_tool(tmp_path):
    class AnyGuidanceJIT(CustomJIT):
        guidance_tool = None

    extension = AnyGuidanceJIT(tmp_path)
    ctx = context(session="any")
    await extension.on_tool(ctx)
    try:
        harness = extension.get_harness("any")
        assert extension.wants_guidance(harness, {"cache_report"})
        assert not extension.wants_guidance(harness, set())
    finally:
        extension._request_finished(ctx)
        extension.forget_session("any")


async def _request_messages(extension, ctx, request):
    recorded = []

    async def next_(current):
        recorded.append(current)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    assert [event async for event in extension.on_model_request(ctx, request, next_)]
    return recorded[0]


async def test_guidance_tool_none_injects_without_repo_create(tmp_path):
    class NoCreateJIT(CustomJIT):
        guidance_tool = None

        def management_tools(self, harness):
            return tuple(tool for tool in super().management_tools(harness) if tool.name != "repo_create")

    extension = NoCreateJIT(tmp_path)
    ctx = context(session="no-create")
    await extension.on_tool(ctx)
    try:
        request = ModelRequest(
            messages=(SystemMessage(content="base"),),
            tools=(ctx.tools["cache_report"].definition,),
        )
        current = await _request_messages(extension, ctx, request)
        assert any(
            isinstance(message, SystemMessage) and message.attributes.get("jit_harness:guidance")
            for message in current.messages
        )
    finally:
        extension._request_finished(ctx)
        extension.forget_session("no-create")


async def test_renamed_guidance_tool_controls_injection(tmp_path):
    class RenamedJIT(JITHarnessExtension):
        guidance_tool = "workspace_create"

        def management_tools(self, harness):
            return tuple(
                replace(tool, name="workspace_create", snippet='workspace_create(repo="kit")')
                if tool.name == "repo_create"
                else tool
                for tool in super().management_tools(harness)
            )

        def instructions(self, harness):
            return "RENAMED\n"

    extension = RenamedJIT(tmp_path)
    ctx = context(session="renamed")
    await extension.on_tool(ctx)
    try:
        request = ModelRequest(
            messages=(SystemMessage(content="base"),),
            tools=(ctx.tools["workspace_create"].definition,),
        )
        current = await _request_messages(extension, ctx, request)
        guidance = next(
            message
            for message in current.messages
            if isinstance(message, SystemMessage) and message.attributes.get("jit_harness:guidance")
        )
        assert guidance.content.startswith("RENAMED")

        filtered = ModelRequest(messages=(SystemMessage(content="base"),))
        current = await _request_messages(extension, ctx, filtered)
        assert all(
            not (isinstance(message, SystemMessage) and message.attributes.get("jit_harness:guidance"))
            for message in current.messages
        )
    finally:
        extension._request_finished(ctx)
        extension.forget_session("renamed")


async def test_a_dangling_guidance_tool_fails_before_attach(tmp_path):
    class DanglingJIT(CustomJIT):
        guidance_tool = "missing"

    extension = DanglingJIT(tmp_path)
    ctx = context(session="dangling")
    with pytest.raises(ValueError, match="guidance_tool"):
        await extension.on_tool(ctx)
    harness = extension.get_harness("dangling")
    assert not harness.active
    extension.forget_session("dangling")


async def test_instructions_override_reaches_the_injected_system_message(tmp_path):
    class PromptJIT(JITHarnessExtension):
        def instructions(self, harness):
            return "CUSTOM INSTRUCTIONS\n"

    extension = PromptJIT(tmp_path)
    ctx = context(session="prompt")
    await extension.on_tool(ctx)
    try:
        request = ModelRequest(
            messages=(SystemMessage(content="base"),), tools=tuple(t.definition for t in ctx.tools.values())
        )
        recorded = []

        async def next_(current):
            recorded.append(current)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

        assert [event async for event in extension.on_model_request(ctx, request, next_)]
        guidance = next(
            message
            for message in recorded[0].messages
            if isinstance(message, SystemMessage) and message.attributes.get("jit_harness:guidance")
        )
        assert guidance.content.startswith("CUSTOM INSTRUCTIONS")
        assert guidance.persist is False
    finally:
        extension._request_finished(ctx)
        extension.forget_session("prompt")


def test_attach_rejects_invalid_management_tools_and_stays_detached(tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    cases = [
        ((add, add), "Duplicate"),
        ((object(),), "AgentTool"),
        ((replace(add, name="!"),), "Invalid"),
        ((replace(add, deferred=True),), "ordinary"),
        ((replace(add, local_tool_search=True),), "ordinary"),
        ((add,), "conflicts"),
    ]
    for tools, message in cases:
        with pytest.raises(ValueError, match=message):
            harness.attach(context(add), tools)
        assert not harness.active
        assert harness.management_names == frozenset()
    harness.close()


async def test_sync_tools_rejects_a_late_base_tool_with_a_management_name(tmp_path):
    extension = CustomJIT(tmp_path)
    ctx = context(session="late")
    await extension.on_tool(ctx)
    harness = extension.get_harness("late")
    try:
        ctx.tools["cache_report"] = replace(add, name="cache_report")
        with pytest.raises(ValueError, match="management"):
            harness.sync_tools(ctx)
    finally:
        extension._request_finished(ctx)
        extension.forget_session("late")


async def test_a_repository_tool_cannot_shadow_a_custom_management_name(tmp_path):
    harness = JITHarness(tmp_path)
    ctx = context()
    harness.attach(ctx, tuple(CustomJIT(tmp_path).management_tools(harness)))
    try:
        put_tool(harness, SHADOW_SOURCE, "cache_report", reload=False)
        with pytest.raises(ValueError, match="reserved for JIT management"):
            harness.load_repo()
    finally:
        harness.detach(ctx)
        harness.close()


def test_resources_follow_the_last_successful_repository_scan(tmp_path):
    harness = JITHarness(tmp_path)
    assert harness.resources == []
    harness.create_repo("kit")
    assert harness.resources == []
    put_resource(harness, "debugging", "Use when a test fails.", "Read the traceback.", kind="skill")
    try:
        resources = harness.resources
        assert [item["name"] for item in resources] == ["debugging"]
        assert resources[0]["kind"] == "skill"
        assert resources[0]["description"] == "Use when a test fails."
        assert resources[0]["path"].endswith("resources/debugging.md")
    finally:
        harness.close()


def test_the_notes_catalog_is_off_by_default(tmp_path):
    harness = JITHarness(tmp_path)
    put_resource(harness, "debugging", "Use when a test fails.", "Read the traceback.", kind="skill")
    extension = JITHarnessExtension(tmp_path)
    try:
        assert extension.notes_catalog(harness) == ""
        prompt = extension._guidance(harness, "")
        assert "Use when a test fails." not in prompt
        assert "Read the traceback." not in prompt
    finally:
        harness.close()


def test_an_enabled_catalog_is_empty_when_there_are_no_notes(tmp_path):
    class NotesJIT(JITHarnessExtension):
        include_notes = True

    harness = JITHarness(tmp_path)
    harness.create_repo("kit")
    try:
        assert NotesJIT(tmp_path).notes_catalog(harness) == ""
    finally:
        harness.close()


def test_notes_catalog_can_filter_the_default_format(tmp_path):
    class SkillsOnly(JITHarnessExtension):
        def notes_catalog(self, harness):
            return render_note_catalog([note for note in harness.resources if note["kind"] == "skill"])

    harness = JITHarness(tmp_path)
    put_resource(harness, "debugging", "Use when a test fails.", "Read the traceback.", kind="skill")
    put_resource(harness, "release", "Use when cutting a release.", "Publish it.", kind="workflow")
    try:
        catalog = SkillsOnly(tmp_path).notes_catalog(harness)
        assert "`debugging`" in catalog
        assert "`release`" not in catalog
    finally:
        harness.close()


def test_resources_keep_the_previous_scan_after_a_failed_load(tmp_path):
    harness = JITHarness(tmp_path)
    put_resource(harness, "debugging", "Use when a test fails.", "Read the traceback.", kind="skill")
    try:
        before = harness.resources
        assert harness.repo_path is not None
        (harness.repo_path / "resources" / "broken.md").write_text(
            "---\nkind: unknown\ndescription: Broken\n---\nBody\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="kind"):
            harness.load_repo()
        assert harness.resources == before
    finally:
        harness.close()


def test_a_deleted_note_stays_listed_until_the_next_successful_scan(tmp_path):
    harness = JITHarness(tmp_path)
    put_resource(harness, "debugging", "Use when a test fails.", "Read the traceback.", kind="skill")
    try:
        assert harness.repo_path is not None
        (harness.repo_path / "resources" / "debugging.md").unlink()
        assert [note["name"] for note in harness.resources] == ["debugging"]
        harness.load_repo()
        assert harness.resources == []
    finally:
        harness.close()


async def test_include_notes_adds_the_catalog_to_the_injected_guidance(tmp_path):
    class NotesJIT(JITHarnessExtension):
        include_notes = True

    extension = NotesJIT(tmp_path)
    ctx = context(session="notes")
    await extension.on_tool(ctx)
    harness = extension.get_harness("notes")
    try:
        put_resource(harness, "debugging", "Use when a test fails.", "Read the traceback.", kind="skill")
        request = ModelRequest(
            messages=(SystemMessage(content="base"),), tools=tuple(t.definition for t in ctx.tools.values())
        )
        current = await _request_messages(extension, ctx, request)
        guidance = next(
            message
            for message in current.messages
            if isinstance(message, SystemMessage) and message.attributes.get("jit_harness:guidance")
        )
        assert "## Notes" in guidance.content
        assert "Use when a test fails." in guidance.content
        assert str(harness.repo_path / "resources" / "debugging.md") in guidance.content
        assert "Read the traceback." not in guidance.content
        assert guidance.content.index("Your repository") < guidance.content.index("## Notes")
        assert guidance.content.index("## Notes") < guidance.content.index("# Tool guidelines")
    finally:
        extension._request_finished(ctx)
        extension.forget_session("notes")


async def test_a_resumed_session_includes_notes_without_calling_repo_load(tmp_path):
    class NotesJIT(JITHarnessExtension):
        include_notes = True

    repo = seed_repo(tmp_path, "resumed", {})
    resources = repo / "resources"
    resources.mkdir()
    (resources / "debugging.md").write_text(
        "---\nkind: skill\ndescription: Use when a test fails.\n---\nRead the traceback.\n",
        encoding="utf-8",
    )
    extension = NotesJIT(tmp_path)
    ctx = context(session="resumed")
    await extension.on_tool(ctx)
    try:
        harness = extension.get_harness("resumed")
        assert [note["name"] for note in harness.resources] == ["debugging"]
        request = ModelRequest(
            messages=(SystemMessage(content="base"),), tools=tuple(t.definition for t in ctx.tools.values())
        )
        current = await _request_messages(extension, ctx, request)
        guidance = next(
            message
            for message in current.messages
            if isinstance(message, SystemMessage) and message.attributes.get("jit_harness:guidance")
        )
        assert "Use when a test fails." in guidance.content
    finally:
        extension._request_finished(ctx)
        extension.forget_session("resumed")


def test_the_notes_catalog_is_bounded_and_names_the_remaining_notes():
    resources = [
        {
            "name": f"note-{index}",
            "kind": TextKind.WORKFLOW,
            "description": "x" * 512,
            "path": f"/repo/resources/note-{index}.md",
        }
        for index in range(50)
    ]
    catalog = render_note_catalog(resources)
    assert len(catalog) <= MAX_NOTE_CATALOG_CHARS
    assert "more note(s); call `repo_load` for the complete, untruncated list." in catalog
    assert "note-49" not in catalog

    short = [
        {
            "name": f"short-{index}",
            "kind": TextKind.WORKFLOW,
            "description": "short",
            "path": f"/repo/resources/short-{index}.md",
        }
        for index in range(20)
    ]
    bounded = render_note_catalog(short, limit=500)
    assert "more note(s); call `repo_load` for the complete, untruncated list." in bounded
    assert len(bounded) <= 500
    assert render_note_catalog(short, limit=50) == ""
    full = render_note_catalog(short)
    assert render_note_catalog(short, limit=len(full)) == full
    with pytest.raises(ValueError, match="positive"):
        render_note_catalog(short, limit=0)


def test_the_catalog_drops_entries_rather_than_omit_the_summary():
    notes = [
        {
            "name": f"note-{index}",
            "kind": TextKind.WORKFLOW,
            "description": "x" * 80,
            "path": f"/repo/resources/note-{index}.md",
        }
        for index in range(2)
    ]
    one = render_note_catalog(notes[:1])
    assert one
    # One entry fits exactly, but its closing line does not: drop it rather than truncate it.
    assert render_note_catalog(notes, limit=len(one)) == ""
    assert render_note_catalog(notes, limit=len(one) - 1) == ""
    summary = "- 1 more note(s); call `repo_load` for the complete, untruncated list."
    exact = render_note_catalog(notes, limit=len(one) + len(summary) + 1)
    assert exact == one + "\n" + summary
