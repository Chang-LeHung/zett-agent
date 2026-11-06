"""Subclass hooks for the control tools and the prompt they contribute."""

from __future__ import annotations

from dataclasses import replace

import pytest

from jit_support import put_tool
from zett_agent.agent import AgentRunConfig, AgentRunContext, AgentState
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
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
