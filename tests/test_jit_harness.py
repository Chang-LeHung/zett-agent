"""Trusted Python tools: a repository on disk, composition, mutation, and isolation."""

from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from threading import Event
from types import ModuleType

import httpx
import pytest
from pydantic import ValidationError

from jit_support import HEADER, put_tool, put_workflow, seed_repo, write_module
from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.events import AgentEventType
from zett_agent.extensions.ask_user import AskUserExtension
from zett_agent.extensions.base import AgentExtension
from zett_agent.extensions.external import ExternalEvent
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.jit_harness.repo import MANAGEMENT_NAMES
from zett_agent.messages import (
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolMessage,
)
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.tools.base import _SYNC_TOOL_RUNNER, ToolExecutionMode, current_tool_call_id, tool

SOURCE = dedent('''\
    @tool(guidelines="Use for composed integer addition.")
    async def twice(value: int) -> int:
        """Add the same integer twice."""
        return await harness.call_tool("add", left=value, right=value)
''')
SYNC_SOURCE = SOURCE.replace("async def", "def").replace("await harness.call_tool", "harness.call_tool_sync")
PURE_SOURCE = SOURCE.replace('await harness.call_tool("add", left=value, right=value)', "value * 2")


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@tool(guidelines="Write one repository file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


def context(*tools, session="test", extensions=()):
    return AgentRunContext(AgentRunConfig(session), AgentState(), {tool.name: tool for tool in tools}, extensions)


class QuietModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request):
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))


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


@pytest.mark.parametrize("source", [SOURCE, SYNC_SOURCE])
async def test_create_compose_unload_and_load_again(source, tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    put_tool(harness, source, "twice")
    assert harness.describe("twice")["origin"] == "python"
    assert await harness.call_tool("twice", value=21) == 42
    harness.unload_tool("add")
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("twice", value=21)
    assert harness.describe("add")["status"] == "unloaded"
    harness.load_tool("add")
    assert harness.describe("add")["status"] == "loaded"
    assert await harness.call_tool("twice", value=21) == 42
    harness.close()


async def test_failed_reloads_preserve_old_tool(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, PURE_SOURCE, "twice")
    with pytest.raises(ValueError):
        put_tool(harness, "invalid python !", "twice")
    assert await harness.call_tool("twice", value=3) == 6
    put_tool(harness, PURE_SOURCE.replace("value * 2", "value * 3"), "twice")
    assert await harness.call_tool("twice", value=3) == 9
    harness.close()


@pytest.mark.parametrize("name", sorted(MANAGEMENT_NAMES))
def test_management_names_reserved(name, tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, PURE_SOURCE.replace("twice", name), name, reload=False)
    with pytest.raises(ValueError, match="reserved"):
        harness.load_repo()
    for operation in (harness.unload_tool, harness.load_tool, harness.describe):
        with pytest.raises(ValueError, match="management"):
            operation(name)


def test_repository_tool_docstrings_preserve_guidelines_and_serial_execution(tmp_path):
    expected = {
        "repo_create": (
            "Reuse existing capabilities; create tools only for missing functionality or worthwhile workflow savings.",
            "Keep every reusable tool and workflow in this one repository.",
        ),
        "repo_load": ("Call after every edit; a failed load keeps the previous version and names the file and line.",),
        "repo_unload": ("Unload when the tools are no longer needed; repo_load restores them.",),
    }
    management = JITHarnessExtension(tmp_path)._management_tools(JITHarness(tmp_path))
    assert {registered.name: registered.guidelines for registered in management} == expected
    assert all(registered.execution_mode is ToolExecutionMode.SERIAL for registered in management)


def test_reserved_management_names_match_the_exposed_management_tools(tmp_path):
    """Every reserved name must be one the extension actually exposes, and back."""
    management = JITHarnessExtension(tmp_path)._management_tools(JITHarness(tmp_path))
    assert {registered.name for registered in management} == set(MANAGEMENT_NAMES)


@pytest.mark.parametrize("kind", ["base", "python", "text"])
async def test_unload_tool_is_the_only_unload_entrypoint_for_every_kind(kind, tmp_path):
    extension = JITHarnessExtension(tmp_path)
    ctx = context(add)
    await extension.on_tool(ctx)
    harness = extension.get_harness("test")
    if kind == "python":
        put_tool(harness, PURE_SOURCE, "twice")
    elif kind == "text":
        put_workflow(harness, "addition_workflow", "Use for addition.", "# Add two numbers")
    name = {"base": "add", "python": "twice", "text": "addition_workflow"}[kind]
    arguments = {"base": {"left": 21, "right": 21}, "python": {"value": 21}, "text": {}}[kind]
    original = ctx.tools[name]
    try:
        assert not hasattr(harness, "offload_tool")
        assert "offload_tool" not in ctx.tools
        assert "repo_unload" in ctx.tools
        expected = await original(arguments)
        if kind == "text":
            assert expected["content"] == "# Add two numbers"
        else:
            assert expected == 42

        assert harness.unload_tool(name) == {"name": name, "status": "unloaded"}
        assert name not in ctx.tools
        assert harness.describe(name)["status"] == "unloaded"
        assert harness.origin("repo_unload").value == "management"
        assert harness.origin("offload_tool").value == "unknown"
        with pytest.raises(ValueError, match="unloaded"):
            await original(arguments)

        assert harness.load_tool(name) == {"name": name, "status": "loaded"}
        assert harness.describe(name)["status"] == "loaded"
        assert await ctx.tools[name](arguments) == expected
    finally:
        extension._request_finished(ctx)
        harness.close()


async def test_management_state_changes_answer_with_a_compact_digest(tmp_path):
    extension = JITHarnessExtension(tmp_path)
    ctx = context(add)
    await extension.on_tool(ctx)
    harness = extension.get_harness("test")
    source = PURE_SOURCE.replace("twice", "summarize") + "# padding for size\n" * 60
    digest_keys = {"repo", "path", "status", "tools"}
    try:
        created = await ctx.tools["repo_create"]({"repo": "kit"})
        assert set(created) == digest_keys and created["tools"] == [] and created["status"] == "created"
        write_module(harness, "summarize", source)  # The model writes its own files with ordinary file tools.
        loaded = await ctx.tools["repo_load"]({})
        assert set(loaded) == digest_keys and loaded["tools"] == ["summarize"] and loaded["status"] == "loaded"
        assert source not in json.dumps(loaded)  # The model already wrote that source.
        for value in range(8):
            assert await ctx.tools["summarize"]({"value": value}) == value * 2
        unloaded = await ctx.tools["repo_unload"]({})
        assert set(unloaded) == digest_keys and unloaded["status"] == "unloaded"
        assert "summarize" not in ctx.tools

        # Measurements are still retained, but only queries return them.
        assert harness.profile_stats("summarize").call_count == 8
        assert len(harness.profile("summarize").calls) == 8
        assert harness.describe("summarize")["status"] == "unloaded"
        assert [item["name"] for item in harness.list_tools()] == ["add", "summarize"]
    finally:
        extension._request_finished(ctx)
        harness.close()


async def test_direct_state_changes_do_not_query_profiling(tmp_path, monkeypatch):
    harness = JITHarness(tmp_path, tools=[add])

    def unexpected_summary(name):
        raise AssertionError(f"State changes must not read profiling for {name}")

    with monkeypatch.context() as patch:
        patch.setattr(harness._profiler, "summary", unexpected_summary)
        assert harness.create_repo("kit")["status"] == "created"
        put_tool(harness, PURE_SOURCE, "twice", reload=False)
        assert harness.load_repo()["status"] == "loaded"
        assert harness.unload_repo()["status"] == "unloaded"
        assert harness.load_repo()["status"] == "loaded"
        assert harness.unload_tool("twice") == {"name": "twice", "status": "unloaded"}
        assert harness.load_tool("twice") == {"name": "twice", "status": "loaded"}
        assert harness.unload_tool("add") == {"name": "add", "status": "unloaded"}
        assert harness.load_tool("add") == {"name": "add", "status": "loaded"}
    assert await harness.call_tool("twice", value=21) == 42
    assert harness.profile("twice").stats[0].completed_count == 1
    harness.close()


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (PURE_SOURCE.replace("@tool(", "@tool(deferred=True, "), "ordinary"),
        (PURE_SOURCE.replace("@tool(", "@tool(local_tool_search=True, "), "ordinary"),
        (PURE_SOURCE.replace("@tool(", '@tool(name="!", '), "Tool names"),
        (PURE_SOURCE.replace("value: int", "value"), "annotation"),
        (PURE_SOURCE.replace('guidelines="Use for composed integer addition."', 'guidelines=""'), "guideline"),
    ],
)
def test_source_validation_leaves_catalog_unchanged(source, error, tmp_path):
    before = {key for key in sys.modules if key.startswith("zett_jit_")}
    harness = JITHarness(tmp_path)
    put_tool(harness, source, "twice", reload=False)
    with pytest.raises(ValueError, match=error):
        harness.load_repo()
    assert harness.list_tools() == []
    assert {key for key in sys.modules if key.startswith("zett_jit_")} == before


async def test_module_helpers_imports_and_future_annotations(tmp_path):
    source = dedent('''\
        from __future__ import annotations
        from dataclasses import dataclass
        import math

        from zett_agent.tools.base import tool

        @dataclass
        class Result:
            value: int

        @tool(guidelines="Return a typed dataclass.")
        def answer(value: int) -> Result:
            """Return the square root."""
            return Result(math.isqrt(value))
    ''')
    harness = JITHarness(tmp_path)
    write_module(harness, "answer", source, header=False)  # A future import must stay the first statement.
    harness.load_repo()
    assert (await harness.call_tool("answer", value=81)).value == 9
    put_tool(harness, PURE_SOURCE.replace("twice", "failing").replace("value * 2", "1 / 0"), "failing")
    with pytest.raises(ZeroDivisionError):
        await harness.call_tool("failing", value=3)
    harness.close()


@pytest.mark.parametrize("expression", ['harness.unload_tool("add")', 'harness.create_repo("other")'])
def test_loading_cannot_mutate_catalog_via_harness(expression, tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    write_module(harness, "evil", expression + "\n" + SOURCE)
    with pytest.raises(ValueError, match="Cannot change the harness while loading"):
        harness.load_repo()
    assert harness.describe("add")["status"] == "loaded"


async def test_missing_base_bindings_allowed_until_called(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, SOURCE, "twice")
    bound = context(add)
    harness.attach(bound)
    assert await bound.tools["twice"]({"value": 1}) == 2
    harness.detach(bound)
    assert harness.describe("add")["status"] == "missing"
    with pytest.raises(ValueError, match="Missing base tool"):
        await harness.call_tool("twice", value=1)
    with pytest.raises(ValueError, match="Missing base tool"):
        harness.load_tool("add")
    for operation in (harness.unload_tool, harness.load_tool, harness.describe):
        with pytest.raises(ValueError, match="Unknown tool"):
            operation("unknown")
    harness.close()


def test_a_repository_tool_may_not_shadow_an_application_tool(tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    with pytest.raises(ValueError, match="already an application tool"):
        put_tool(harness, PURE_SOURCE.replace("twice", "add"), "add")
    assert harness.describe("add")["status"] == "loaded"
    harness.close()


@pytest.mark.parametrize("source", [SOURCE, SYNC_SOURCE])
async def test_shared_hooks_middleware_raw_results_and_only_outer_events(source, tmp_path):
    observed = []

    class Hooks(AgentExtension):
        async def before_tool(self, context, call):
            observed.append(("before", call.name))

        async def on_tool_call(self, context, call, call_next):
            observed.append(("middleware", call.name, current_tool_call_id()))
            return await call_next()

        async def after_tool(self, context, call, result, error):
            observed.append(("after", call.name, result.success))

    seed_repo(tmp_path, "nested", {"twice": source})
    extension = JITHarnessExtension(tmp_path)

    class Model(QuietModel):
        async def stream(self, request):
            if isinstance(request.messages[-1], ToolMessage):
                assert request.messages[-1].content == "42"
                assert sum(isinstance(item, ToolMessage) for item in request.messages) == 1
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="42")))
            else:
                yield ModelEvent.completed(
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "twice", {"value": 21}),)))
                )

    agent = await Agent.create(Model(), tools=[add], extensions=[Hooks(), extension])
    events = [event async for event in agent.stream("calculate", config=AgentRunConfig("nested"))]
    calls = [event for event in events if event.type in (AgentEventType.TOOL_STARTED, AgentEventType.TOOL_COMPLETED)]
    assert len(calls) == 2 and all(event.tool_calls[0].name == "twice" for event in calls)
    assert [(record[0], record[1]) for record in observed] == [
        ("before", "twice"),
        ("middleware", "twice"),
        ("before", "add"),
        ("middleware", "add"),
        ("after", "add"),
        ("after", "twice"),
    ]
    assert observed[1][2] == "outer" and observed[3][2] != "outer"
    assert current_tool_call_id() is None
    assert extension.get_harness("nested").active is False
    extension.forget_session("nested")


async def test_old_wrapper_rechecks_reload_unload_request_and_policy_removal(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, PURE_SOURCE, "twice")
    first = context()
    harness.attach(first)
    wrapper = first.tools["twice"]
    put_tool(harness, PURE_SOURCE.replace("value * 2", "value * 3"), "twice")
    assert await wrapper({"value": 2}) == 6
    harness.unload_tool("twice")
    assert "twice" not in first.tools
    with pytest.raises(ValueError, match="unloaded"):
        await wrapper({"value": 2})
    harness.load_tool("twice")
    first.tools.pop("twice")
    assert harness.describe("twice")["status"] == "unavailable"
    with pytest.raises(ValueError, match="not registered"):
        await wrapper({"value": 2})
    harness.detach(first)
    second = context()
    harness.attach(second)
    with pytest.raises(RuntimeError, match="closed"):
        await wrapper({"value": 2})
    with pytest.raises(RuntimeError, match="active"):
        harness.attach(context())
    with pytest.raises(RuntimeError, match="inside"):
        await harness.call_tool("twice", value=2)
    harness.detach(second)
    harness.close()


async def test_nested_failure_hook_transformation_and_recursion_bound(tmp_path):
    class Hooks(AgentExtension):
        async def after_tool(self, context, call, result, error):
            if call.name == "add":
                result.content = "transformed"

    harness = JITHarness(tmp_path, tools=[add], max_call_depth=4)
    put_tool(harness, SOURCE, "twice")
    ctx = context(add, extensions=(Hooks(),))
    harness.attach(ctx)
    assert await ctx.tools["twice"]({"value": 2}) == "transformed"
    with pytest.raises(ValidationError):
        await ctx.tools["twice"]({"value": "bad"})
    harness.detach(ctx)
    recursive = PURE_SOURCE.replace("value * 2", 'await harness.call_tool("twice", value=value)')
    put_tool(harness, recursive, "twice")
    with pytest.raises(RuntimeError, match="depth exceeded"):
        await harness.call_tool("twice", value=1)
    with pytest.raises(RuntimeError, match="synchronous JIT"):
        harness.call_tool_sync("add", left=1, right=2)
    put_tool(harness, SOURCE.replace("await harness.call_tool", "harness.call_tool_sync"), "twice")
    with pytest.raises(RuntimeError, match="event-loop"):
        await harness.call_tool("twice", value=1)
    harness.close()


async def test_session_isolation_rebinding_compaction_and_forget(tmp_path):
    extension = JITHarnessExtension(tmp_path)
    agent = await Agent.create(QuietModel(), tools=[add], extensions=[extension])
    with pytest.raises(KeyError):
        extension.get_harness("absent")
    await asyncio.gather(*(agent.run("hi", config=AgentRunConfig(name)) for name in ("one", "two")))
    first = extension.get_harness("one")
    put_tool(first, SOURCE, "twice")
    first.unload_tool("add")
    await agent.run("again", config=AgentRunConfig("one"))
    assert extension.get_harness("one") is first
    assert first.describe("twice")["status"] == "loaded"
    assert first.describe("add")["status"] == "unloaded"
    assert extension.get_harness("two").describe("add")["status"] == "missing"
    with pytest.raises(ValueError, match="Unknown tool"):
        extension.get_harness("two").describe("twice")
    assert [event async for event in agent.compact(config=AgentRunConfig("one"))] == []
    assert first.active is False
    await agent.run("again", config=AgentRunConfig("one"))
    extension.forget_session("one")
    with pytest.raises(KeyError):
        extension.get_harness("one")


async def test_interactive_nested_tool_runs_before_hook_without_nested_messages(tmp_path):
    source = dedent('''\
        @tool(guidelines="Ask the user for a decision.")
        async def decide() -> str:
            """Ask one question."""
            answer = await harness.call_tool("ask_user", question="Proceed?")
            return answer.payload["answer"]
    ''')
    seed_repo(tmp_path, "interactive", {"decide": source})

    class Model(QuietModel):
        async def stream(self, request):
            if isinstance(request.messages[-1], ToolMessage):
                assert request.messages[-1].content == "yes"
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))
            else:
                yield ModelEvent.completed(
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "decide", {}),)))
                )

    jit = JITHarnessExtension(tmp_path)
    agent = await Agent.create(Model(), extensions=[AskUserExtension(), jit])
    async with aclosing(agent.stream("decide", config=AgentRunConfig("interactive"))) as events:
        async for event in events:
            if event.type == AgentEventType.CUSTOM and event.name == "ask_user":
                call_id = event.payload["tool_call_id"]
                assert call_id != "outer"
                agent.emit_external_event(
                    ExternalEvent("ask_user_response", {"tool_call_id": call_id, "answer": "yes"}),
                    config=AgentRunConfig("interactive"),
                )
    assert jit.get_harness("interactive").active is False
    jit.forget_session("interactive")


@pytest.mark.parametrize("responses", [False, True])
async def test_both_protocols_create_write_load_unload_and_replay(responses, tmp_path):
    module = HEADER + SOURCE
    steps = [
        ("repo_create", {"repo": "kit"}),
        ("put", {"path": str(tmp_path / "wire" / "kit" / "twice.py"), "content": module}),
        ("repo_load", {}),
        ("twice", {"value": 21}),
        ("repo_unload", {}),
        ("repo_load", {}),
        ("twice", {"value": 21}),
        ("twice", {"value": 21}),
    ]
    # The second run reuses the repository the first one left loaded.
    offering = {3, 4, 6, 7, 8, 9}
    payloads = []

    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        index = len(payloads) - 1
        # The Responses payload may also declare its hosted tool_search endpoint,
        # which is not a function and carries no name.
        names = {item["function"]["name"] if "function" in item else item.get("name") for item in payload["tools"]}
        assert ("twice" in names) == (index in offering)
        assert request.url.path.endswith("/responses" if responses else "/chat/completions")
        if index >= len(steps):
            if responses:
                output = [
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "42", "annotations": []}],
                    }
                ]
            else:
                delta = {"content": "42"}
        else:
            name, arguments = steps[index]
            if responses:
                output = [
                    {
                        "type": "function_call",
                        "id": f"fc{index}",
                        "call_id": f"call{index}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                        "status": "completed",
                    }
                ]
            else:
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                }
        if responses:
            event = {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": f"r{index}",
                    "object": "response",
                    "created_at": 0,
                    "model": "test",
                    "status": "completed",
                    "output": output,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            }
            sse = f"event: response.completed\ndata: {json.dumps(event)}\n\n"
        else:
            event = {
                "id": f"r{index}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            sse = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    provider = OpenAIProvider(
        "test", "test", response=responses, transport=httpx.MockTransport(respond), retry=RetryOptions(max_retries=0)
    )
    extension = JITHarnessExtension(tmp_path)
    memory = InMemoryMessageAccumulator()
    try:
        agent = await Agent.create(
            provider,
            tools=[add, put],
            config=AgentRunConfig("wire"),
            extensions=[memory, ToolGuidelinesExtension(), extension],
        )
        assert (await agent.run("create and use tools")).content == "42"
        assert (await agent.run("reuse tools")).content == "42"
    finally:
        await provider.aclose()
    assert len(payloads) == 10
    assert all(
        registered.execution_mode is ToolExecutionMode.SERIAL
        for registered in extension._management_tools(extension.get_harness("wire"))
    )
    assert all(not isinstance(message, SystemMessage) for message in memory.messages("wire"))
    extension.forget_session("wire")


async def test_observation_stays_silent_for_a_session_without_a_harness(tmp_path):
    """A defensive guard: no harness means no JIT event, and no failure."""
    jit = JITHarnessExtension(tmp_path)
    ctx = context(add, session="unbound")
    call = ToolCall("call", "add", {"left": 1, "right": 1})
    result = ToolMessage(tool_call_id="call", name="add", content="2")
    assert jit._observed_origin(ctx, "add") is None
    await jit.before_tool(ctx, call)
    await jit.after_tool(ctx, call, result, None)
    assert jit._observed_origin(ctx, "add") is None


@pytest.mark.parametrize("factory", [JITHarness, JITHarnessExtension])
def test_depth_configuration_is_validated(factory, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        factory(tmp_path, max_call_depth=0)


def test_constructor_collisions_and_attach_conflicts(tmp_path):
    harness = JITHarness(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        JITHarness(tmp_path, tools=[add, add])
    with pytest.raises(ValueError, match="client-side"):
        JITHarness(tmp_path, tools=[replace(add, local_tool_search=True)])
    put_tool(harness, PURE_SOURCE, "twice")
    for candidate in (replace(add, name="twice"), replace(add, local_tool_search=True)):
        with pytest.raises(ValueError, match="conflicts|client-side"):
            harness.attach(context(candidate))
    assert harness.active is False
    harness.close()


async def test_nested_exceptions_and_multimodal_results_keep_hooks_and_raw_values(tmp_path):
    seen = []

    class Hooks(AgentExtension):
        async def after_tool(self, context, call, result, error):
            seen.append((call.name, result.success, error))

    @tool(guidelines="Return a value for tests.")
    async def result(kind: str):
        """Return different result types."""
        if kind == "failure":
            raise ValueError("nested failure")
        if kind == "text":
            return TextContent("text")
        return [TextContent("text"), ImageContent(ImageBytesSource(b"image", "image/png"))]

    harness = JITHarness(tmp_path, tools=[result])
    source = dedent('''\
        @tool(guidelines="Return a dependency result.")
        async def proxy(kind: str):
            """Proxy the result tool."""
            return await harness.call_tool("result", kind=kind)
    ''')
    put_tool(harness, source, "proxy")
    ctx = context(result, extensions=(Hooks(),))
    harness.attach(ctx)
    assert await ctx.tools["proxy"]({"kind": "text"}) == TextContent("text")
    assert len(await ctx.tools["proxy"]({"kind": "parts"})) == 2
    with pytest.raises(ValueError, match="nested failure"):
        await ctx.tools["proxy"]({"kind": "failure"})
    assert seen[-1][0:2] == ("result", False) and isinstance(seen[-1][2], ValueError)
    assert not harness._nested_tasks and current_tool_call_id() is None
    harness.detach(ctx)
    harness.close()


async def test_sync_composition_does_not_depend_on_available_default_pool_workers(tmp_path):
    # A single default worker would deadlock a pooling parent→child bridge.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    harness = JITHarness(tmp_path, tools=[add])
    put_tool(harness, SYNC_SOURCE, "twice")
    source = SYNC_SOURCE.replace("twice", "four_times").replace(
        '"add", left=value, right=value', '"twice", value=value * 2'
    )
    put_tool(harness, source, "four_times")
    assert await asyncio.wait_for(harness.call_tool("four_times", value=3), timeout=2) == 12
    assert _SYNC_TOOL_RUNNER.get() is None
    harness.close()


async def test_inflight_execution_keeps_old_implementation_during_reload_and_unload(tmp_path):
    entered, finish = asyncio.Event(), asyncio.Event()

    @tool(guidelines="Wait for controlled tests.")
    async def wait() -> int:
        """Wait and then return."""
        entered.set()
        await finish.wait()
        return 2

    harness = JITHarness(tmp_path, tools=[wait])
    source = PURE_SOURCE.replace("value * 2", '(await harness.call_tool("wait")) * value')
    put_tool(harness, source, "twice")
    old = asyncio.create_task(harness.call_tool("twice", value=3))
    await entered.wait()
    put_tool(harness, PURE_SOURCE.replace("value * 2", "value * 3"), "twice")
    assert await harness.call_tool("twice", value=3) == 9
    harness.unload_tool("twice")
    finish.set()
    assert await old == 6
    with pytest.raises(ValueError, match="unloaded"):
        await harness.call_tool("twice", value=3)
    harness.close()


async def test_binding_cleanup_after_model_failure_and_registration_conflict(tmp_path):
    extension = JITHarnessExtension(tmp_path)

    class Fail(QuietModel):
        async def stream(self, request):
            raise RuntimeError("model failed")
            yield

    agent = await Agent.create(Fail(), tools=[add], config=AgentRunConfig("failure"), extensions=[extension])
    with pytest.raises(RuntimeError, match="model failed"):
        await agent.run("fail")
    assert extension.get_harness("failure").active is False
    await agent.run("recover", model=QuietModel())
    reserved = replace(add, name="repo_create")
    conflict = await Agent.create(
        QuietModel(), tools=[reserved], config=AgentRunConfig("conflict"), extensions=[extension]
    )
    with pytest.raises(ValueError, match="reserved"):
        await conflict.run("fail")
    assert extension.get_harness("conflict").active is False
    extension._release(context(session="never started"))


async def test_cancelled_sync_tool_cannot_call_or_mutate_a_later_request(monkeypatch, tmp_path):
    entered, release, finished = Event(), Event(), Event()
    errors = []

    @tool(guidelines="Wait for the test to release the worker.")
    def wait() -> None:
        """Block outside the loop until the test releases the thread."""
        entered.set()
        release.wait(5)

    @tool(guidelines="Collect a worker's late errors.")
    def report() -> None:
        """Mark the late worker as finished."""
        finished.set()

    state = ModuleType("zett_jit_test_state")
    state.__dict__.update(entered=entered, release=release, finished=finished, errors=errors)
    monkeypatch.setitem(sys.modules, state.__name__, state)
    source = dedent('''\
        import zett_jit_test_state as state
        @tool(guidelines="Exercise cancellation.")
        def late() -> None:
            """Wait, then attempt calls after request cancellation."""
            state.entered.set()
            state.release.wait(5)
            for operation in (
                lambda: harness.call_tool_sync("report"),
                lambda: harness.unload_tool("report"),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    state.errors.append(str(error))
            state.finished.set()
    ''')
    seed_repo(tmp_path, "cancelled", {"late": source})
    extension = JITHarnessExtension(tmp_path)

    class Model(QuietModel):
        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "late", {}),))))

    agent = await Agent.create(
        Model(), tools=[wait, report], config=AgentRunConfig("cancelled"), extensions=[extension]
    )
    running = asyncio.create_task(agent.run("wait"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        harness = extension.get_harness("cancelled")
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert harness.active is False
        new = context(wait, report, session="cancelled")
        harness.attach(new)
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        assert len(errors) == 2 and all("closed" in error for error in errors)
        assert harness.describe("report")["status"] == "loaded"
        harness.detach(new)
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    extension.forget_session("cancelled")


async def test_forget_active_session_and_policy_filtered_guidance(tmp_path):
    extension = JITHarnessExtension(tmp_path)
    ctx = context(add)
    await extension.on_tool(ctx)
    with pytest.raises(RuntimeError, match="active"):
        extension.forget_session("test")
    harness = extension.get_harness("test")
    harness.unload_tool("add")
    request = ModelRequest(
        messages=(SystemMessage(content="stale", attributes={"tool_guidelines": True}),),
        tools=tuple(tool.definition for tool in ctx.tools.values()),
    )
    recorded = []

    async def next_(request):
        recorded.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))

    assert [event async for event in extension.on_model_request(ctx, request, next_)]
    assert "stale" not in "\n".join(item.content for item in recorded[-1].messages)
    assert "## add" not in recorded[-1].messages[0].content
    filtered = replace(request, tools=())
    assert [event async for event in extension.on_model_request(ctx, filtered, next_)]
    assert recorded[-1] is filtered
    harness.detach(ctx)
    extension.forget_session("test")


async def test_management_wrappers_reject_old_request_references(tmp_path):
    extension = JITHarnessExtension(tmp_path)
    first = context(add)
    await extension.on_tool(first)
    controls = {name: first.tools[name] for name in ("repo_create", "repo_unload")}
    harness = extension.get_harness("test")
    harness.detach(first)
    for control in controls.values():
        with pytest.raises(RuntimeError, match="closed"):
            await control({})
    second = context(add)
    await extension.on_tool(second)
    for control in controls.values():
        with pytest.raises(RuntimeError, match="closed"):
            await control({})
    assert harness.describe("add")["status"] == "loaded"
    assert (await second.tools["repo_create"]({"repo": "kit"}))["status"] == "created"
    current = second.tools.pop("repo_create")
    with pytest.raises(ValueError, match="not registered"):
        await current({})
    harness.detach(second)


async def test_binding_cleanup_survives_earlier_terminal_hook_failures(tmp_path):
    class BrokenHooks(AgentExtension):
        async def on_event(self, context, event):
            if context.state.phase.value in ("failed", "cancelled"):
                raise RuntimeError("terminal event failed")

        async def on_error(self, context, error):
            raise RuntimeError("error hook failed")

    class Fail(QuietModel):
        async def stream(self, request):
            raise ValueError("model failed")
            yield

    jit = JITHarnessExtension(tmp_path)
    agent = await Agent.create(Fail(), tools=[add], extensions=[BrokenHooks(), jit])
    with pytest.raises(ValueError, match="model failed"):
        await agent.run("fail", config=AgentRunConfig("failure"))
    assert jit.get_harness("failure").active is False
    assert not jit.pending
    assert (await agent.run("recover", config=AgentRunConfig("failure"), model=QuietModel())).content == "ok"


async def test_cancelled_sync_bridge_releases_nested_task_and_worker(tmp_path):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    finished = Event()
    jit = JITHarnessExtension(tmp_path)

    @tool(guidelines="Wait until the test cancels the request.")
    async def wait() -> None:
        """Wait for request cancellation."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    @tool(guidelines="Compose the waiting dependency from a sync worker.")
    def parent() -> None:
        """Bridge to the request loop and always signal worker completion."""
        try:
            jit.get_harness("bridge").call_tool_sync("wait")
        finally:
            finished.set()

    class Model(QuietModel):
        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(tool_calls=(ToolCall("outer", "parent", {}),))))

    agent = await Agent.create(Model(), tools=[wait, parent], extensions=[jit])
    running = asyncio.create_task(agent.run("wait", config=AgentRunConfig("bridge")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        assert await asyncio.to_thread(finished.wait, 2)
        harness = jit.get_harness("bridge")
        assert harness.active is False and not harness._nested_tasks
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
