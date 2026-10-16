"""Build a Python tool through the model loop, then save and reuse it offline."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent.agent import AgentRunConfig
from zett_agent.client import create_agent
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.jit_harness import JITHarness, JITHarnessExtension, JITHarnessSnapshot, ToolKind
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

SOURCE = '''\
@tool(guidelines="Use to add an integer to itself.")
async def twice(value: int) -> int:
    """Add the same integer twice."""
    return await harness.call_tool("add", left=value, right=value)
'''

PIPELINE_SOURCE = '''\
@tool(guidelines="Use to double every integer and sum the results.")
def sum_doubled(values: list[int]) -> int:
    """Compose a dynamic tool that itself calls a base tool."""
    return sum(harness.call_tool_sync("twice", value=value) for value in values)
'''


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


class BuilderModel:
    """Scripted decisions exercise real registration and composition, offline."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        last = request.messages[-1]
        if isinstance(last, ToolMessage) and last.name == "inspect_tool_calls":
            info = json.loads(last.text)
            assert info["calls"][0]["output_chars"] == 2
            message = AssistantMessage(content="42")
        elif isinstance(last, ToolMessage) and last.name == "twice":
            assert last.content == "42"
            message = AssistantMessage(tool_calls=(ToolCall("inspect", "inspect_tool_calls", {"name": "twice"}),))
        elif not any(definition.name == "twice" for definition in request.tools):
            message = AssistantMessage(
                tool_calls=(ToolCall("create", "create_tool", {"source": SOURCE, "entrypoint": "twice"}),)
            )
        else:
            message = AssistantMessage(tool_calls=(ToolCall("run", "twice", {"value": 21}),))
        yield ModelEvent.completed(ModelResponse(message))


async def exercise_lifecycle(path: Path) -> None:
    """Check composition, failure recovery, and portable unload state offline."""
    harness = JITHarness(tools=[add])
    harness.create_tool(SOURCE, entrypoint="twice")
    harness.create_tool(PIPELINE_SOURCE, entrypoint="sum_doubled")
    workflow = "# Sum doubled values\n\nCall sum_doubled once; verify empty input returns zero."
    harness.create_tool(
        workflow, kind=ToolKind.TEXT, name="sum_workflow", description="Use to sum doubled integers efficiently."
    )
    assert (await harness.call_tool("sum_workflow"))["content"] == workflow
    assert await harness.call_tool("sum_doubled", values=[-3, 0, 24]) == 42
    assert await harness.call_tool("sum_doubled", values=[]) == 0
    profile = harness.profile("sum_doubled")
    assert profile.stats[0].call_count == 2
    assert profile.stats[0].average_output_chars == 1.5  # "42" and "0".
    assert all(call.duration_ms >= 0 for call in profile.calls)

    before = harness.snapshot()
    try:
        harness.update_tool("twice", "invalid python !", entrypoint="twice")
    except SyntaxError:
        pass
    else:
        raise AssertionError("An invalid update must fail")
    assert harness.snapshot() == before
    assert await harness.call_tool("sum_doubled", values=[21]) == 42

    harness.unload_tool("add")
    harness.unload_tool("sum_workflow")
    assert harness.inspect_tool("sum_doubled")["status"] == "loaded"
    try:
        await harness.call_tool("sum_doubled", values=[21])
    except ValueError as error:
        assert "unloaded" in str(error)
    else:
        raise AssertionError("An unloaded dependency must not be callable")
    recent_errors = harness.tool_errors("add")
    assert recent_errors.errors[0].message == "Tool 'add' is unloaded"
    assert recent_errors.errors[0].caller_tool_name == "twice"

    harness.save(path)
    saved = JITHarnessSnapshot.load(path)
    restored = JITHarness(saved, tools=[add])
    assert restored.profile().calls == ()  # Reuse tools, not the old session's measurements.
    assert restored.tool_errors().errors == ()
    assert restored.inspect_tool("add")["status"] == "unloaded"
    restored.load_tool("add")
    assert restored.describe("sum_workflow")["status"] == "unloaded"
    restored.load_tool("sum_workflow")
    assert (await restored.call_tool("sum_workflow"))["content"] == workflow
    assert await restored.call_tool("sum_doubled", values=[21]) == 42
    assert harness.inspect_tool("add")["status"] == "unloaded"
    assert JITHarnessSnapshot.load(path) == saved  # Loading never saves implicitly.

    missing = JITHarness(saved)  # Base implementations are not serialized.
    try:
        missing.load_tool("add")
    except ValueError as error:
        assert "Missing base tool" in str(error)
    else:
        raise AssertionError("Loading requires the application to rebind its base tool")


async def main() -> None:
    jit = JITHarnessExtension()
    client = await create_agent(
        BuilderModel(),
        config=AgentRunConfig(session_id="builder"),
        tools=[add],
        extensions=[InMemoryMessageAccumulator(), jit],
    )
    assert (await client.run("Build and use twice.")).content == "42"
    measurements = jit.get_harness("builder").profile()
    assert next(stat for stat in measurements.stats if stat.tool_name == "twice").completed_count == 1
    assert next(call for call in measurements.calls if call.tool_name == "add").caller_tool_name == "twice"
    with TemporaryDirectory(prefix="zett-jit-example-") as directory:
        path = Path(directory) / "harness.json"
        jit.get_harness("builder").save(path)
        reused = JITHarnessExtension(JITHarnessSnapshot.load(path))
        other = await create_agent(
            BuilderModel(),
            config=AgentRunConfig(session_id="reused"),
            tools=[add],
            extensions=[InMemoryMessageAccumulator(), reused],
        )
        assert (await other.run("Use the saved twice tool.")).content == "42"
        await exercise_lifecycle(Path(directory) / "pipeline.json")
    print("JIT tool created and reused: 42")
    print("JIT invariants checked: composition, failed update, unload, snapshot isolation")
    print("JIT profiling checked: per-tool totals and nested call records")
    print("JIT diagnostics checked: recent dependency error retained")


if __name__ == "__main__":
    asyncio.run(main())
