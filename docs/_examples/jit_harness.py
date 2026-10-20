"""Let a model write a tool into its repository on disk, then reuse it offline."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent.agent import AgentRunConfig
from zett_agent.client import create_agent
from zett_agent.extensions.memory import InMemoryMessageAccumulator
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

SOURCE = '''\
from zett_agent.tools.base import tool


@tool(guidelines="Use to add an integer to itself.")
async def twice(value: int) -> int:
    """Add the same integer twice."""
    return await harness.call_tool("add", left=value, right=value)
'''

PIPELINE_SOURCE = '''\
from zett_agent.tools.base import tool


@tool(guidelines="Use to double every integer and sum the results.")
def sum_doubled(values: list[int]) -> int:
    """Compose a tool of this repository that itself calls a base tool."""
    return sum(harness.call_tool_sync("twice", value=value) for value in values)
'''

WORKFLOW = "# Sum doubled values\n\nCall sum_doubled once; verify empty input returns zero."
WORKFLOW_FILE = f"---\nkind: workflow\ndescription: Use to sum doubled integers efficiently.\n---\n{WORKFLOW}\n"


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@tool(guidelines="Use to write a file; the model edits its repository with tools like this.")
def write_file(path: str, content: str) -> str:
    """Write a text file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


class BuilderModel:
    """Scripted decisions exercise real registration and composition, offline."""

    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        last = request.messages[-1]
        offered = {definition.name for definition in request.tools}
        if isinstance(last, ToolMessage) and last.name == "twice":
            assert last.content == "42"
            message = AssistantMessage(content="42")
        elif isinstance(last, ToolMessage) and last.name == "repo_create":
            path = json.loads(last.text)["path"]
            arguments = {"path": f"{path}/twice.py", "content": SOURCE}
            message = AssistantMessage(tool_calls=(ToolCall("write", "write_file", arguments),))
        elif isinstance(last, ToolMessage) and last.name == "write_file":
            message = AssistantMessage(tool_calls=(ToolCall("load", "repo_load", {}),))
        elif "twice" in offered:
            message = AssistantMessage(tool_calls=(ToolCall("run", "twice", {"value": 21}),))
        else:
            message = AssistantMessage(tool_calls=(ToolCall("create", "repo_create", {"repo": "kit"}),))
        yield ModelEvent.completed(ModelResponse(message))


async def exercise_lifecycle(directory: Path) -> None:
    """Check composition, failure recovery, and a restart from the files alone."""
    harness = JITHarness(directory, tools=[add])
    repo = Path(harness.create_repo("kit")["path"])
    (repo / "twice.py").write_text(SOURCE)
    (repo / "pipeline.py").write_text(PIPELINE_SOURCE)
    (repo / "resources").mkdir()
    (repo / "resources" / "sum_workflow.md").write_text(WORKFLOW_FILE)
    loaded = harness.load_repo()
    assert loaded["tools"] == ["sum_doubled", "twice"]  # The note is not a tool.
    [note] = loaded["resources"]
    assert (note["name"], note["kind"], note["description"]) == (
        "sum_workflow",
        "workflow",
        "Use to sum doubled integers efficiently.",
    )
    assert WORKFLOW in Path(note["path"]).read_text()  # The model opens the file itself when it applies.
    assert harness.load_repo() == loaded  # Loading again changes nothing and imports nothing.
    assert await harness.call_tool("sum_doubled", values=[-3, 0, 24]) == 42
    assert await harness.call_tool("sum_doubled", values=[]) == 0
    profile = harness.profile("sum_doubled")
    assert profile.stats[0].call_count == 2
    assert profile.stats[0].average_output_chars == 1.5  # "42" and "0".
    assert all(call.duration_ms >= 0 for call in profile.calls)

    # What the model reads through profile_tools: the few numbers that say whether to keep a tool.
    # It answers for the application's tool (`add`) as well as the model's own.
    asked = harness.profile_tools(["sum_doubled", "twice", "add"])
    rows = {row["name"]: row for row in asked["tools"]}
    assert rows["sum_doubled"]["calls"] == 2 and rows["sum_doubled"]["success_rate"] == 1.0
    assert rows["twice"]["calls"] == 3 and rows["add"]["calls"] == 3  # One per value, from inside sum_doubled.

    (repo / "twice.py").write_text("invalid python !")
    try:
        harness.load_repo()
    except ValueError as error:
        assert "twice.py:1" in str(error)  # The model gets the file and line to fix.
    else:
        raise AssertionError("A broken file must fail the load")
    assert await harness.call_tool("sum_doubled", values=[21]) == 42  # The previous version stays.
    (repo / "twice.py").write_text(SOURCE)
    harness.load_repo()

    harness.unload_tool("add")
    harness.unload_repo()
    assert harness.describe("sum_doubled")["status"] == "unloaded"
    harness.load_repo()
    try:
        await harness.call_tool("sum_doubled", values=[21])
    except ValueError as error:
        assert "unloaded" in str(error)
    else:
        raise AssertionError("An unloaded dependency must not be callable")
    recent_errors = harness.tool_errors("add")
    assert recent_errors.errors[0].message == "Tool 'add' is unloaded"
    assert recent_errors.errors[0].caller_tool_name == "twice"
    harness.close()

    restored = JITHarness(directory, tools=[add])  # A new process finds only the files.
    assert restored.profile().calls == ()  # Reuse tools, not the old session's measurements.
    assert restored.tool_errors().errors == ()
    assert restored.create_repo("kit")["status"] == "loaded"  # Asking for it adopts and loads the files.
    again = restored.load_repo()
    assert again["tools"] == ["sum_doubled", "twice"]
    assert [item["name"] for item in again["resources"]] == ["sum_workflow"]
    assert await restored.call_tool("sum_doubled", values=[21]) == 42
    restored.close()


async def main() -> None:
    with TemporaryDirectory(prefix="zett-jit-example-") as directory:
        jit = JITHarnessExtension(directory)
        client = await create_agent(
            BuilderModel(),
            config=AgentRunConfig(session_id="builder"),
            tools=[add, write_file],
            extensions=[InMemoryMessageAccumulator(), jit],
        )
        assert (await client.run("Build and use twice.")).content == "42"
        measurements = jit.get_harness("builder").profile()
        assert next(stat for stat in measurements.stats if stat.tool_name == "twice").completed_count == 1
        assert next(call for call in measurements.calls if call.tool_name == "add").caller_tool_name == "twice"
        jit.forget_session("builder")  # Release the modules; the files stay on disk.

        # A later process with the same root and session id finds the repository again.
        resumed = JITHarnessExtension(directory)
        other = await create_agent(
            BuilderModel(),
            config=AgentRunConfig(session_id="builder"),
            tools=[add, write_file],
            extensions=[InMemoryMessageAccumulator(), resumed],
        )
        assert (await other.run("Use the saved twice tool.")).content == "42"
        resumed.forget_session("builder")
        await exercise_lifecycle(Path(directory) / "pipeline")
    print("JIT tool created and reused: 42")
    print("JIT invariants checked: composition, failed reload, unload, restart from files")
    print("JIT profiling checked: per-tool totals and nested call records")
    print("JIT diagnostics checked: recent dependency error retained")


if __name__ == "__main__":
    asyncio.run(main())
