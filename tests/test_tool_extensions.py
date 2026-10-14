"""One extension, one tool: every built-in tool can be added on its own."""

import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.bash import BashExtension
from zett_agent.extensions.coding import CodingExtension
from zett_agent.extensions.file_system import (
    DeleteFileExtension,
    EditExtension,
    FileSystemExtension,
    GlobExtension,
    GrepExtension,
    ReadExtension,
    ReadFileExtension,
    ViewImageExtension,
    WriteFileExtension,
)
from zett_agent.extensions.tool import ToolExtension
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.tools.base import tool
from zett_agent.tools.coding import bash, delete_file, edit, glob, grep, read, read_file, write_file
from zett_agent.tools.images import view_image

SINGLE = [
    (ReadFileExtension, "read_file"),
    (ReadExtension, "read"),
    (ViewImageExtension, "view_image"),
    (GlobExtension, "glob"),
    (GrepExtension, "grep"),
    (WriteFileExtension, "write_file"),
    (EditExtension, "edit"),
    (DeleteFileExtension, "delete_file"),
    (BashExtension, "bash"),
]


class RecordingModel:
    def __init__(self, *calls: ToolCall) -> None:
        self.calls = calls
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        message = AssistantMessage(tool_calls=self.calls) if len(self.requests) == 1 and self.calls else None
        yield ModelEvent.completed(ModelResponse(message or AssistantMessage(content="done")))


@pytest.mark.parametrize(("extension_type", "name"), SINGLE)
async def test_each_extension_registers_exactly_its_tool(extension_type, name):
    model = RecordingModel()
    extension = extension_type()
    agent = await Agent.create(model, config=AgentRunConfig(f"single-{name}"), extensions=[extension])

    await agent.run("hi")

    assert [definition.name for definition in model.requests[0].tools] == [name]
    assert [registered.name for registered in extension.tools] == [name]
    assert extension.tool.name == name
    assert isinstance(extension, ToolExtension)


async def test_a_subset_of_extensions_runs_only_its_own_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\n")
    model = RecordingModel(
        ToolCall("1", "read_file", {"path": "a.txt"}),
        ToolCall("2", "grep", {"pattern": "alpha"}),
        ToolCall("3", "write_file", {"path": "b.txt", "content": "x"}),
    )
    agent = await Agent.create(
        model, config=AgentRunConfig("subset"), extensions=[ReadFileExtension(), GrepExtension()]
    )

    await agent.run("go")

    results = {m.tool_call_id: m for m in model.requests[1].messages if isinstance(m, ToolMessage)}
    assert results["1"].success and results["1"].content == "alpha\n"
    assert results["2"].success
    assert not results["3"].success  # write_file was not offered
    assert not (tmp_path / "b.txt").exists()


def test_the_bundles_are_exactly_the_union_of_the_single_tool_extensions():
    read_only = {registered.name for registered in FileSystemExtension(read_only=True).tools}
    writable = {registered.name for registered in FileSystemExtension().tools}

    assert read_only == {"read_file", "view_image", "glob", "grep"}
    assert writable == read_only | {"write_file", "edit", "delete_file"}
    singles = {
        registered.name
        for extension_type, name in SINGLE
        if name not in {"bash", "read"}
        for registered in extension_type().tools
    }
    assert singles == writable
    # `read` is opt-in: the bundles keep their long-standing tool sets.
    assert "read" not in writable
    assert {registered.name for registered in CodingExtension().tools} == writable | {"bash"}


def test_tool_extension_accepts_any_tool_and_rejects_everything_else():
    @tool(guidelines="Use for tests.")
    async def custom() -> str:
        """Do a thing."""
        return "ok"

    assert ToolExtension(custom).tool is custom
    for bad in (custom.handler, "custom", None):
        with pytest.raises(TypeError, match="AgentTool"):
            ToolExtension(bad)


async def test_a_custom_tool_is_added_without_writing_an_extension():
    @tool(guidelines="Use for tests.")
    async def custom() -> str:
        """Do a thing."""
        return "ok"

    model = RecordingModel(ToolCall("1", "custom", {}))
    agent = await Agent.create(model, config=AgentRunConfig("custom"), extensions=[ToolExtension(custom)])

    await agent.run("go")

    [result] = [m for m in model.requests[1].messages if isinstance(m, ToolMessage)]
    assert result.content == "ok"


async def test_one_tool_extension_works_in_many_sessions_without_leaking_registrations():
    model = RecordingModel()
    extension = EditExtension()
    agent = await Agent.create(model, config=AgentRunConfig("one"), extensions=[extension])

    await agent.run("a")
    await agent.run("b", config=AgentRunConfig("two"))

    assert [[d.name for d in request.tools] for request in model.requests] == [["edit"], ["edit"]]
    assert agent.tools == {}


def test_every_builtin_tool_is_reachable_through_one_extension():
    exposed = [registered for extension_type, _ in SINGLE for registered in extension_type().tools]

    assert {id(registered) for registered in exposed} == {
        id(registered) for registered in (read, read_file, view_image, glob, grep, write_file, edit, delete_file, bash)
    }


async def test_the_read_tool_returns_an_image_through_the_agent_loop(tmp_path, monkeypatch):
    import base64

    from zett_agent.messages import ImageContent

    monkeypatch.chdir(tmp_path)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )
    (tmp_path / "pixel.png").write_bytes(png)
    (tmp_path / "note.txt").write_text("hello\n")
    model = RecordingModel(ToolCall("1", "read", {"path": "pixel.png"}), ToolCall("2", "read", {"path": "note.txt"}))
    agent = await Agent.create(model, config=AgentRunConfig("read"), extensions=[ReadExtension()])

    await agent.run("look")

    results = {m.tool_call_id: m for m in model.requests[1].messages if isinstance(m, ToolMessage)}
    assert results["1"].success and isinstance(results["1"].content[1], ImageContent)
    assert results["1"].content[1].source.data == png
    assert results["2"].content == "hello\n"
