"""Coding extension integration with model schemas and tool dispatch."""

import json
from pathlib import Path

from zett_agent import (
    Agent,
    AgentRunConfig,
    AssistantMessage,
    CodingExtension,
    FileSystemExtension,
    ModelEvent,
    ModelResponse,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
)


async def test_filesystem_extension_selects_tools_from_read_only_mode():
    class Model:
        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    expected = {
        True: {"read_file", "glob", "grep"},
        False: {"read_file", "glob", "grep", "write_file", "replace_in_file", "delete_file"},
    }
    for read_only, tool_names in expected.items():
        model = Model()
        agent = await Agent.create(
            model,
            config=AgentRunConfig(f"filesystem-{read_only}"),
            extensions=[FileSystemExtension(read_only=read_only), ToolGuidelinesExtension()],
        )

        await agent.run("Inspect the workspace")

        assert {definition.name for definition in model.requests[0].tools} == tool_names
        assert "run_shell" not in tool_names
        guidance = "\n".join(
            message.content for message in model.requests[0].messages if isinstance(message, SystemMessage)
        )
        assert f'Current working directory: "{Path.cwd().resolve()}".' in guidance
        assert "Relative filesystem paths are resolved from this directory" in guidance
        assert all(f"## {name}" in guidance for name in tool_names)
        if read_only:
            assert "## write_file" not in guidance
            assert "## replace_in_file" not in guidance
            assert "## delete_file" not in guidance


async def test_coding_extension_executes_tools_and_registers_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = (
        ToolCall("write", "write_file", {"path": "note.txt", "content": "hello"}),
        ToolCall("replace", "replace_in_file", {"path": "note.txt", "old_text": "hello", "new_text": "world"}),
        ToolCall("read", "read_file", {"path": "note.txt"}),
        ToolCall("glob", "glob", {"pattern": "*.txt"}),
        ToolCall("grep", "grep", {"pattern": "world"}),
        ToolCall("shell", "run_shell", {"command": "printf coding-extension"}),
        ToolCall("delete", "delete_file", {"path": "note.txt"}),
    )

    class Model:
        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(tool_calls=calls) if len(self.requests) == 1 else AssistantMessage(content="done")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("files"),
        extensions=[ToolGuidelinesExtension(), CodingExtension()],
        parallel_tool_call=False,
    )
    await agent.run("Edit and search a file")
    environment_messages = [
        message
        for message in model.requests[0].messages
        if isinstance(message, SystemMessage) and "# Filesystem environment" in message.content
    ]
    assert len(environment_messages) == 1
    assert f'Current working directory: "{tmp_path}".' in environment_messages[0].content
    assert "Shell commands run from this directory." in environment_messages[0].content
    results = {
        message.name: json.loads(message.content)
        for message in model.requests[1].messages
        if isinstance(message, ToolMessage)
    }
    assert results["read_file"]["content"] == "world"
    assert results["run_shell"]["stdout"] == "coding-extension"
    assert results["run_shell"]["exit_code"] == 0
    assert results["replace_in_file"]["replacements"] == 1
    assert results["glob"]["paths"] == ["note.txt"]
    assert len(results["grep"]["matches"]) == 1
    assert results["delete_file"] == {"path": "note.txt", "deleted": True}
    assert not (tmp_path / "note.txt").exists()
    assert all(message.success for message in model.requests[1].messages if isinstance(message, ToolMessage))

    await agent.run("Another request", config=AgentRunConfig("other-files"))
    for request in model.requests:
        assert {definition.name for definition in request.tools} == {call.name for call in calls}
        guidance = "\n".join(message.content for message in request.messages if isinstance(message, SystemMessage))
        assert all(f"## {call.name}" in guidance for call in calls)
    assert agent.tools == {}
