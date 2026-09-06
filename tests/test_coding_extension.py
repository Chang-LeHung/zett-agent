"""Coding extension integration with model schemas and tool dispatch."""

import json

from kcs_agent import (
    Agent,
    AgentConfig,
    AssistantMessage,
    CodingExtension,
    ModelEvent,
    ModelResponse,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
)


async def test_coding_extension_executes_tools_and_registers_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = (
        ToolCall("write", "write_file", {"path": "note.txt", "content": "hello"}),
        ToolCall("replace", "replace_in_file", {"path": "note.txt", "old_text": "hello", "new_text": "world"}),
        ToolCall("read", "read_file", {"path": "note.txt"}),
        ToolCall("glob", "glob", {"pattern": "*.txt"}),
        ToolCall("grep", "grep", {"pattern": "world"}),
        ToolCall("shell", "run_shell", {"command": "printf coding-extension"}),
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
        config=AgentConfig("files"),
        extensions=[ToolGuidelinesExtension(), CodingExtension()],
    )
    await agent.run("Edit and search a file")
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
    assert (tmp_path / "note.txt").read_text() == "world"
    assert all(message.success for message in model.requests[1].messages if isinstance(message, ToolMessage))

    await agent.run("Another request", config=AgentConfig("other-files"))
    for request in model.requests:
        assert {definition.name for definition in request.tools} == {call.name for call in calls}
        guidance = "\n".join(message.content for message in request.messages if isinstance(message, SystemMessage))
        assert all(f"## {call.name}" in guidance for call in calls)
    assert agent.tools == {}
