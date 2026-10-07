"""Execute the actual downloadable tutorial files, without any API credentials."""

import ast
import asyncio
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "_examples"
EXPECTED = {
    "synchronous": "Synchronous history persisted",
    "first_agent": "Turn 2: Remember this conversation",
    "streaming_tools": "Answer: 42",
    "on_demand": "Skill loaded: code-review",
    "custom_extension": "Request state cleared",
    "approval": "Response accepted; duplicate rejected",
    "sessions": "Raw roles: system, user, assistant, system, user, assistant",
    "observability": "Reasoning and content boundaries recorded",
    "compaction": "Raw messages: 6; checkpoint boundary: 3; raw tail: 3",
    "subagent": "Two isolated sessions; child links to parent",
    "cancellation": "Cancelled request; extension state cleared",
    "goal": "Implementation and tests are complete.",
    "storage_adapter": "load, append:system, append:user, append:assistant",
    "project_instructions": "Project guidance: concise reviews",
    "images": "Received text and image in order.",
    "application": "Timeout cleaned up; session reused.",
    "mcp_tools": "MCP connection closed.",
    "provider_mock": "Mock provider reply: Hello from a mock endpoint.",
}


@pytest.mark.parametrize("name", EXPECTED)
def test_downloadable_example_runs_offline(name, tmp_path):
    script = EXAMPLES / f"{name}.py"
    runner = (
        "import runpy, socket, sys\n"
        "def no_network(*args, **kwargs):\n"
        "    raise AssertionError('Tutorial attempted a network connection')\n"
        "socket.create_connection = no_network\n"
        "socket.socket.connect = no_network\n"
        "runpy.run_path(sys.argv[1], run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", runner, str(script)], cwd=tmp_path, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert EXPECTED[name] in result.stdout


def test_every_example_is_classified_and_valid_python():
    assert {path.stem for path in EXAMPLES.glob("*.py")} == {*EXPECTED, "real_provider"}
    for path in EXAMPLES.glob("*.py"):
        ast.parse(path.read_text())


async def test_example_extension_isolates_concurrent_requests_and_cleans_errors():
    from zett_agent.agent import AgentRunConfig
    from zett_agent.client import create_agent

    example = runpy.run_path(str(EXAMPLES / "custom_extension.py"))
    extension = example["NoteExtension"]()
    clients = [
        await create_agent(example["NoteModel"](), config=AgentRunConfig(session_id=str(index)), extensions=[extension])
        for index in range(2)
    ]
    results = await asyncio.gather(*(client.run("/note extensions") for client in clients))
    assert all(result.content == "Extensions customize lifecycle hooks." for result in results)
    assert not extension.active
    # The teaching model deliberately rejects input not transformed by /note.
    with pytest.raises(AssertionError):
        await clients[0].run("invalid fixture input")
    assert not extension.active


@pytest.mark.parametrize("answer", [False, "not-a-boolean"])
async def test_approval_example_rejection_validation_and_routing(answer):
    from contextlib import aclosing

    from zett_agent.agent import AgentRunConfig
    from zett_agent.client import create_agent
    from zett_agent.events import AgentEventType
    from zett_agent.extensions.external import ExternalEvent

    example = runpy.run_path(str(EXAMPLES / "approval.py"))
    config = AgentRunConfig(session_id="approval-edge", request_id="request-edge")
    client = await create_agent(example["ApprovalModel"](), config=config, extensions=[example["ApprovalExtension"]()])
    observed = []
    async with aclosing(client.stream("Ask first")) as events:
        async for event in events:
            if event.type == AgentEventType.CUSTOM:
                response = ExternalEvent("approval.response", {"tool_call_id": "approval-1", "approved": answer})
                assert client.agent.emit_external_event(response, config=AgentRunConfig(session_id="other")) == []
                assert client.agent.emit_external_event(response, config=config) == ["ApprovalExtension"]
            if event.type in (AgentEventType.TOOL_COMPLETED, AgentEventType.TOOL_FAILED):
                observed.append(event)
                # Stop before the teaching model's success-only final step.
                break
    assert len(observed) == 1
    assert observed[0].type == (AgentEventType.TOOL_COMPLETED if answer is False else AgentEventType.TOOL_FAILED)
    assert (
        client.agent.emit_external_event(
            ExternalEvent("approval.response", {"tool_call_id": "approval-1", "approved": True}), config=config
        )
        == []
    )


@pytest.mark.parametrize("source", ["readme", "provider-example"])
def test_public_quickstarts_complete_an_offline_tool_round_trip(source, monkeypatch, capsys):
    """Execute public quickstarts through the actual provider and tool handler."""
    import json

    import httpx

    import zett_agent.providers.openai as openai_module

    original = openai_module.OpenAIProvider
    models = []
    requests = []

    def offline_provider(**options):
        model = original(**options, transport=httpx.MockTransport(respond))
        models.append(model)
        return model

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert any(item["function"]["name"] == "add" for item in payload["tools"])
        if payload["messages"][-1]["role"] == "tool":
            assert payload["messages"][-1]["content"] == "42"
            delta = {"content": "Offline README answer: 42"}
            finish_reason = "stop"
        else:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "readme-add",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"left":20,"right":22}'},
                    }
                ]
            }
            finish_reason = "tool_calls"
        chunks = [
            {"choices": [{"delta": delta}]},
            {"choices": [{"delta": {}, "finish_reason": finish_reason}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body + "data: [DONE]\n\n",
        )

    monkeypatch.setattr(openai_module, "OpenAIProvider", offline_provider)
    monkeypatch.setenv("OPENAI_MODEL", "offline-fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    readme = EXAMPLES.parents[1] / "README.md"
    programs = [code for code in re.findall(r"```python\n(.*?)```", readme.read_text(), re.S) if "asyncio.run(" in code]
    assert len(programs) == 1
    if source == "readme":
        exec(compile(programs[0], str(readme), "exec"), {"__name__": "__main__"})
    else:
        runpy.run_path(str(EXAMPLES / "real_provider.py"), run_name="__main__")
    assert "Offline README answer: 42" in capsys.readouterr().out
    assert len(requests) == 2
    assert models and all(model._http_client.is_closed for model in models)
