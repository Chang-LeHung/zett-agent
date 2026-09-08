"""Execute the actual downloadable tutorial files, without any API credentials."""

import ast
import asyncio
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "_examples"
EXPECTED = {
    "first_agent": "Turn 2: Remember this conversation",
    "streaming_tools": "Answer: 42",
    "custom_extension": "Request state cleared",
    "approval": "Response accepted; duplicate rejected",
    "sessions": "Raw roles: user, assistant, user, assistant",
    "observability": "Reasoning and content boundaries recorded",
    "compaction": "Raw messages: 4; checkpoint boundary: 2; raw tail: 2",
    "subagent": "Two isolated sessions; child links to parent",
    "cancellation": "Cancelled request; extension state cleared",
    "goal": "Implementation and tests are complete.",
    "storage_adapter": "load, append:user, append:assistant",
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
    from zett_agent import AgentConfig, create_agent

    example = runpy.run_path(str(EXAMPLES / "custom_extension.py"))
    extension = example["NoteExtension"]()
    clients = [
        await create_agent(example["NoteModel"](), config=AgentConfig(session_id=str(index)), extensions=[extension])
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

    from zett_agent import AgentConfig, AgentEventType, ExternalEvent, create_agent

    example = runpy.run_path(str(EXAMPLES / "approval.py"))
    config = AgentConfig(session_id="approval-edge", request_id="request-edge")
    client = await create_agent(example["ApprovalModel"](), config=config, extensions=[example["ApprovalExtension"]()])
    observed = []
    async with aclosing(client.stream("Ask first")) as events:
        async for event in events:
            if event.type == AgentEventType.CUSTOM:
                response = ExternalEvent("approval.response", {"tool_call_id": "approval-1", "approved": answer})
                assert client.agent.emit_external_event(response, config=AgentConfig(session_id="other")) == []
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
