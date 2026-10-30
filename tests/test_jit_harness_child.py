"""The child server, driven in-process so its frames are exercised without a spawn."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from zett_agent.jit_harness import child
from zett_agent.jit_harness.child import HarnessProxy, _quiet_handler, _Server, _Waiter
from zett_agent.jit_harness.host import PROTOCOL_VERSION, JITHostError
from zett_agent.jit_harness.repo import discard
from zett_agent.json_types import JsonValue

_SERVERS: list[_Server] = []


def make_server(*frames: dict[str, JsonValue]) -> tuple[_Server, io.BytesIO]:
    output = io.BytesIO()
    source = io.BytesIO(b"".join(json.dumps(frame).encode() + b"\n" for frame in frames))
    server = _Server(output, source)
    _SERVERS.append(server)
    return server, output


@pytest.fixture(autouse=True)
def no_residue():
    """A child server imports into this process in these tests, so drop what it imported."""
    yield
    for server in _SERVERS:
        if server._repo is not None:
            server._repo.close()
            server._repo = None
        if not server._loop.is_closed():
            server._loop.close()
    _SERVERS.clear()
    # ``read_forever`` may have imported a package whose server was replaced, and a
    # half-finished import can leave modules behind; discard every alias that is left.
    for alias in sorted({name.split(".")[0] for name in sys.modules if name.startswith("zett_jit_")}):
        discard(alias)


def replies(output: io.BytesIO) -> list[dict[str, JsonValue]]:
    return [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "kit"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "MANIFEST").write_text('name = "kit"\nversion = "0.1.0"\ndescription = "Doubling."\n')
    (root / "tools.py").write_text(
        "from zett_agent.tools.base import tool\n\n"
        '@tool(guidelines="Double it.")\ndef double(value: int) -> int:\n'
        '    """Double an integer."""\n    return value * 2\n'
    )
    return root


def test_the_child_loads_a_repository_and_runs_a_tool(tmp_path):
    server, output = make_server()
    root = repository(tmp_path)
    server._loop.run_until_complete(server._load(1, str(root)))
    [loaded] = replies(output)
    assert loaded["reply_to"] == 1 and loaded["ok"] is True
    assert [tool["name"] for tool in loaded["tools"]] == ["double"]

    server._loop.run_until_complete(server._invoke(2, {"name": "double", "arguments": {"value": 21}}))
    assert replies(output)[-1] == {"reply_to": 2, "ok": True, "value": 42, "v": PROTOCOL_VERSION}


def test_a_tool_error_comes_back_as_a_message_not_a_crash(tmp_path):
    server, output = make_server()
    server._loop.run_until_complete(server._load(1, str(repository(tmp_path))))
    server._loop.run_until_complete(server._invoke(2, {"name": "missing", "arguments": {}}))
    failure = replies(output)[-1]
    assert failure["ok"] is False and "Unknown tool: missing" in failure["error"]
    server._repo = None
    server._loop.run_until_complete(server._invoke(3, {"name": "double", "arguments": {"value": 1}}))
    assert "No repository is loaded" in replies(output)[-1]["error"]


def test_a_broken_import_is_reported_as_a_load_failure(tmp_path):
    server, output = make_server()
    root = repository(tmp_path)
    (root / "tools.py").write_text("raise RuntimeError('broken at import')\n")
    server._loop.run_until_complete(server._load(1, str(root)))
    [failure] = replies(output)
    assert failure["ok"] is False and failure["kind"] == "load" and "broken at import" in failure["error"]


def test_frames_are_read_until_the_agent_stops_the_child(tmp_path):
    server, output = make_server(
        {"v": PROTOCOL_VERSION, "id": 1, "op": "load", "directory": str(repository(tmp_path))},
        {"v": PROTOCOL_VERSION, "op": "shutdown"},
    )


def test_a_load_frame_is_answered_and_a_shutdown_frame_stops_the_child(tmp_path):
    server, output = make_server()
    server._dispatch({"v": PROTOCOL_VERSION, "id": 1, "op": "load", "directory": str(repository(tmp_path))})
    deadline = time.monotonic() + 5
    while not replies(output) and time.monotonic() < deadline:
        server._loop.run_until_complete(asyncio.sleep(0.02))
    assert replies(output)[0]["ok"] is True

    server._dispatch({"v": PROTOCOL_VERSION, "op": "shutdown"})
    assert server._stopping.is_set()


def test_reading_stops_at_the_end_of_the_pipe():
    server, _ = make_server()
    server.read_forever()
    assert server._stopping.is_set()


@pytest.mark.parametrize("frame", [b"not json\n", b'{"v": 99}\n', b'["a"]\n'])
def test_a_frame_that_is_not_a_protocol_message_stops_reading(frame):
    server, _ = make_server()
    server._source = io.BytesIO(frame)
    server.read_forever()
    assert server._stopping.is_set()


def test_a_nested_call_waits_for_the_agent_and_decodes_the_answer():
    server, output = make_server()
    loop = server._loop
    loop.call_later(0.05, server._dispatch, {"v": PROTOCOL_VERSION, "reply_to": 1, "ok": True, "value": [1, 2]})
    assert loop.run_until_complete(server.call("double", {"value": 2})) == [1, 2]
    [sent] = replies(output)
    assert sent["op"] == "call" and sent["arguments"] == {"value": 2}


def test_a_synchronous_nested_call_blocks_its_worker_until_answered():
    server, _ = make_server()
    values: list[JsonValue] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            values.append(server.call_sync("double", {"value": 3}))
        except BaseException as error:  # pragma: no cover - a failed call is a test failure
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 2
    while server._next_id == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    server._dispatch({"v": PROTOCOL_VERSION, "reply_to": 1, "ok": True, "value": 6})
    thread.join(2)
    assert values == [6] and errors == []


def test_a_nested_call_reports_what_the_agent_said():
    server, _ = make_server()
    loop = server._loop
    loop.call_later(0.05, server._dispatch, {"v": PROTOCOL_VERSION, "reply_to": 1, "ok": False, "error": "nope"})
    with pytest.raises(JITHostError, match="nope"):
        loop.run_until_complete(server.call("double", {"value": 2}))


def test_the_proxy_refuses_calls_whenever_the_protocol_says_so():
    server, _ = make_server()
    proxy = HarnessProxy(server)
    with pytest.raises(RuntimeError, match="loading Python source"):
        server._loop.run_until_complete(proxy.call_tool("double", value=1))
    with pytest.raises(RuntimeError, match="loading Python source"):
        proxy.call_tool_sync("double", value=1)
    proxy.ready()
    with pytest.raises(RuntimeError, match="cannot block the child's event loop"):
        proxy.call_tool_sync("double", value=1)


def test_a_waiter_carries_a_value_or_an_error():
    loop = asyncio.new_event_loop()
    try:
        waiter = _Waiter(loop)
        waiter.succeed(5)
        assert waiter.event.is_set() and waiter.value == 5
        loop.run_until_complete(asyncio.sleep(0))
        assert waiter.future.result() == 5

        failed = _Waiter(loop)
        failed.fail(JITHostError("gone"))
        loop.run_until_complete(asyncio.sleep(0))
        assert isinstance(failed.error, JITHostError)
        assert failed.future.exception() is not None
    finally:
        loop.close()


def test_loop_errors_are_reported_instead_of_raised(capsys):
    loop = asyncio.new_event_loop()
    try:
        _quiet_handler(loop, {"exception": RuntimeError("boom")})
        _quiet_handler(loop, {"message": "plain message"})
    finally:
        loop.close()
    captured = capsys.readouterr()
    assert "boom" in captured.err and "plain message" in captured.err


def test_the_child_ends_when_its_parent_is_no_longer_there(monkeypatch):
    """The pipe can stay open in someone else's hands, so the parent pid decides."""
    server, _ = make_server()
    parents = iter([100, 100, 200])
    monkeypatch.setattr(child.os, "getppid", lambda: next(parents, 200))
    exits: list[int] = []
    monkeypatch.setattr(server, "exit_now", lambda: (exits.append(1), server._stopping.set()))

    server.watch_parent(interval=0.001)

    assert exits == [1]


def test_a_bad_frame_reports_a_failed_status():
    server, _ = make_server()
    server._source = [b"not json\n"]  # type: ignore[list-item]  # a stand-in for the pipe

    server.read_forever()

    assert server.exit_status == 1


def test_a_stop_the_agent_asked_for_reports_success():
    server, _ = make_server()

    server.read_forever()

    assert server.exit_status == 0
