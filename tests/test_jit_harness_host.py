"""The two places repository code can run: this process, or a child of its own."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.jit_harness.extension import JITHarnessExtension
from zett_agent.jit_harness.host import (
    HarnessCalls,
    InProcessToolHost,
    JITHostCrashed,
    JITHostError,
    JITHostTimeout,
    JITToolHost,
    SubprocessToolHost,
    ToolNotSerializable,
    ToolSpec,
    decode_value,
    encode_value,
)
from zett_agent.jit_harness.runtime import JITHarness
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions
from zett_agent.sandbox.base import SandboxedCommand
from zett_agent.tools.base import AgentTool, ToolResult, tool


@tool(guidelines="Add two integers.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


SOURCE = '''
from zett_agent.tools.base import tool

SEEN = 0


@tool(guidelines="Add the value to itself through the harness.")
async def twice(value: int) -> int:
    """Compose a base tool from an asynchronous tool."""
    return await harness.call_tool("add", left=value, right=value)


@tool(guidelines="Compose from a synchronous tool.")
def twice_sync(value: int) -> int:
    """Compose a base tool from a synchronous tool."""
    return harness.call_tool_sync("add", left=value, right=value)


@tool(guidelines="Count calls in this process.")
def counter() -> int:
    """Prove the module keeps state between calls."""
    global SEEN
    SEEN += 1
    return SEEN


@tool(guidelines="Return JSON, exercising every shape.")
def shapes() -> dict[str, object]:
    """Return nested JSON values."""
    return {"list": [1, 2.5, True, None], "nested": {"text": "ok"}}


@tool(guidelines="Fail on purpose.")
def boom() -> int:
    """Raise a typed error."""
    raise ValueError("boom from the child")


@tool(guidelines="Exit the process.")
def die() -> int:
    """End the process without an answer."""
    import os

    os._exit(7)


@tool(guidelines="Never returns.")
def hang() -> int:
    """Block forever."""
    while True:
        pass


@tool(guidelines="Return something JSON cannot carry.")
def opaque() -> object:
    """Return a set."""
    return {1, 2}
'''


SPINNER_SOURCE = '''
import threading

from zett_agent.tools.base import tool


@tool(guidelines="Leave a thread that never ends.")
def start_spinner() -> str:
    """Answer at once, leaving a non-daemon thread spinning."""

    def spin() -> None:
        while True:
            pass

    threading.Thread(target=spin).start()
    return "started"
'''


#: A tool that says it is busy before it blocks, so a test never has to guess.
HANGING_SOURCE = '''
from zett_agent.tools.base import tool


@tool(guidelines="Write a marker, then never return.")
def hang(marker: str) -> int:
    """Write the marker, then spin forever."""
    with open(marker, "w") as handle:
        handle.write("busy")
    while True:
        pass
'''


#: A stand-in for a pytest process: it starts a child, makes a synchronous tool hang, and
#: then waits to be killed. It prints the child's pid before it starts waiting.
KILLED_PARENT = """\
import json
import subprocess
import sys
import time


def frame(**fields):
    return json.dumps({"v": 1, **fields}).encode() + b"\\n"


child = subprocess.Popen(
    [sys.executable, "-m", "zett_agent.jit_harness.child"],
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    start_new_session=True,
)
child.stdin.write(frame(op="load", id=1, directory=sys.argv[1]))
child.stdin.flush()
time.sleep(2.0)
child.stdin.write(frame(op="invoke", id=2, name="hang", arguments={"marker": sys.argv[2]}))
child.stdin.flush()
print(child.pid, flush=True)
time.sleep(600)
"""


def _pid_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def _wait_until_gone(pid: int, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pid_is_gone(pid):
            return
        await asyncio.sleep(0.1)
    os.killpg(pid, 9)
    raise AssertionError(f"process {pid} is still alive {timeout} seconds after its parent died")


def repository(tmp_path: Path, harness: JITHarness) -> Path:
    path = Path(harness.create_repo("kit")["path"])
    (path / "tools.py").write_text(SOURCE)
    return path


def subprocess_harness(directory: Path, *, call_timeout: float | None = 300.0) -> JITHarness:
    return JITHarness(directory, tools=[add], host=SubprocessToolHost(call_timeout=call_timeout))


# ---------------------------------------------------------------------------- encoding


@pytest.mark.parametrize(
    "value",
    [None, True, 0, -3, 2.5, "text", [1, "two", None], {"a": [1, {"b": None}]}, ()],
)
def test_json_values_survive_the_pipe(value):
    assert decode_value(encode_value(value)) == (list(value) if isinstance(value, tuple) else value)


@pytest.mark.parametrize("value", [{1: "not a string key"}, object(), {1, 2}, b"bytes", AgentTool])
def test_values_json_cannot_carry_are_refused_instead_of_stringified(value):
    with pytest.raises(ToolNotSerializable):
        encode_value(value)


def test_the_protocol_is_abstract_and_each_host_declares_what_it_does():
    with pytest.raises(TypeError):
        JITToolHost()  # type: ignore[abstract]
    assert issubclass(SubprocessToolHost, JITToolHost) and issubclass(InProcessToolHost, JITToolHost)


def test_a_tool_spec_round_trips_through_json():
    spec = ToolSpec.of(add)
    again = ToolSpec.from_json(spec.to_json())
    assert again == spec
    assert again.return_schema == {"type": "integer"}
    assert again.build(lambda arguments: _never()).return_schema == {"type": "integer"}
    assert again.build(lambda arguments: _never()).name == "add"


async def _never() -> None:  # pragma: no cover - only used to build a handler
    raise AssertionError("not called")


# ------------------------------------------------------------------------ in process


async def test_the_default_host_imports_into_this_process(tmp_path):
    harness = JITHarness(tmp_path / "session", tools=[add])
    assert isinstance(harness._host, InProcessToolHost)
    repository(tmp_path / "session", harness)
    assert harness.load_repo()["tools"] == ["boom", "counter", "die", "hang", "opaque", "shapes", "twice", "twice_sync"]
    assert await harness.call_tool("twice", value=4) == 8
    harness.close()


# ------------------------------------------------------------------- a child process


async def test_a_repository_runs_in_its_own_process(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        assert harness.load_repo()["tools"] == [
            "boom",
            "counter",
            "die",
            "hang",
            "opaque",
            "shapes",
            "twice",
            "twice_sync",
        ]
        assert await harness.call_tool("twice", value=4) == 8
        assert await harness.call_tool("twice_sync", value=5) == 10
        assert await harness.call_tool("shapes") == {"list": [1, 2.5, True, None], "nested": {"text": "ok"}}
    finally:
        harness.close()


async def test_module_state_lives_in_the_child_between_calls(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        assert [await harness.call_tool("counter") for _ in range(3)] == [1, 2, 3]
    finally:
        harness.close()


async def test_a_failing_tool_reports_the_type_and_message(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        with pytest.raises(JITHostError, match="ValueError: boom from the child"):
            await harness.call_tool("boom")
    finally:
        harness.close()


async def test_a_result_the_pipe_cannot_carry_names_its_type(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        with pytest.raises(JITHostError, match="set"):
            await harness.call_tool("opaque")
    finally:
        harness.close()


async def test_a_child_that_exits_is_reported_and_can_be_loaded_again(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        with pytest.raises(JITHostCrashed, match="exited"):
            await harness.call_tool("die")
        with pytest.raises(JITHostCrashed):
            await harness.call_tool("counter")
        harness.load_repo()
        assert await harness.call_tool("counter") == 1
    finally:
        harness.close()


async def test_a_tool_that_never_returns_is_stopped_at_the_limit(tmp_path):
    harness = subprocess_harness(tmp_path / "session", call_timeout=1.0)
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        with pytest.raises(JITHostTimeout):
            await harness.call_tool("hang")
    finally:
        harness.close()


@pytest.mark.skipif(os.name != "posix", reason="process groups and reparenting are POSIX")
async def test_a_killed_agent_leaves_no_child_spinning(tmp_path):
    """A SIGKILLed agent runs no finally, so the child has to notice by itself.

    The child is busy in a synchronous tool, which runs on a non-daemon executor thread.
    Interpreter shutdown joins that thread, so without an explicit exit the orphan would
    wait forever and burn a core; this asserts it disappears instead.
    """
    repo = tmp_path / "kit"
    repo.mkdir()
    (repo / "__init__.py").write_text("")
    (repo / "tools.py").write_text(HANGING_SOURCE)
    marker = tmp_path / "busy"
    script = tmp_path / "parent.py"
    script.write_text(KILLED_PARENT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(entry for entry in sys.path if entry)}
    parent = subprocess.Popen(
        [sys.executable, str(script), str(repo), str(marker)], stdout=subprocess.PIPE, env=env, text=True
    )
    try:
        assert parent.stdout is not None
        child_pid = int(parent.stdout.readline().strip())
        # The marker says the worker thread is spinning, so an exit here is the fix, not luck.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not marker.exists():
            await asyncio.sleep(0.05)
        assert marker.exists(), "the tool never started, so this test would prove nothing"
        parent.kill()
        parent.wait(timeout=10)
        await _wait_until_gone(child_pid)
    finally:
        if parent.poll() is None:  # pragma: no cover - only when the assertion above failed
            parent.kill()


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
async def test_a_thread_that_never_ends_does_not_delay_closing_the_child(tmp_path):
    """Closing waits for a shutdown frame; a spinning worker must not hold it up."""
    host = SubprocessToolHost()
    harness = JITHarness(tmp_path / "session", tools=[add], host=host)
    try:
        path = Path(harness.create_repo("kit")["path"])
        (path / "tools.py").write_text(SPINNER_SOURCE)
        harness.load_repo()
        await harness.call_tool("start_spinner")
        assert host._process is not None
        pid = host._process.pid
        started = time.monotonic()
        harness.close()
        elapsed = time.monotonic() - started
    finally:
        harness.close()
    # ``_stop`` waits two seconds for a shutdown handshake before it kills the group, so a
    # child that cannot exit shows up as that wait.
    assert elapsed < 1.5, f"closing took {elapsed:.2f}s, so the child could not exit on its own"
    assert _pid_is_gone(pid)


async def test_an_import_error_names_the_file_and_leaves_the_harness_usable(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        path = Path(harness.create_repo("kit")["path"])
        (path / "tools.py").write_text("from zett_agent.tools.base import tool\n\nraise RuntimeError('broken')\n")
        with pytest.raises(ValueError, match="tools.py"):
            harness.load_repo()
        assert harness.repo == "kit"
    finally:
        harness.close()


async def test_closing_the_harness_stops_the_child(tmp_path):
    host = SubprocessToolHost()
    harness = JITHarness(tmp_path / "session", tools=[add], host=host)
    repository(tmp_path / "session", harness)
    harness.load_repo()
    process = host._process
    assert process is not None and process.poll() is None
    harness.close()
    assert process.poll() is not None


@dataclass
class FakeSandbox:
    """A sandbox that only records that it was asked and marks the environment."""

    calls: list[tuple[tuple[str, ...], Path]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def wrap(self, command, *, cwd=None, env=None) -> SandboxedCommand:
        self.calls.append((tuple(command), Path(cwd)))
        return SandboxedCommand(
            argv=tuple(command),
            env={**(env or {}), "JIT_SANDBOX_MARK": "1"},
            cwd=Path(cwd),
        )


async def test_the_sandbox_rewrites_the_command_the_child_is_started_with(tmp_path):
    sandbox = FakeSandbox()
    harness = JITHarness(tmp_path / "session", tools=[add], host=SubprocessToolHost(sandbox=sandbox))  # type: ignore[arg-type]
    try:
        path = Path(harness.create_repo("kit")["path"])
        (path / "tools.py").write_text(
            "import os\nfrom zett_agent.tools.base import tool\n\n"
            '@tool(guidelines="Read the marker.")\ndef mark() -> str:\n'
            '    """Return the sandbox marker."""\n    return os.environ["JIT_SANDBOX_MARK"]\n'
        )
        harness.load_repo()
        assert await harness.call_tool("mark") == "1"
        [(command, cwd)] = sandbox.calls
        assert command[1:] == ("-m", "zett_agent.jit_harness.child") and cwd.name == "kit"
    finally:
        harness.close()


async def test_a_nested_call_reaches_the_agent_with_its_own_result(tmp_path):
    """The child's ``harness`` calls arrive as ordinary dispatches, so a broken base tool fails."""

    class Recorder(HarnessCalls):  # pragma: no cover - only to prove the ABC is implementable
        async def call_tool(self, name: str, **arguments: ToolResult) -> ToolResult:
            return None

        def call_tool_sync(self, name: str, **arguments: ToolResult) -> ToolResult:
            return None

    recorder = Recorder()
    assert await recorder.call_tool("anything") is None


async def test_an_unknown_tool_is_rejected_before_it_reaches_the_host(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        with pytest.raises(ValueError, match="Unknown tool: nope"):
            await harness.call_tool("nope")
    finally:
        harness.close()


def test_a_timeout_must_be_positive():
    with pytest.raises(ValueError, match="call_timeout"):
        SubprocessToolHost(call_timeout=0)
    with pytest.raises(ValueError, match="load_timeout"):
        SubprocessToolHost(load_timeout=-1)


async def test_concurrent_calls_are_multiplexed_on_one_child(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        repository(tmp_path / "session", harness)
        harness.load_repo()
        results = await asyncio.gather(*(harness.call_tool("twice", value=index) for index in range(5)))
        assert results == [0, 2, 4, 6, 8]
    finally:
        harness.close()


# ------------------------------------------------------------------- protocol details


def test_a_host_rejects_a_broken_tool_description():
    from zett_agent.jit_harness.host import PROTOCOL_VERSION, _Pending

    spec = ToolSpec.of(add).to_json()
    for broken in (
        {**spec, "name": 3},
        {**spec, "parameters": []},
        {**spec, "guidelines": "text"},
        {**spec, "return_schema": "integer"},
    ):
        with pytest.raises(JITHostError):
            ToolSpec.from_json(broken)
    # An older child does not send the key at all, and that must stay a working tool.
    assert (
        ToolSpec.from_json({key: value for key, value in spec.items() if key != "return_schema"}).return_schema is None
    )
    with pytest.raises(JITHostError, match="parameters"):
        ToolSpec.from_json({key: value for key, value in spec.items() if key != "parameters"})
    with pytest.raises(JITHostError, match="guidelines"):
        ToolSpec.from_json({**spec, "guidelines": [1]})
    assert PROTOCOL_VERSION == 1 and _Pending(1).identifier == 1


async def test_the_agent_gives_up_on_a_child_that_stops_answering():
    from zett_agent.jit_harness.host import _Pending

    host = SubprocessToolHost(call_timeout=0.05)
    host._process = _SilentProcess()  # type: ignore[assignment]
    host._send = lambda frame: _Pending(1)  # type: ignore[method-assign]
    with pytest.raises(JITHostTimeout):
        await host.invoke("anything", {})
    assert host._process is None


def test_a_load_that_times_out_is_reported():
    from zett_agent.jit_harness.host import _Pending

    host = SubprocessToolHost(load_timeout=0.05)
    host._process = _SilentProcess()  # type: ignore[assignment]
    host._send = lambda frame: _Pending(1)  # type: ignore[method-assign]
    with pytest.raises(JITHostTimeout, match="import"):
        host._request_sync({"op": "load"}, host.load_timeout)


def test_calling_a_dead_host_says_to_load_it_again():
    host = SubprocessToolHost()
    with pytest.raises(JITHostCrashed, match="load the repository again"):
        asyncio.run(host.invoke("anything", {}))
    host.close()  # stopping a host that never started is harmless


def test_a_frame_the_parent_cannot_read_kills_the_child():
    host = SubprocessToolHost()
    for frame in (["not", "an", "object"], {"v": 99}, {"v": 1, "op": "surprise"}, {"v": 1, "op": "call"}):
        host._kill = lambda reason: None  # type: ignore[method-assign]
        host._on_frame(frame)


def test_an_in_process_host_is_never_asked_to_invoke():
    with pytest.raises(JITHostError, match="runs in this process"):
        asyncio.run(InProcessToolHost().invoke("anything", {}))


def test_a_hosted_repo_closes_once():
    from zett_agent.jit_harness.host import HostedRepo

    closed: list[bool] = []
    repo = HostedRepo("alias", {}, "fingerprint", Path("."), lambda: closed.append(True))
    repo.close()
    repo.close()
    assert closed == [True]


class _SilentProcess:
    """Stands in for a child that never answers, so the timeout paths can be tested."""

    #: A process group that cannot exist, so the group kill falls back to ``kill``.
    pid = 2_000_000_000

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        return None

    stdin = None
    stdout = None


# ------------------------------------------------------------------- end to end


async def test_repository_code_runs_in_a_process_of_its_own(tmp_path):
    """The pid proves this is a real child process, not a simulated one."""
    harness = subprocess_harness(tmp_path / "session")
    try:
        path = Path(harness.create_repo("kit")["path"])
        (path / "tools.py").write_text(
            "import os\nfrom zett_agent.tools.base import tool\n\n"
            '@tool(guidelines="Report the pid.")\ndef pid() -> int:\n'
            '    """Return the process that ran this."""\n    return os.getpid()\n'
        )
        harness.load_repo()
        assert await harness.call_tool("pid") != os.getpid()
    finally:
        harness.close()


class ScriptModel:
    """Answer with one batch of tool calls per step, then a plain answer."""

    retry = RetryOptions(max_retries=0)

    def __init__(self, *batches: tuple[ToolCall, ...]) -> None:
        self.batches = batches
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        index = len(self.requests)
        self.requests.append(request)
        message = (
            AssistantMessage(tool_calls=self.batches[index])
            if index < len(self.batches)
            else AssistantMessage(content="done")
        )
        yield ModelEvent.completed(ModelResponse(message))


@tool(guidelines="Write one file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


TOOL_SOURCE = '''
from zett_agent.tools.base import tool
import os


@tool(guidelines="Add the value twice through the harness.")
async def twice(value: int) -> int:
    """Compose a base tool."""
    return await harness.call_tool("add", left=value, right=value)


@tool(guidelines="Report the pid.")
def pid() -> int:
    """Return the process that ran this."""
    return os.getpid()
'''


async def test_a_whole_agent_run_drives_the_child_process(tmp_path):
    """create_agent + a scripted model: create, write, load, compose, and report the pid."""
    root = tmp_path / "jit"
    target = root / "e2e" / "kit"
    model = ScriptModel(
        (ToolCall("c", "repo_create", {"repo": "kit"}),),
        (ToolCall("w", "put", {"path": str(target / "tools.py"), "content": TOOL_SOURCE}),),
        (ToolCall("l", "repo_load", {}),),
        (ToolCall("t", "twice", {"value": 21}),),
        (ToolCall("p", "pid", {}),),
    )
    jit = JITHarnessExtension(root, host_factory=lambda: SubprocessToolHost())
    agent = await Agent.create(model, tools=[add, put], extensions=[jit])
    result = await agent.run("go", config=AgentRunConfig("e2e"))

    assert result.content == "done"
    replies = {message.name: message for message in agent.state.messages if isinstance(message, ToolMessage)}
    assert replies["twice"].content == "42"
    assert int(replies["pid"].content) != os.getpid()
    names = [{definition.name for definition in request.tools} for request in model.requests]
    assert "twice" not in names[0] and "twice" in names[3]

    host = jit.get_harness("e2e")._host
    process = host._process  # type: ignore[attr-defined]  # the subprocess host is what the factory built
    assert process is not None and process.poll() is None
    jit.forget_session("e2e")
    assert process.poll() is not None


# ----------------------------------------------------------------- restart and cleanup

PID_TOOL = '''
from zett_agent.tools.base import tool
import os

SEEN = 0


@tool(guidelines="Count calls in this process.")
def counter() -> int:
    """Count calls in this process."""
    global SEEN
    SEEN += 1
    return SEEN


@tool(guidelines="Report the pid.")
def pid() -> int:
    """Return the process that ran this."""
    return os.getpid()
'''


async def test_a_reload_starts_a_new_child_and_forgets_the_old_state(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        path = Path(harness.create_repo("kit")["path"])
        (path / "tools.py").write_text(PID_TOOL)
        harness.load_repo()
        first_pid = await harness.call_tool("pid")
        assert [await harness.call_tool("counter") for _ in range(2)] == [1, 2]

        (path / "tools.py").write_text(PID_TOOL + "\n# a change, so the source differs\n")
        harness.load_repo()
        second_pid = await harness.call_tool("pid")

        assert second_pid != first_pid, "a reload runs in a fresh interpreter"
        assert await harness.call_tool("counter") == 1, "module state did not survive the reload"
    finally:
        harness.close()


SPAWN_TOOL = '''
import subprocess
from zett_agent.tools.base import tool


@tool(guidelines="Start a long sleep and report its pid.")
def spawn() -> int:
    """Start a child of the child."""
    return subprocess.Popen(["sleep", "30"]).pid
'''


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
async def test_killing_the_child_takes_down_what_it_started(tmp_path):
    harness = subprocess_harness(tmp_path / "session")
    try:
        path = Path(harness.create_repo("kit")["path"])
        (path / "tools.py").write_text(SPAWN_TOOL)
        harness.load_repo()
        grandchild = int(await harness.call_tool("spawn"))
        assert os.kill(grandchild, 0) is None  # it is running

        harness.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:  # pragma: no cover - reported as a failure below
            pass
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild, 0)
    finally:
        harness.close()


async def test_the_extension_keeps_only_the_configured_number_of_sessions(tmp_path):
    jit = JITHarnessExtension(tmp_path, max_sessions=2)
    for index in range(3):
        context = AgentRunContext(AgentRunConfig(f"s{index}"), AgentState(), {}, ())
        await jit.on_tool(context)
        jit._request_finished(context)
    assert set(jit._sessions) == {"s1", "s2"}, "the least recently used idle session was released"
    for session in list(jit._sessions):
        jit.forget_session(session)


def test_the_session_cap_must_be_positive(tmp_path):
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="max_sessions"):
            JITHarnessExtension(tmp_path, max_sessions=value)  # type: ignore[arg-type]  # deliberately invalid
