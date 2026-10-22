"""Where a repository's Python runs, and how its tools reach the agent.

A :class:`JITToolHost` answers two questions: which tools a repository defines
(:meth:`JITToolHost.load`) and what running one returns
(:meth:`JITToolHost.invoke`). Two hosts implement it:

* :class:`InProcessToolHost` imports the package into the agent's interpreter, which is
  fast and shared, and gives repository code the application's whole process.
* :class:`SubprocessToolHost` imports and runs it in a child process of its own, so a
  crash, a hang, or a memory blow-up stays there. Code that calls ``harness.call_tool``
  is answered by the agent over a pipe, and takes the normal path through extensions,
  events, and profiling.

Every host runs repository code with the same operating-system identity. A child
process is not, by itself, a security boundary; pass a
:class:`~zett_agent.sandbox.base.Sandbox` to :class:`SubprocessToolHost` to also bound
what the child may read, write, and connect to.

The pipe carries one request per line in each direction. A tool call is two round
trips, because the child may ask the agent to run a loaded tool while it runs:

.. code-block:: text

    agent process                                child process
         |                                            |
         |  spawn: python -m zett_agent.jit_harness.child
         |------------------------------------------->|
         |  {"op":"load","directory":"/.../kit"}        |
         |------------------------------------------->|  import the package
         |  {"reply_to":1,"ok":true,"tools":[...]}      |  collect @tool
         |<-------------------------------------------|
         |                                            |
         |  {"op":"invoke","name":"twice","value":21}  |
         |------------------------------------------->|  await twice(21)
         |      {"op":"call","name":"add","arguments"...}
         |<-------------------------------------------|  harness.call_tool
         |  agent runs "add": hooks, events, profile    |
         |  {"reply_to":2,"ok":true,"value":42}        |
         |------------------------------------------->|  twice returns
         |  {"reply_to":1,"ok":true,"value":42}        |
         |<-------------------------------------------|

Frames are JSON with ``v``, an ``id`` on each request, and ``reply_to`` on each answer;
the two directions keep separate id spaces, so neither side can answer its own
request. Values cross as JSON: ``None``, booleans, numbers, strings, lists, and
dictionaries of those.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..json_types import JsonValue
from ..tools.base import AgentTool, ToolExecutionMode, ToolResult
from .repo import LoadedRepo, RepoScan, load_repo

if TYPE_CHECKING:
    from ..sandbox.base import Sandbox

#: Protocol version of the parent/child frames. Both sides refuse anything else, so an
#: installed child can never be paired with an incompatible parent.
PROTOCOL_VERSION = 1
#: Longest line the parent accepts from a child, covering a large image in one result.
MAX_FRAME_BYTES = 64 * 1024 * 1024


class JITHostError(RuntimeError):
    """A repository host could not do what was asked of it."""


class JITHostCrashed(JITHostError):
    """The child process that ran a repository is gone."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class JITHostTimeout(JITHostError):
    """The child process did not answer within the configured limit."""


class ToolNotSerializable(JITHostError, TypeError):
    """A tool returned a value that cannot cross a process boundary."""

    def __init__(self, value: object) -> None:
        super().__init__(
            f"A repository tool returned {type(value).__name__}, which cannot leave the child process. "
            "Return JSON values, TextContent, ImageContent, or a list of those content parts."
        )


class HarnessCalls(ABC):
    """What repository code may call on ``harness``.

    :class:`~zett_agent.jit_harness.runtime.JITHarness` implements it for code running in
    the agent's process, and the child's proxy implements it by writing to the pipe.
    """

    @abstractmethod
    async def call_tool(self, name: str, **arguments: ToolResult) -> ToolResult:
        """Run a loaded tool from an asynchronous repository tool."""

    @abstractmethod
    def call_tool_sync(self, name: str, **arguments: ToolResult) -> ToolResult:
        """Run a loaded tool from a synchronous repository tool."""


@dataclass(frozen=True)
class ToolSpec:
    """Everything the agent needs to know about a repository tool before calling it.

    It is the serializable half of an :class:`~zett_agent.tools.base.AgentTool`: the
    schema and the prompt-facing text, without the handler that only lives where the
    code runs.
    """

    name: str
    description: str
    parameters: dict[str, JsonValue]
    guidelines: tuple[str, ...]
    snippet: str
    execution_mode: ToolExecutionMode

    @classmethod
    def of(cls, tool: AgentTool) -> ToolSpec:
        """Describe one imported tool."""
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=cast("dict[str, JsonValue]", json.loads(json.dumps(tool.parameters))),
            guidelines=tuple(tool.guidelines),
            snippet=tool.snippet,
            execution_mode=tool.execution_mode,
        )

    def to_json(self) -> dict[str, JsonValue]:
        """Render this spec as the JSON object the child sends."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "guidelines": list(self.guidelines),
            "snippet": self.snippet,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, JsonValue]) -> ToolSpec:
        """Read one spec from the child, rejecting anything that is not a tool description."""
        try:
            name = payload["name"]
            description = payload["description"]
            parameters = payload["parameters"]
            guidelines = payload["guidelines"]
            snippet = payload["snippet"]
            execution_mode = payload["execution_mode"]
        except KeyError as missing:
            raise JITHostError(f"The repository process sent a tool without {missing.args[0]}") from None
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(parameters, dict)
            or not isinstance(snippet, str)
            or not isinstance(execution_mode, str)
        ):
            raise JITHostError("The repository process sent a malformed tool description")
        if not isinstance(guidelines, list) or not all(isinstance(item, str) for item in guidelines):
            raise JITHostError("The repository process sent malformed tool guidelines")
        return cls(
            name,
            description,
            parameters,
            tuple(cast("list[str]", guidelines)),
            snippet,
            ToolExecutionMode(execution_mode),
        )

    def build(self, invoke: Callable[[dict[str, JsonValue]], Awaitable[ToolResult]]) -> AgentTool:
        """Rebuild the tool in the agent, with ``invoke`` as its handler."""

        async def handler(**arguments: ToolResult) -> ToolResult:
            return await invoke(cast("dict[str, JsonValue]", arguments))

        return AgentTool(
            name=self.name,
            description=self.description,
            parameters=dict(self.parameters),
            handler=handler,
            guidelines=self.guidelines,
            snippet=self.snippet,
            execution_mode=self.execution_mode,
        )


def encode_value(value: ToolResult) -> JsonValue:
    """Render one tool result as a value that survives the pipe.

    Repository calls carry JSON: ``None``, booleans, numbers, strings, lists, and
    dictionaries with string keys. Anything else raises :class:`ToolNotSerializable`
    instead of being silently turned into a string, so a tool that returns a dataclass
    or an image is told to change rather than surprising its caller later.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ToolNotSerializable(value)
        return {key: encode_value(item) for key, item in value.items()}
    raise ToolNotSerializable(value)


def decode_value(payload: JsonValue) -> JsonValue:
    """Read one value the child sent, refusing anything JSON cannot carry."""
    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, list):
        return [decode_value(item) for item in payload]
    if isinstance(payload, dict):
        return {key: decode_value(item) for key, item in payload.items()}
    raise JITHostError(f"The repository process sent a value of type {type(payload).__name__}")


class HostedRepo:
    """The tools of one loaded version of a repository, wherever its code runs."""

    def __init__(
        self,
        alias: str,
        tools: dict[str, AgentTool],
        fingerprint: str,
        directory: Path,
        close: Callable[[], None] | None = None,
    ) -> None:
        self.alias = alias
        self.tools = tools
        self.fingerprint = fingerprint
        self.directory = directory
        self._close = close

    def close(self) -> None:
        """Release what this version held; the files stay on disk."""
        if self._close is not None:
            self._close()
            self._close = None


class JITToolHost(ABC):
    """Import a repository's tools and run them, in this process or another."""

    @abstractmethod
    def load(self, directory: Path, scan: RepoScan, bridge: HarnessCalls) -> HostedRepo:
        """Import every module of ``directory`` and describe the tools it defines.

        Args:
            directory: Repository root.
            scan: What the files looked like before the import.
            bridge: The object repository code sees as ``harness``. Hosts that run the
                code elsewhere use their own proxy and ignore it.
        """

    @abstractmethod
    async def invoke(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        """Run one tool of the loaded repository and return its raw result."""

    @abstractmethod
    def close(self) -> None:
        """Release every process and imported module this host owns."""

    @property
    def needs_reload(self) -> bool:
        """Whether the loaded version must be imported again even if its source did not change.

        A host whose code runs elsewhere reports True after its process died, so
        ``repo_load`` starts a fresh one instead of assuming the old tools still work.
        """
        return False

    def bind_dispatch(self, dispatch: Callable[[str, dict[str, JsonValue]], Awaitable[ToolResult]]) -> None:
        """Accept the agent-side dispatcher used to answer nested ``harness`` calls.

        The in-process host calls the harness directly and ignores it.
        """
        return None


class InProcessToolHost(JITToolHost):
    """Import and run a repository in the agent's own interpreter.

    This is the fast host: module globals stay in the agent, nested calls are ordinary
    method calls, and a tool may return any value. Repository code can also do anything
    the agent process can do, including exit it or never return.

    Examples:
        Construct a harness whose repository shares this interpreter::

            harness = JITHarness(directory, host=InProcessToolHost())
    """

    def __init__(self) -> None:
        self._loaded: LoadedRepo | None = None

    def load(self, directory: Path, scan: RepoScan, bridge: HarnessCalls) -> HostedRepo:
        """Import the package here, with ``bridge`` available in module globals."""
        loaded = load_repo(directory, bridge, scan)
        self._loaded = loaded
        return HostedRepo(loaded.alias, loaded.tools, loaded.fingerprint, directory, loaded.close)

    async def invoke(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        """Run a tool that was imported here.

        The harness reaches a tool through the ``AgentTool`` it registered, so this host
        is never asked to run one; the call exists so both hosts answer the same way.
        """
        raise JITHostError(f"{name!r} runs in this process; the registered handler was already used")

    def close(self) -> None:
        """Drop the imported modules."""
        if self._loaded is not None:
            self._loaded.close()
            self._loaded = None


class SubprocessToolHost(JITToolHost):
    """Import and run a repository in a child process, and call back over a pipe.

    The child starts with :data:`sys.executable`, imports the package, and then answers
    two kinds of frame: the agent asks it to run a tool, and the tool asks the agent to
    run a loaded tool through ``harness.call_tool``. Nested calls therefore take the
    same path as before, with extensions, events, and profiling in the agent.

    A crash, a hang, or a runaway allocation stays in the child; :meth:`invoke` gives up
    after ``call_timeout`` and kills it. The next :meth:`load` starts a fresh one.

    Lifecycle, and when a process exists:

    .. code-block:: text

        repo_create ──► no process

        repo_load   ──► spawn ──► import ──► tool schemas
                          ▲                    |
           source changed ┤                    |  invoke / call frames
                          │                    ▼
           crash, timeout ─┤              reuse the same child
           or close()    ──┘                    |
                                               |
        close/forget_session ─────────────────► shutdown, kill after 2 s

    A reload is a new interpreter: the previous version's modules, its module-level
    state, and the alias it registered are gone instead of lingering until the process
    exits. Tool calls never spawn: with no live child they fail with
    :class:`JITHostCrashed` and the model must call ``repo_load`` again.

    Args:
        python: Interpreter for the child; the agent's own by default.
        sandbox: Boundary to start the child inside, such as one built by
            :func:`~zett_agent.sandbox.factory.create_sandbox`. The child only needs to
            read the repository and the interpreter, and never writes.
        call_timeout: Seconds one tool may run before it is killed and reported as a
            timeout. ``None`` waits as long as it takes.
        load_timeout: Seconds an import may take. Module bodies run here too, so this is
            a real limit, not a formality.
        env: Extra environment variables for the child.

    Note:
        Values cross the pipe as JSON. A tool returns JSON values, ``TextContent``,
        ``ImageContent``, bytes, or lists and dictionaries of them; anything else raises
        :class:`ToolNotSerializable`, because pickle would let the child run code in the
        agent's process. User code's ``print`` goes to stderr.

    Examples:
        Run each repository in a read-only sandbox::

            policy = SandboxPolicy(...)
            host = SubprocessToolHost(sandbox=create_sandbox(policy))
            harness = JITHarness(directory, host=host)
    """

    def __init__(
        self,
        *,
        python: str | None = None,
        sandbox: Sandbox | None = None,
        call_timeout: float | None = 300.0,
        load_timeout: float | None = 120.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        for name, value in (("call_timeout", call_timeout), ("load_timeout", load_timeout)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be a positive number of seconds, or None")
        self.python = python or sys.executable
        self.sandbox = sandbox
        self.call_timeout = call_timeout
        self.load_timeout = load_timeout
        self.env = dict(env or {})
        self._process: subprocess.Popen[bytes] | None = None
        self._pending: dict[int, _Pending] = {}
        self._next_id = 0
        self._write_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatch: Callable[[str, dict[str, JsonValue]], Awaitable[ToolResult]] | None = None
        self._directory: Path | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    def bind_dispatch(self, dispatch: Callable[[str, dict[str, JsonValue]], Awaitable[ToolResult]]) -> None:
        """Remember the agent-side dispatcher used for nested ``harness`` calls."""
        self._dispatch = dispatch

    def load(self, directory: Path, scan: RepoScan, bridge: HarnessCalls) -> HostedRepo:
        """Start a fresh child process, import the package there, and build its tools.

        Any earlier child is stopped first. A reload therefore begins from an empty
        interpreter: the previous version's modules, its module-level state, and any
        alias it registered are gone, which is what the in-process host gets from
        discarding its old package.
        """
        self._stop()
        self._start(directory)
        reply = self._request_sync({"op": "load", "directory": str(directory)}, self.load_timeout)
        specs = self._specs_of(reply)
        tools = {spec.name: spec.build(self._invoker(spec.name)) for spec in specs}
        return HostedRepo(f"subprocess:{directory.name}", tools, scan.fingerprint, directory)

    def close(self) -> None:
        """Stop the child process, failing anything still waiting on it."""
        self._stop()

    def _start(self, directory: Path) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        argv = [self.python, "-m", "zett_agent.jit_harness.child"]
        env = {**self._child_env(), **self.env}
        cwd = str(directory)
        preexec = None
        if self.sandbox is not None:
            command = self.sandbox.wrap(argv, cwd=directory, env=self.env)
            argv = list(command.argv)
            env = dict(command.env)
            cwd = str(command.cwd)
            preexec = command.preexec
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            cwd=cwd,
            preexec_fn=preexec,
            start_new_session=True,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_frames, name="zett-jit-child", daemon=True)
        self._reader.start()

    @property
    def needs_reload(self) -> bool:
        """Whether the child must be started again because it is gone."""
        return self._process is None or self._process.poll() is not None

    def _child_env(self) -> dict[str, str]:
        """Pass the agent's import path, so the child can import the application's modules."""
        env = dict(os.environ)
        entries = [entry for entry in sys.path if entry]
        if entries:
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join([*entries, *([existing] if existing else [])])
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._write({"v": PROTOCOL_VERSION, "op": "shutdown"}, process)
                    process.wait(timeout=2)
                except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
                    pass
            # Sweep the group even after a clean exit: a tool may have started
            # processes that outlived the interpreter, and nothing else would reach them.
            _terminate_tree(process)
        finally:
            for pending in list(self._pending.values()):
                pending.fail(JITHostCrashed("The repository process was stopped"))
            self._pending.clear()
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:  # pragma: no cover - the pipe may already be gone
                    pass

    # -- frames --------------------------------------------------------------------------------

    async def invoke(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        """Run one tool in the child process and return its decoded result."""
        return await self._request(
            {"op": "invoke", "name": name, "arguments": encode_value(arguments)}, self.call_timeout
        )

    def _invoker(self, name: str) -> Callable[[dict[str, JsonValue]], Awaitable[ToolResult]]:
        async def invoke(arguments: dict[str, JsonValue]) -> ToolResult:
            return await self.invoke(name, arguments)

        return invoke

    @staticmethod
    def _specs_of(reply: Mapping[str, JsonValue]) -> list[ToolSpec]:
        raw = reply.get("tools")
        if not isinstance(raw, list):
            raise JITHostError("The repository process reported no tools")
        specs: list[ToolSpec] = []
        for item in raw:
            if not isinstance(item, dict):
                raise JITHostError("The repository process sent a tool that is not an object")
            specs.append(ToolSpec.from_json(item))
        return specs

    async def _request(self, frame: dict[str, JsonValue], timeout: float | None) -> JsonValue:
        """Send one request and await its reply, killing the child if it runs out of time."""
        if self._process is None or self._process.poll() is not None:
            raise JITHostCrashed("The repository process is not running; load the repository again")
        self._loop = asyncio.get_running_loop()
        pending = self._send(frame)
        try:
            reply = await asyncio.wait_for(asyncio.wrap_future(pending.future), timeout)
        except TimeoutError:
            self._kill(f"The repository process did not answer within {timeout:g} seconds")
            raise JITHostTimeout(f"A repository tool ran for more than {timeout:g} seconds and was stopped") from None
        return decode_value(reply.get("value"))

    def _request_sync(self, frame: dict[str, JsonValue], timeout: float | None) -> Mapping[str, JsonValue]:
        pending = self._send(frame)
        try:
            return pending.future.result(timeout)
        except FutureTimeout:
            self._kill(f"The repository process did not load within {timeout} seconds")
            raise JITHostTimeout("The repository took too long to import and was stopped") from None

    def _send(self, frame: dict[str, JsonValue]) -> _Pending:
        process = self._process
        assert process is not None
        self._next_id += 1
        identifier = self._next_id
        pending = _Pending(identifier)
        self._pending[identifier] = pending
        try:
            self._write({**frame, "v": PROTOCOL_VERSION, "id": identifier}, process)
        except (BrokenPipeError, OSError) as error:
            self._pending.pop(identifier, None)
            raise JITHostCrashed("The repository process exited before the request could be sent") from error
        return pending

    def _write(self, frame: Mapping[str, JsonValue], process: subprocess.Popen[bytes]) -> None:
        line = json.dumps(frame, ensure_ascii=False).encode("utf-8") + b"\n"
        stream = process.stdin
        if stream is None:  # pragma: no cover - only when the process was never started
            raise JITHostCrashed("The repository process has no input pipe")
        with self._write_lock:
            stream.write(line)
            stream.flush()

    def _read_frames(self) -> None:
        """Read replies and nested calls until the pipe closes, then fail what is waiting."""
        process = self._process
        if process is None or process.stdout is None:  # pragma: no cover - guarded by _start
            return
        try:
            for line in process.stdout:
                if len(line) > MAX_FRAME_BYTES:
                    self._kill("The repository process sent a frame that is too large")
                    return
                if not line.strip():
                    continue
                self._on_frame(json.loads(line))
        except (OSError, ValueError):
            pass
        finally:
            code = process.poll()
            for pending in list(self._pending.values()):
                pending.fail(
                    JITHostCrashed(f"The repository process exited (status {code}) before answering; load it again")
                )
            self._pending.clear()

    def _on_frame(self, frame: JsonValue) -> None:
        if not isinstance(frame, Mapping):
            self._kill("The repository process sent a frame that is not an object")
            return
        if frame.get("v") != PROTOCOL_VERSION:
            self._kill("The repository process speaks a different protocol version")
            return
        if "reply_to" in frame:
            reply_to = frame["reply_to"]
            if not isinstance(reply_to, int):
                self._kill("The repository process answered without a request id")
                return
            pending = self._pending.pop(reply_to, None)
            if pending is None:
                return
            if frame.get("ok"):
                pending.succeed(frame)
            else:
                message = str(frame.get("error", "The repository tool failed"))
                # A failed import is a repository the model must fix, which the harness
                # reports as a ValueError carrying the file and line.
                pending.fail(ValueError(message) if frame.get("kind") == "load" else JITHostError(message))
            return
        if frame.get("op") == "call":
            self._answer_call(frame)
            return
        self._kill(f"The repository process sent an unknown frame: {frame.get('op')!r}")

    def _answer_call(self, frame: Mapping[str, JsonValue]) -> None:
        """Run one nested ``harness`` call in the agent and send its result back."""
        loop, dispatch = self._loop, self._dispatch
        if loop is None or dispatch is None or loop.is_closed():
            self._kill("The repository process asked the agent for a tool call too early")
            return
        name, arguments, identifier = frame.get("name"), frame.get("arguments"), frame.get("id")
        if not isinstance(name, str) or not isinstance(identifier, int):
            self._kill("The repository process sent a malformed tool call")
            return
        try:
            decoded = decode_value(arguments)
        except JITHostError as error:
            self._kill(str(error))
            return
        if not isinstance(decoded, Mapping):
            self._kill("The repository process sent tool arguments that are not a mapping")
            return

        async def answer() -> None:
            try:
                value = await dispatch(name, dict(decoded))
                frame_out = {"v": PROTOCOL_VERSION, "reply_to": identifier, "ok": True, "value": encode_value(value)}
            except Exception as error:
                frame_out = {
                    "v": PROTOCOL_VERSION,
                    "reply_to": identifier,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._write(frame_out, process)
                except (BrokenPipeError, OSError):  # pragma: no cover - the child died mid-call
                    pass

        asyncio.run_coroutine_threadsafe(answer(), loop)

    def _kill(self, reason: str) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            _terminate_tree(process)
        for pending in list(self._pending.values()):
            pending.fail(JITHostCrashed(reason))
        self._pending.clear()


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill the child and everything it started, not only the process we spawned.

    The child is started in its own session (``start_new_session=True``), so its
    process group id is its pid and one call reaches a sandbox launcher and the
    interpreter it started. Windows has no process groups, so it falls back to
    killing the direct child.
    """
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):  # pragma: no cover - gone or not ours
            pass
    process.kill()


class _Pending:
    """One frame the agent is waiting for, answered by the reader thread."""

    def __init__(self, identifier: int) -> None:
        self.identifier = identifier
        self.future: Future[Mapping[str, JsonValue]] = Future()

    def succeed(self, reply: Mapping[str, JsonValue]) -> None:
        if not self.future.done():
            self.future.set_result(reply)

    def fail(self, error: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(error)


__all__ = [
    "HarnessCalls",
    "HostedRepo",
    "InProcessToolHost",
    "JITHostCrashed",
    "JITHostError",
    "JITHostTimeout",
    "JITToolHost",
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "SubprocessToolHost",
    "ToolNotSerializable",
    "ToolSpec",
    "decode_value",
    "encode_value",
]
