"""The child process that runs one repository.

It reads newline-delimited JSON frames from stdin and writes them to stdout, imports the
package when asked, runs its tools, and answers every ``harness.call_tool`` through the
agent with a frame of its own. Repository code never sees this protocol: it gets a
:class:`HarnessProxy` that looks like the in-process harness.

Start it only through :class:`~zett_agent.jit_harness.host.SubprocessToolHost`, which
starts it with the parent's interpreter and import path.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Awaitable, Iterable, Mapping
from typing import BinaryIO, cast

from ..json_types import JsonValue
from ..tools.base import ToolResult
from .host import PROTOCOL_VERSION, HarnessCalls, JITHostError, ToolSpec, decode_value, encode_value
from .repo import LoadedRepo, load_repo, scan_repo


class HarnessProxy(HarnessCalls):
    """What repository code calls as ``harness`` inside the child.

    Each call becomes one frame to the agent, which runs the loaded tool there and
    sends the result back. Asynchronous tools wait on the loop; synchronous tools wait
    on an event from whatever worker thread they run in.
    """

    def __init__(self, server: _Server) -> None:
        self._server = server
        self._loading = True

    def ready(self) -> None:
        """Allow calls again once the package has finished importing."""
        self._loading = False

    async def call_tool(self, name: str, **arguments: ToolResult) -> ToolResult:
        """Ask the agent to run one loaded tool from an asynchronous tool."""
        self._check()
        return await self._server.call(name, cast("dict[str, JsonValue]", arguments))

    def call_tool_sync(self, name: str, **arguments: ToolResult) -> ToolResult:
        """Ask the agent to run one loaded tool from a synchronous tool."""
        self._check()
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("call_tool_sync cannot block the child's event loop; use await call_tool")
        return self._server.call_sync(name, cast("dict[str, JsonValue]", arguments))

    def _check(self) -> None:
        if self._loading:
            raise RuntimeError("Cannot call tools while loading Python source")


class _Waiter:
    """One nested call the child is waiting for, answered by the reader thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.future: asyncio.Future[JsonValue] = loop.create_future()
        self.event = threading.Event()
        self.value: JsonValue = None
        self.error: BaseException | None = None

    def succeed(self, value: JsonValue) -> None:
        try:
            self.value = decode_value(value)
        except Exception as error:  # pragma: no cover - a malformed frame from the agent
            self.fail(error)
            return
        self.event.set()
        self.loop.call_soon_threadsafe(self._resolve, self.value)

    def _resolve(self, value: JsonValue) -> None:
        if not self.future.done():
            self.future.set_result(value)

    def _reject(self, error: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(error)

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.event.set()
        self.loop.call_soon_threadsafe(self._reject, error)


class _Server:
    """Read frames, run tool calls, and answer nested calls until told to stop."""

    def __init__(self, output: BinaryIO, source: Iterable[bytes] | None = None) -> None:
        self._output = output
        self._source: Iterable[bytes] = source if source is not None else sys.stdin.buffer
        self._write_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._waiters: dict[int, _Waiter] = {}
        self._next_id = 0
        self._repo: LoadedRepo | None = None
        self._stopping = threading.Event()

    # -- frames ----------------------------------------------------------------------------

    def write(self, frame: Mapping[str, JsonValue]) -> None:
        """Write one frame, serializing writers from the loop and from worker threads."""
        line = json.dumps({**frame, "v": PROTOCOL_VERSION}, ensure_ascii=False).encode("utf-8") + b"\n"
        with self._write_lock:
            self._output.write(line)
            self._output.flush()

    def read_forever(self) -> None:
        """Read frames from the agent until the pipe closes."""
        for raw in self._source:
            if raw.strip():
                try:
                    frame = json.loads(raw)
                except ValueError:
                    break
                if not isinstance(frame, Mapping) or frame.get("v") != PROTOCOL_VERSION:
                    break
                self._dispatch(frame)
            if self._stopping.is_set():
                break
        self._stopping.set()
        for waiter in list(self._waiters.values()):
            waiter.fail(JITHostError("The agent closed the pipe"))
        self._waiters.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _dispatch(self, frame: Mapping[str, JsonValue]) -> None:
        if "reply_to" in frame:
            waiter = self._waiters.pop(int(cast(int, frame["reply_to"])), None)
            if waiter is None:
                return
            if frame.get("ok"):
                waiter.succeed(frame.get("value"))
            else:
                waiter.fail(JITHostError(str(frame.get("error", "The agent refused the call"))))
            return
        operation = frame.get("op")
        if operation == "load":
            request_id = int(cast(int, frame.get("id", 0)))
            self._call_soon(self._load(request_id, str(frame.get("directory", ""))))
        elif operation == "invoke":
            request_id = int(cast(int, frame.get("id", 0)))
            self._call_soon(self._invoke(request_id, frame))
        else:
            self._stopping.set()
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _call_soon(self, coroutine: Awaitable[None]) -> None:
        def schedule() -> None:
            asyncio.ensure_future(coroutine)

        self._loop.call_soon_threadsafe(schedule)

    # -- operations ------------------------------------------------------------------------

    async def _load(self, request_id: int, directory: str) -> None:
        bridge = HarnessProxy(self)
        try:
            from pathlib import Path

            root = Path(directory)
            self._repo = load_repo(root, bridge, scan_repo(root))
            specs: list[JsonValue] = [
                spec.to_json() for spec in (ToolSpec.of(tool) for tool in self._repo.tools.values())
            ]
        except Exception as error:
            self.write({"reply_to": request_id, "ok": False, "kind": "load", "error": str(error)})
            return
        bridge.ready()
        self.write({"reply_to": request_id, "ok": True, "tools": specs})

    async def _invoke(self, request_id: int, frame: Mapping[str, JsonValue]) -> None:
        name = str(frame.get("name", ""))
        try:
            if self._repo is None:
                raise JITHostError("No repository is loaded in this process")
            tool = self._repo.tools.get(name)
            if tool is None:
                raise JITHostError(f"Unknown tool: {name}")
            arguments = decode_value(frame.get("arguments"))
            value = await tool(cast("dict[str, JsonValue]", arguments if isinstance(arguments, dict) else {}))
            answer: dict[str, JsonValue] = {"reply_to": request_id, "ok": True, "value": encode_value(value)}
        except Exception as error:
            answer = {
                "reply_to": request_id,
                "ok": False,
                "kind": "invoke",
                "error": f"{type(error).__name__}: {error}",
            }
        self.write(answer)

    # -- nested calls ----------------------------------------------------------------------

    def _allocate(self) -> int:
        self._next_id += 1
        return self._next_id

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        """Send one nested call and await the agent's answer."""
        request_id = self._allocate()
        waiter = _Waiter(self._loop)
        self._waiters[request_id] = waiter
        self.write({"id": request_id, "op": "call", "name": name, "arguments": encode_value(dict(arguments))})
        return await waiter.future

    def call_sync(self, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        """Send one nested call from a worker thread and block until it is answered."""
        request_id = self._allocate()
        waiter = _Waiter(self._loop)
        self._waiters[request_id] = waiter
        self.write({"id": request_id, "op": "call", "name": name, "arguments": encode_value(dict(arguments))})
        waiter.event.wait()
        self._waiters.pop(request_id, None)
        if waiter.error is not None:
            raise waiter.error
        return waiter.value

    def run(self) -> int:
        """Run the loop until the agent stops this process."""
        self._loop.set_exception_handler(_quiet_handler)
        reader = threading.Thread(target=self.read_forever, name="zett-jit-host", daemon=True)
        reader.start()
        self._loop.run_forever()
        return 0


def _quiet_handler(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
    """Report loop errors on stderr instead of aborting the process."""
    message = context.get("exception") or context.get("message")
    print(f"zett jit host: {message}", file=sys.stderr)


def main() -> int:
    """Run the child protocol on stdin and stdout until the agent stops it."""
    protocol = sys.stdout
    sys.stdout = sys.stderr
    return _Server(protocol.buffer).run()


if __name__ == "__main__":  # pragma: no cover - exercised through the host
    raise SystemExit(main())
