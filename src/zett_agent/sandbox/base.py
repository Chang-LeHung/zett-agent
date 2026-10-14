"""The sandbox interface and the process runner every backend shares."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .._compat import timeout as async_timeout
from .errors import SandboxError, SandboxPolicyError
from .policy import SandboxPolicy, build_environment, command_arguments

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "Sandbox",
    "SandboxResult",
    "SandboxedCommand",
]

#: Wall-clock limit of one run unless the caller passes another (or None for none).
DEFAULT_TIMEOUT_SECONDS = 60.0
#: Bytes of each output stream a run keeps. The rest is read and discarded, so a
#: process that prints without end can neither fill memory nor block on a full pipe.
DEFAULT_MAX_OUTPUT_BYTES = 1_048_576
#: Seconds the runner waits for the output streams after the process has exited.
_DRAIN_GRACE_SECONDS = 1.0
_CHUNK_BYTES = 65_536
#: Whether process groups exist. Windows has none, so only the process itself is stopped there.
_POSIX = os.name == "posix"


@dataclass(frozen=True, slots=True)
class SandboxedCommand:
    """A command rewritten to run inside a sandbox, ready for a process API.

    :meth:`Sandbox.wrap` returns one without running anything, so a host that
    owns its process handling (streaming output, a pty, its own timeout) can
    spawn ``argv`` itself with ``env`` and ``cwd``. Start it in its own process
    group (``start_new_session=True`` on POSIX) so the whole tree can be stopped.

    Attributes:
        argv: The program and its arguments, including the sandbox launcher.
        env: The complete environment to start it with.
        cwd: The working directory to start it in.
        warnings: Parts of the policy this backend could not enforce for this
            command. A caller that must not run with a weaker boundary checks it.
    """

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """What one sandboxed run produced.

    Attributes:
        exit_code: The process's exit status. A negative number is the signal
            that ended it, which is what a timeout produces.
        stdout: Standard output, decoded as UTF-8 with undecodable bytes replaced.
        stderr: Standard error, decoded the same way.
        timed_out: Whether the run exceeded its time limit and was killed.
        stdout_truncated: Whether standard output exceeded the byte limit.
        stderr_truncated: Whether standard error exceeded the byte limit.
        duration: Seconds the run took.
        sandbox: Name of the backend that ran it.
        isolated: Whether that backend enforces a boundary at all.
        warnings: Parts of the policy the backend could not enforce.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration: float = 0.0
    sandbox: str = ""
    isolated: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Whether the process exited with status 0 within its time limit."""
        return self.exit_code == 0 and not self.timed_out


class _Capture:
    """Keep the first bytes of a stream and discard the rest while still reading it."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def feed(self, chunk: bytes) -> None:
        room = self.limit - len(self.data)
        if room > 0:
            self.data += chunk[:room]
        if len(chunk) > max(room, 0):
            self.truncated = True

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


class _Run(asyncio.SubprocessProtocol):
    """Collect output and report the process's exit separately from its pipes closing.

    ``asyncio.subprocess.Process.wait`` returns only once the pipes are closed,
    so a background child that still holds standard output would keep a finished
    command waiting. The protocol's ``process_exited`` fires when the process
    itself ends, which is the moment the runner stops the rest of its group.
    """

    def __init__(self, limit: int) -> None:
        loop = asyncio.get_running_loop()
        self.out, self.err = _Capture(limit), _Capture(limit)
        self.exited: asyncio.Future[None] = loop.create_future()
        self.drained: asyncio.Future[None] = loop.create_future()
        self._open = {1, 2}

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        (self.out if fd == 1 else self.err).feed(data)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        self._open.discard(fd)
        if not self._open and not self.drained.done():
            self.drained.set_result(None)

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)


def _kill(pid: int, transport: asyncio.SubprocessTransport) -> None:
    """Stop a process and, on POSIX, everything it started in its group."""
    with suppress(ProcessLookupError, PermissionError):
        if _POSIX:
            os.killpg(pid, signal.SIGKILL)
        else:
            transport.kill()


class Sandbox(ABC):
    """Run a command inside a boundary described by a :class:`SandboxPolicy`.

    A backend implements two things: :meth:`check`, which says whether it can
    run on this machine, and :meth:`_wrap`, which rewrites a command so the
    operating system enforces the policy on it. Everything else, including
    the environment, the time limit, the output cap, stopping the whole process
    tree and cancellation, is shared, so every backend behaves the same except
    in what it is able to confine.

    The interface is deliberately split in two. :meth:`wrap` returns the
    rewritten command without running it, for hosts that manage processes
    themselves; :meth:`run` runs it and collects the result.

    Args:
        policy: The boundary to enforce. Defaults to
            :meth:`SandboxPolicy.read_only`.

    Examples:
        Run a program that may write one directory::

            sandbox = create_sandbox(SandboxPolicy.workspace_write("/work/project"))
            result = await sandbox.run(["python3", "build.py"], cwd="/work/project")
            if not result.succeeded:
                print(result.stderr)

        Run a shell pipeline::

            result = await sandbox.run("grep -r TODO . | wc -l", cwd="/work/project")

    .. note::
        The sandbox confines a process and everything it starts. It does not
        decide whether a command should run at all; asking a person first is a
        separate control, for example
        :class:`~zett_agent.extensions.shell_approval.ShellApprovalExtension`.

    .. warning::
        Several runs share one policy but no state: each starts a fresh process
        in a fresh boundary. Files a run writes inside a writable path stay
        there.
    """

    #: Short stable name of the backend, reported in results.
    name: ClassVar[str]
    #: Whether this backend enforces a boundary. False only for the backend that
    #: runs commands as they are.
    isolated: ClassVar[bool]

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy if policy is not None else SandboxPolicy.read_only()
        if not isinstance(self.policy, SandboxPolicy):
            raise SandboxPolicyError("policy must be a SandboxPolicy")

    @abstractmethod
    def check(self) -> None:
        """Raise :class:`SandboxUnavailableError` unless this backend can run here.

        The error says what is missing and how to fix it. Calling it is cheap
        after the first time.
        """

    @abstractmethod
    def _wrap(self, argv: tuple[str, ...], cwd: Path, env: dict[str, str]) -> SandboxedCommand:
        """Rewrite one argument vector so the platform enforces the policy on it."""

    def wrap(
        self,
        command: str | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SandboxedCommand:
        """Return ``command`` rewritten to run inside this sandbox, without running it.

        Args:
            command: A shell string, run through ``/bin/sh -c``, or the program
                followed by its arguments, run directly.
            cwd: Working directory; the current directory by default.
            env: Variables added to the policy's inherited environment.

        Raises:
            SandboxUnavailableError: If the backend cannot run here.
            SandboxPolicyError: If the command, directory or policy is invalid.
        """
        self.check()
        arguments = command_arguments(command)
        directory = Path.cwd() if cwd is None else Path(cwd).expanduser()
        try:
            directory = directory.resolve(strict=True)
        except OSError as error:
            raise SandboxPolicyError(f"working directory does not exist: {directory}") from error
        if not directory.is_dir():
            raise SandboxPolicyError(f"working directory is not a directory: {directory}")
        environment = build_environment(self.policy.inherit_environment, env)
        return self._wrap(arguments, directory, environment)

    async def run(
        self,
        command: str | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | bytes | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> SandboxResult:
        """Run ``command`` inside the sandbox and collect what it produced.

        The process runs in its own process group. A timeout, a cancellation of
        the awaiting task, and the end of the run each stop the whole group, so
        no helper process outlives its command.

        Args:
            command: A shell string or the program followed by its arguments.
            cwd: Working directory; the current directory by default.
            env: Variables added to the policy's inherited environment.
            stdin: Input to feed the process; it gets no input otherwise.
            timeout: Seconds before the run is killed; None waits without limit.
            max_output_bytes: Bytes kept of each of standard output and error.

        Raises:
            SandboxUnavailableError: If the backend cannot run here.
            SandboxPolicyError: If an argument or the policy is invalid.
            SandboxError: If the process could not be started.
            asyncio.CancelledError: When the caller is cancelled, after the
                process group has been stopped.
        """
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1:
            raise SandboxPolicyError("max_output_bytes must be a positive integer")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
            raise SandboxPolicyError("timeout must be a positive number of seconds or None")
        wrapped = self.wrap(command, cwd=cwd, env=env)
        data = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            transport, run = await loop.subprocess_exec(
                lambda: _Run(max_output_bytes),
                *wrapped.argv,
                cwd=wrapped.cwd,
                env=dict(wrapped.env),
                stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=_POSIX,
            )
        except OSError as error:
            raise SandboxError(f"could not start {wrapped.argv[0]!r}: {error}") from error
        pid = transport.get_pid()
        timed_out = False
        try:
            if data is not None:
                stdin_pipe = transport.get_pipe_transport(0)
                stdin_pipe.write(data)
                stdin_pipe.write_eof()
            try:
                if timeout is None:
                    await run.exited
                else:
                    # Shielded: the timeout cancels the wait, not the future, which is awaited again below.
                    async with async_timeout(timeout):
                        await asyncio.shield(run.exited)
            except (TimeoutError, asyncio.TimeoutError):
                timed_out = True
            # Whether the command ended or ran out of time, stop what is left of
            # its group: a background child must not outlive the run, and one that
            # still holds a pipe would otherwise keep the output open.
            _kill(pid, transport)
            if timed_out:
                await run.exited
            with suppress(TimeoutError, asyncio.TimeoutError):
                async with async_timeout(_DRAIN_GRACE_SECONDS):
                    await asyncio.shield(run.drained)
        except asyncio.CancelledError:
            _kill(pid, transport)
            raise
        finally:
            transport.close()
        exit_code = transport.get_returncode()
        out, err = run.out, run.err
        return SandboxResult(
            exit_code=exit_code,
            stdout=out.text(),
            stderr=err.text(),
            timed_out=timed_out,
            stdout_truncated=out.truncated,
            stderr_truncated=err.truncated,
            duration=time.monotonic() - started,
            sandbox=self.name,
            isolated=self.isolated,
            warnings=wrapped.warnings,
        )
