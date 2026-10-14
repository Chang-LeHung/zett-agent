"""The sandbox interface and the process runner every backend shares."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from .errors import SandboxError, SandboxPolicyError
from .policy import ResourceLimits, SandboxPolicy, build_environment, command_arguments
from .resources import (
    CgroupUsage,
    MemoryStrategy,
    ResourcePlan,
    ResourceReport,
    descendant_pids,
    plan_resources,
    process_tree_rss_bytes,
    watchdog_available,
)

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MemoryStrategy",
    "Sandbox",
    "SandboxResult",
    "SandboxedCommand",
]

#: Signals the kernel uses to report a limit that was reached, with the limit that
#: raises each and the field of ``ResourceLimits`` that has to have asked for it.
_LIMIT_SIGNALS = {
    getattr(signal, "SIGXCPU", -1000): ("cpu", "cpu_seconds"),
    getattr(signal, "SIGXFSZ", -1001): ("file_size", "file_size_bytes"),
}

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
        limits: The resource limits the policy asked for. A host that spawns the
            command itself applies the kernel limits with ``preexec``; the ones
            only :meth:`Sandbox.run` can enforce are named in ``warnings``.
        preexec: Apply the kernel resource limits in the child before ``exec``,
            for ``subprocess``' ``preexec_fn``. None when there is nothing to apply.
        warnings: Parts of the policy or the limits this backend could not enforce
            for this command. A caller that must not run with a weaker boundary
            checks it.
    """

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    limits: ResourceLimits = ResourceLimits()
    preexec: Callable[[], None] | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """What one sandboxed run produced.

    Attributes:
        exit_code: The process's exit status. A negative number is the signal
            that ended it, which is what a timeout produces. Bubblewrap runs the
            command under its own init, which reports a signal as ``128`` plus its
            number (137 for ``SIGKILL``), as a shell does.
        stdout: Standard output, decoded as UTF-8 with undecodable bytes replaced.
        stderr: Standard error, decoded the same way.
        timed_out: Whether the run exceeded its time limit and was killed.
        stdout_truncated: Whether standard output exceeded the byte limit.
        stderr_truncated: Whether standard error exceeded the byte limit.
        duration: Seconds the run took.
        sandbox: Name of the backend that ran it.
        isolated: Whether that backend enforces a boundary at all.
        warnings: Parts of the policy the backend could not enforce. Resource
            warnings are on ``resources`` instead.
        resources: How each limit was enforced, what the run measured, and the
            limits it could not enforce here.
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
    resources: ResourceReport = ResourceReport()

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


def _kill(pid: int, transport: asyncio.SubprocessTransport, *, tree: bool = False) -> None:
    """Stop a process and, on POSIX, everything it started in its group.

    With ``tree`` the descendants are found first and killed too, which reaches a
    child that started a session of its own and so left the group. That costs a
    ``ps`` call, so it is used when a command is being stopped (a timeout, a
    cancellation, a memory limit) and not after one that finished by itself, where
    the parent links it would follow are already gone.
    """
    if _POSIX and tree:
        for process in descendant_pids(pid):
            with suppress(ProcessLookupError, PermissionError):
                os.kill(process, signal.SIGKILL)
    with suppress(ProcessLookupError, PermissionError):
        if _POSIX:
            os.killpg(pid, signal.SIGKILL)
        else:
            transport.kill()


@dataclass
class _MemoryWatchdog:
    """Sample the memory of a process tree, and remember the highest value seen.

    The parent runs this, outside the sandbox, so it sees the whole tree. It is
    the last resort for a memory limit, used where the kernel cannot enforce one;
    between two samples the group can go over the limit.
    """

    pid: int
    limit_bytes: int
    sample_seconds: float
    peak_bytes: int | None = None
    exceeded: bool = False
    unreadable: bool = False

    async def watch(self, exited: asyncio.Future[None]) -> None:
        """Sample until the process exits or the limit is passed."""
        while not exited.done():
            measured = await asyncio.to_thread(process_tree_rss_bytes, self.pid)
            if measured is None:
                # Nothing was measured, so nothing is enforced. It is reported
                # rather than passed off as a run within its limit.
                self.unreadable = True
            else:
                self.peak_bytes = measured if self.peak_bytes is None else max(self.peak_bytes, measured)
                if measured > self.limit_bytes:
                    self.exceeded = True
                    return
            await asyncio.sleep(self.sample_seconds)


def _limit_for_exit(exit_code: int | None, requested: tuple[str, ...]) -> str | None:
    """Name the limit whose signal ended a process, from its exit status.

    A process killed by a signal reports ``-N`` when it is run directly. Bubblewrap
    runs the command under its own init and reports ``128 + N`` instead, as a shell
    does, so both are read. The limit has to have been asked for: a status that
    merely looks like a signal is not attributed to a limit nobody set.
    """
    if exit_code is None:
        return None
    for number, (limit, field) in _LIMIT_SIGNALS.items():
        if field in requested and exit_code in (-number, 128 + number):
            return limit
    return None


def _measurements(
    usage: CgroupUsage | None,
    watchdog: _MemoryWatchdog | None,
    exit_code: int | None,
    requested: tuple[str, ...] = (),
) -> tuple[int | None, tuple[str, ...], tuple[str, ...]]:
    """Collect what the run measured: peak memory, and which limits it reached.

    Peak memory comes from wherever the limit was enforced: the kernel's own
    accounting in a cgroup, which is read before the group is released, or the
    parent's sampling. A limit the kernel reports by killing the process is read
    from the signal that ended it.
    """
    peak = watchdog.peak_bytes if watchdog is not None else None
    hit: list[str] = []
    if watchdog is not None and watchdog.exceeded:
        hit.append("memory")
    if usage is not None:
        if usage.peak_memory_bytes is not None:
            peak = usage.peak_memory_bytes
        if usage.oom_kills or usage.memory_events_max:
            hit.append("memory")
    limit = _limit_for_exit(exit_code, requested)
    if limit is not None:
        hit.append(limit)
    warnings: list[str] = []
    if watchdog is not None and watchdog.unreadable:
        warnings.append(
            "memory_bytes: the watchdog could not read the command's memory on this machine, "
            "so nothing enforced the limit"
        )
    return peak, tuple(dict.fromkeys(hit)), tuple(warnings)


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
        policy: The boundary to enforce, including its resource limits. Defaults
            to :meth:`SandboxPolicy.read_only`.
        memory_strategy: How to enforce ``memory_bytes``; see
            :class:`~zett_agent.sandbox.resources.MemoryStrategy`. The default
            takes the strongest mechanism the machine has.

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

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        *,
        memory_strategy: MemoryStrategy | str = MemoryStrategy.AUTO,
    ) -> None:
        self.policy = policy if policy is not None else SandboxPolicy.read_only()
        if not isinstance(self.policy, SandboxPolicy):
            raise SandboxPolicyError("policy must be a SandboxPolicy")
        try:
            self.memory_strategy = MemoryStrategy(memory_strategy)
        except ValueError:
            raise SandboxPolicyError(f"unknown memory strategy: {memory_strategy!r}") from None
        if self.memory_strategy is MemoryStrategy.WATCHDOG and not watchdog_available():
            raise SandboxPolicyError("MemoryStrategy.WATCHDOG needs a POSIX system with a ps command")

    @abstractmethod
    def check(self) -> None:
        """Raise :class:`SandboxUnavailableError` unless this backend can run here.

        The error says what is missing and how to fix it. Calling it is cheap
        after the first time.
        """

    def capabilities(self) -> ResourceReport:
        """Return how each limit of this sandbox's policy is enforced on this machine.

        Nothing is created or changed, so it is safe to ask before running
        anything: a host can refuse a policy that this machine cannot enforce as
        written instead of discovering it from a run. The mechanisms named here
        are the ones a run will use.

        Examples:
            Refuse to run unless the machine enforces memory exactly::

                report = sandbox.capabilities()
                if report.memory not in ("cgroup", "rlimit_as"):
                    raise RuntimeError(report.warnings)

        .. note::
            A report is about this machine, not about the policy alone: the same
            policy reports ``cgroup`` on a Linux host with delegated controllers
            and ``watchdog`` on macOS.
        """
        return self._plan(create_cgroup=False).report

    def _plan(self, *, create_cgroup: bool = True) -> ResourcePlan:
        """Plan the resource limits for this policy, optionally creating the cgroup to use."""
        return plan_resources(
            self.policy.limits,
            strategy=self.memory_strategy,
            name=f"{self.name}-sandbox",
            create_cgroup=create_cgroup,
        )

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
        command = self._wrap(arguments, directory, environment)
        # The kernel limits travel with the command: a host that spawns it applies
        # them through `preexec`. What needs the parent (a cgroup to create, or
        # memory to sample) only happens in `run`, and saying so is the point.
        plan = self._plan(create_cgroup=False)
        parent_side = [
            name for name, mechanism in plan.report.mechanisms.items() if mechanism in ("watchdog", "cgroup")
        ]
        warnings = list(plan.warnings)
        if parent_side:
            warnings.append(
                f"{', '.join(sorted(parent_side))}: only run() enforces this, because it needs the parent process; "
                "spawn the command with preexec_fn=command.preexec for the rest"
            )
        return replace(
            command,
            limits=self.policy.limits,
            preexec=plan.preexec(),
            warnings=(*command.warnings, *warnings),
        )

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
        plan = self._plan()
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
                # The child joins the cgroup and takes its own limits on before exec.
                preexec_fn=plan.preexec(),
            )
        except OSError as error:
            plan.cleanup()
            raise SandboxError(f"could not start {wrapped.argv[0]!r}: {error}") from error
        pid = transport.get_pid()
        timed_out = False
        usage: CgroupUsage | None = None
        watchdog = (
            _MemoryWatchdog(
                pid=pid,
                limit_bytes=self.policy.limits.memory_bytes,  # type: ignore[arg-type]
                sample_seconds=plan.sample_seconds,
            )
            if plan.watchdog
            else None
        )
        watcher = asyncio.create_task(watchdog.watch(run.exited)) if watchdog is not None else None
        try:
            if data is not None:
                stdin_pipe = transport.get_pipe_transport(0)
                stdin_pipe.write(data)
                stdin_pipe.write_eof()
            # Three ways to stop waiting: the command ends, the memory watchdog
            # finds the group over its limit, or the time limit passes.
            waiters: list[asyncio.Future[None]] = [run.exited]
            if watcher is not None:
                waiters.append(watcher)
            done, _ = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            timed_out = not done
            stopped = timed_out or (watchdog is not None and watchdog.exceeded)
            # However it ended, stop what is left of the group: a background child
            # must not outlive the run, and one that still holds a pipe would
            # otherwise keep the output open.
            _kill(pid, transport, tree=stopped)
            if not run.exited.done():
                await run.exited
            with suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(run.drained), _DRAIN_GRACE_SECONDS)
            if plan.cgroup is not None:
                # Read while the group still exists: cleanup removes it.
                usage = plan.cgroup.usage()
        except asyncio.CancelledError:
            _kill(pid, transport, tree=True)
            raise
        finally:
            if watcher is not None and not watcher.done():
                watcher.cancel()
            if watcher is not None:
                await asyncio.gather(watcher, return_exceptions=True)
            plan.cleanup()
            transport.close()
        exit_code = transport.get_returncode()
        out, err = run.out, run.err
        peak, limits_hit, runs_warnings = _measurements(usage, watchdog, exit_code, self.policy.limits.requested())
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
            resources=plan.measured(peak_memory_bytes=peak, limits_hit=limits_hit, warnings=runs_warnings),
        )
