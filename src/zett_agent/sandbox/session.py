"""The simple way to use the sandbox: configure once, use it like ordinary Python, and it cleans up.

:class:`SandboxSession` wraps everything else in this package. One call states
the workspace and the budget; ``with`` opens it, ``run`` runs commands inside it,
and leaving the block removes whatever the session created.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import re
import shutil
import stat
import tempfile
import time
import warnings
import weakref
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from threading import Lock
from types import TracebackType

from ..sync_runtime import SyncRuntime
from .base import DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT_SECONDS, Sandbox, SandboxResult
from .errors import SandboxClosedError, SandboxCommandError, SandboxError, SandboxPolicyError
from .factory import SandboxBackend, create_sandbox
from .policy import NetworkMode, ResourceLimits, SandboxPolicy, _as_paths
from .resources import MemoryStrategy, ResourceReport

__all__ = ["SandboxSession", "parse_size"]

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)
_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
#: Seconds close() waits for runs it cancelled to stop before it removes anything.
_CLOSE_WAIT_SECONDS = 10.0
_DEFAULT = object()


def parse_size(value: int | str, name: str = "size") -> int:
    """Turn ``"512MB"``, ``"1.5G"`` or a byte count into bytes.

    ``K``, ``M``, ``G`` and ``T`` (with or without ``B`` or ``iB``, in any case)
    are powers of 1024, which is how memory is counted everywhere else in this
    package. A bare number is bytes.

    Raises:
        SandboxPolicyError: If the value is not a size.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SandboxPolicyError(f"{name} must be a number of bytes or a string such as '512MB', not {value!r}")
    if isinstance(value, int):
        return value
    match = _SIZE.match(value)
    if match is None:
        raise SandboxPolicyError(f"{name} is not a size: {value!r}; write it like '512MB', '2G' or '65536'")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def _unlock(path: Path) -> None:
    """Make a tree removable: a program may have left directories it cannot enter."""
    for directory, directories, _files in os.walk(path, followlinks=False):
        for name in (".", *directories):
            target = os.path.join(directory, name)
            try:
                if not os.path.islink(target):
                    os.chmod(target, os.stat(target).st_mode | stat.S_IRWXU)
            except OSError:
                continue


def _destroy(path: str) -> bool:
    """Remove a directory tree the session created. Returns whether it is gone."""
    target = Path(path)
    for attempt in range(3):
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            return True
        except OSError:
            _unlock(target)
            time.sleep(0.05 * (attempt + 1))
        else:
            return True
    return not target.exists()


class SandboxSession:
    """Run commands inside a sandbox with the workspace and budget you configure.

    Say where the process may write and how much it may use; then use it like
    ordinary Python. Leaving the ``with`` block cancels anything still running and
    removes everything the session created, whether the block ended normally or
    with an exception.

    * **Workspace.** Pass a directory to let commands write it, and it is never
      deleted. Pass nothing and the session creates a private one and deletes it on
      exit, which is what you want for running a program someone else wrote.
    * **Everything else is read-only,** apart from private scratch space for
      temporary files, which is also deleted.
    * **The network is off** unless you turn it on.
    * **Secrets stay out.** Common credential files are unreadable and the
      environment is an allowlist, so a token in yours does not reach the command.

    Args:
        workspace: Directory commands may write. None creates a private one that is
            deleted when the session ends.
        memory: Memory the process and everything it starts may use, as bytes or a
            string such as ``"512MB"``.
        cpus: Share of processors, in cores: ``0.5`` is half of one core, ``2`` is
            two. Needs cgroup v2, so it is Linux only; elsewhere it is reported as
            not enforced.
        cpu_seconds: Total processor time before the kernel kills the process.
            Sleeping costs none of it.
        processes: Processes a command may add (at least 8). See
            :attr:`~zett_agent.sandbox.policy.ResourceLimits.processes` for what
            is counted on each platform.
        max_file_size: Largest file the process may create, as bytes or a string.
        open_files: File descriptors the process may hold open (at least 16).
        network: ``True`` for unrestricted network. The default is none at all.
        writable: Further directories commands may write.
        deny_write: Paths that stay unwritable even inside a writable directory.
        deny_read: Paths commands cannot read.
        env: Environment variables every command receives, in addition to the
            allowlisted host ones.
        timeout: Seconds each command may run before it is killed; None for no limit.
        backend: ``"auto"`` for this platform's sandbox, or ``"seatbelt"``,
            ``"bubblewrap"`` or ``"passthrough"`` to force one.
        require_isolation: Raise rather than run unisolated on a platform with no
            sandbox (Windows).
        memory_strategy: How to enforce ``memory``; see
            :class:`~zett_agent.sandbox.resources.MemoryStrategy`.

    Examples:
        Let a build write one directory, with a budget::

            with SandboxSession("/work/project", memory="512MB", cpu_seconds=60) as sb:
                sb.run("python build.py", check=True)
                print(sb.run("ls dist").stdout)

        Run a program an agent wrote, in a directory that disappears afterwards::

            with SandboxSession(memory="256MB", timeout=30) as sb:
                (sb.workspace / "job.py").write_text(code)
                result = sb.run(["python3", "job.py"])
            # the workspace and its files are gone here

        From async code::

            async with SandboxSession(memory="256MB") as sb:
                result = await sb.arun(["python3", "job.py"])

    .. note::
        Limits apply to each command, not to the session as a whole: two commands
        run one after the other each get the full memory budget.

    .. warning::
        A path you pass as ``workspace`` or ``writable`` is yours and is never
        removed. Only directories the session creates itself are deleted. On macOS
        a program that writes a hard-coded path under ``/tmp`` instead of using
        ``$TMPDIR`` leaves that file behind, because the sandbox lets it write
        there; Linux gives each command a private ``/tmp`` that disappears.
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str] | None = None,
        *,
        memory: int | str | None = None,
        cpus: float | None = None,
        cpu_seconds: int | None = None,
        processes: int | None = None,
        max_file_size: int | str | None = None,
        open_files: int | None = None,
        network: bool | NetworkMode = False,
        writable: Iterable[str | os.PathLike[str]] = (),
        deny_write: Iterable[str | os.PathLike[str]] = (),
        deny_read: Iterable[str | os.PathLike[str]] = (),
        env: Mapping[str, str] | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
        backend: SandboxBackend | str = SandboxBackend.AUTO,
        require_isolation: bool = False,
        memory_strategy: MemoryStrategy | str = MemoryStrategy.AUTO,
    ) -> None:
        if cpus is not None and (isinstance(cpus, bool) or not isinstance(cpus, (int, float)) or cpus <= 0):
            raise SandboxPolicyError("cpus must be a positive number of cores, such as 0.5 or 2")
        # Validate the whole configuration now, so a mistake fails at the line that made it
        # rather than at the first command. The workspace may not exist yet, so the policy
        # itself is built when the session opens.
        self._limits = ResourceLimits(
            memory_bytes=None if memory is None else parse_size(memory, "memory"),
            cpu_seconds=cpu_seconds,
            cpu_percent=None if cpus is None else max(1, round(cpus * 100)),
            processes=processes,
            file_size_bytes=None if max_file_size is None else parse_size(max_file_size, "max_file_size"),
            open_files=open_files,
        )
        self._network = NetworkMode.FULL if network is True else NetworkMode.NONE if network is False else network
        NetworkMode(self._network)
        self._requested_workspace = None if workspace is None else Path(workspace).expanduser()
        self._writable = tuple(writable)
        self._deny_write = tuple(deny_write)
        self._deny_read = tuple(deny_read)
        self._env = dict(env or {})
        self._timeout = timeout
        self._backend = backend
        self._require_isolation = require_isolation
        self._memory_strategy = memory_strategy
        # Fail on an unknown backend or strategy here too.
        SandboxBackend(backend)
        MemoryStrategy(memory_strategy)

        self._state = "new"  # new -> open -> closed
        self._lock = Lock()
        self._sandbox: Sandbox | None = None
        self._root: Path | None = None
        self._workspace: Path | None = None
        self._scratch: Path | None = None
        self._finalizer: weakref.finalize | None = None
        self._runtime: SyncRuntime | None = None
        self._inflight: dict[asyncio.Task[object], asyncio.AbstractEventLoop] = {}

    # ------------------------------------------------------------ lifecycle

    @property
    def is_open(self) -> bool:
        """Whether commands can be run right now."""
        return self._state == "open"

    @property
    def workspace(self) -> Path:
        """The directory commands may write, which is also where they start by default.

        Raises:
            SandboxClosedError: Before the session is opened or after it is closed.
        """
        self._require_open()
        assert self._workspace is not None
        return self._workspace

    @property
    def sandbox(self) -> Sandbox:
        """The underlying :class:`~zett_agent.sandbox.base.Sandbox`, for the lower-level API."""
        self._require_open()
        assert self._sandbox is not None
        return self._sandbox

    def capabilities(self) -> ResourceReport:
        """Return how each limit is enforced on this machine; see :meth:`Sandbox.capabilities`."""
        return self.sandbox.capabilities()

    def _require_open(self) -> None:
        if self._state != "open":
            raise SandboxClosedError(
                "the session is not open: use it in a with block, or call open() first"
                if self._state == "new"
                else "the session is closed; create a new SandboxSession"
            )

    def open(self) -> SandboxSession:
        """Create the workspace and the sandbox. ``with`` calls this for you.

        Raises:
            SandboxUnavailableError: If the sandbox cannot run on this machine.
            SandboxPolicyError: If a path or a limit is invalid.
            SandboxError: If the session was already opened or closed.
        """
        with self._lock:
            if self._state != "new":
                raise SandboxError(f"the session cannot be opened: it is {self._state}")
            root = Path(tempfile.mkdtemp(prefix="zett-sandbox-")).resolve()
            try:
                scratch = root / "tmp"
                scratch.mkdir()
                if self._requested_workspace is None:
                    workspace = root / "workspace"
                    workspace.mkdir()
                else:
                    try:
                        workspace = self._requested_workspace.resolve(strict=True)
                    except OSError as error:
                        raise SandboxPolicyError(f"workspace does not exist: {self._requested_workspace}") from error
                    if not workspace.is_dir():
                        raise SandboxPolicyError(f"workspace is not a directory: {workspace}")
                policy = SandboxPolicy(
                    writable_paths=(workspace, scratch, *_as_paths(self._writable, "writable_paths")),
                    denied_write_paths=_as_paths(self._deny_write, "denied_write_paths"),
                    denied_read_paths=_as_paths(self._deny_read, "denied_read_paths"),
                    network=self._network,
                    limits=self._limits,
                )
                sandbox = create_sandbox(
                    policy,
                    backend=self._backend,
                    require_isolation=self._require_isolation,
                    memory_strategy=self._memory_strategy,
                )
            except BaseException:
                # A session that failed to open owns nothing: remove what it made so far.
                _destroy(str(root))
                raise
            self._root, self._workspace, self._scratch, self._sandbox = root, workspace, scratch, sandbox
            # Removes the directories even when the session is never closed, such as when
            # the interpreter exits or the object is dropped inside a with-less script.
            self._finalizer = weakref.finalize(self, _destroy, str(root))
            self._state = "open"
        return self

    def close(self) -> None:
        """Stop everything the session started and remove what it created.

        Commands still running are cancelled and their process groups and
        descendants are killed before anything is removed, so nothing keeps writing
        into a directory that is being deleted. A process that detached itself from
        its parent earlier and outlived it cannot be found on macOS; Linux's PID
        namespace ends it with the sandbox. Closing twice is harmless. A directory you
        supplied is left alone.
        """
        with self._lock:
            if self._state != "open":
                self._state = "closed" if self._state == "new" else self._state
                return
            self._state = "closed"
            inflight = list(self._inflight.items())
            runtime, self._runtime = self._runtime, None
        for task, loop in inflight:
            if not loop.is_closed():
                loop.call_soon_threadsafe(task.cancel)
        deadline = time.monotonic() + _CLOSE_WAIT_SECONDS
        while self._inflight and time.monotonic() < deadline:
            time.sleep(0.01)
        if runtime is not None:
            runtime.close()
        if self._finalizer is not None:
            self._finalizer.detach()
        if self._root is not None and not _destroy(str(self._root)):
            warnings.warn(
                f"the sandbox session could not remove its directory {self._root}; remove it by hand",
                ResourceWarning,
                stacklevel=2,
            )

    def __enter__(self) -> SandboxSession:
        return self.open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    async def __aenter__(self) -> SandboxSession:
        # Opening can probe the sandbox and touch the disk, so it stays off the loop.
        return await asyncio.to_thread(self.open)

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await asyncio.to_thread(self.close)

    def __repr__(self) -> str:
        return f"<SandboxSession {self._state} workspace={self._workspace or self._requested_workspace or 'temporary'}>"

    # --------------------------------------------------------------- running

    def _directory(self, cwd: str | os.PathLike[str] | None) -> Path:
        """Resolve a working directory: a relative one is relative to the workspace."""
        if cwd is None:
            return self.workspace
        path = Path(cwd).expanduser()
        return path if path.is_absolute() else self.workspace / path

    async def arun(
        self,
        command: str | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = _DEFAULT,  # type: ignore[assignment]
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        check: bool = False,
    ) -> SandboxResult:
        """Run a command in the sandbox and return what it produced. The async form of :meth:`run`.

        Raises:
            SandboxClosedError: If the session is not open.
            SandboxCommandError: If ``check`` is true and the command did not succeed.
            asyncio.CancelledError: If the session was closed while this ran.
        """
        self._require_open()
        directory = self._directory(cwd)
        task = asyncio.current_task()
        if task is not None:
            self._inflight[task] = asyncio.get_running_loop()
        try:
            result = await self.sandbox.run(
                command,
                cwd=directory,
                env={"TMPDIR": str(self._scratch), **self._env, **(env or {})},
                stdin=input,
                timeout=self._timeout if timeout is _DEFAULT else timeout,
                max_output_bytes=max_output_bytes,
            )
        finally:
            if task is not None:
                self._inflight.pop(task, None)
        if check and not result.succeeded:
            raise SandboxCommandError(_failure_message(command, result), result)
        return result

    def run(
        self,
        command: str | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = _DEFAULT,  # type: ignore[assignment]
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        check: bool = False,
    ) -> SandboxResult:
        """Run a command in the sandbox, wait for it, and return what it produced.

        It behaves like :func:`subprocess.run`. A string runs through ``/bin/sh -c``
        so pipes work; a list runs the program directly with nothing parsed, which is
        safer for anything built from untrusted text. It may be called from several
        threads, and from code that is itself inside an event loop.

        Args:
            command: A shell string, or the program followed by its arguments.
            cwd: Where to start; relative paths are relative to the workspace, which
                is the default.
            env: Variables added for this command only.
            input: Text or bytes to feed to its standard input.
            timeout: Seconds before it is killed; defaults to the session's.
            max_output_bytes: Bytes kept of each of standard output and error.
            check: Raise :class:`~zett_agent.sandbox.errors.SandboxCommandError`
                unless it exits with status 0 inside its time limit.

        Raises:
            SandboxClosedError: If the session is not open, or is closed while this runs.
            SandboxCommandError: If ``check`` is true and the command did not succeed.
        """
        self._require_open()
        runtime = self._sync_runtime()
        try:
            return runtime.call(
                self.arun,
                command,
                cwd=cwd,
                env=env,
                input=input,
                timeout=timeout,
                max_output_bytes=max_output_bytes,
                check=check,
            )
        except (asyncio.CancelledError, concurrent.futures.CancelledError, RuntimeError) as error:
            if self._state == "closed":
                raise SandboxClosedError("the session was closed while this command was running") from error
            raise

    def _sync_runtime(self) -> SyncRuntime:
        """Start the background loop that serves blocking calls, the first time one is made."""
        with self._lock:
            self._require_open()
            if self._runtime is None:
                self._runtime = SyncRuntime()
            return self._runtime


def _failure_message(command: str | Sequence[str], result: SandboxResult) -> str:
    """Describe a failed command in one readable block, ending with its stderr."""
    shown = command if isinstance(command, str) else " ".join(command)
    if result.timed_out:
        reason = "timed out"
    elif result.resources.limits_hit:
        reason = f"hit its {' and '.join(result.resources.limits_hit)} limit (exit status {result.exit_code})"
    else:
        reason = f"exited with status {result.exit_code}"
    tail = result.stderr.strip().splitlines()[-5:]
    return f"command {shown!r} {reason}" + ("\n" + "\n".join(tail) if tail else "")
