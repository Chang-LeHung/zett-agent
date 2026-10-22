"""The Linux sandbox, built on bubblewrap, which composes namespaces and bind mounts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .base import Sandbox, SandboxedCommand
from .binaries import bwrap_path
from .errors import SandboxUnavailableError
from .policy import NetworkMode, ResolvedPolicy, SandboxPolicy, resolve_policy
from .resources import MemoryStrategy

__all__ = ["LinuxSandbox", "build_bubblewrap_arguments"]

#: The directory a sandboxed process uses for temporary files. It is a private
#: tmpfs, so nothing written there outlives the run or is seen by the host.
_TEMP_DIRECTORY = "/tmp"


def _is_inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def build_bubblewrap_arguments(
    resolved: ResolvedPolicy,
    argv: tuple[str, ...],
    cwd: Path,
    env: dict[str, str],
    *,
    bwrap: str = "bwrap",
) -> tuple[list[str], list[str]]:
    """Render a resolved policy and a command as a ``bwrap`` argument vector.

    The root filesystem is mounted read-only, so the process can read the
    machine; each writable path is bound read-write over it; the unwritable
    paths are bound read-only again on top, and the denied-read paths are masked
    last. The mounts are applied in that order, which is what makes a denial
    beat a permission.

    The process gets new user, PID, IPC, UTS and cgroup namespaces, its own
    session (so it cannot inject input into the terminal), and dies with its
    parent. The network namespace is new unless the policy opens the network.

    Returns:
        The argument vector, and warnings for entries bubblewrap cannot enforce.
    """
    warnings: list[str] = []
    args = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user-try",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
    ]
    if resolved.network is NetworkMode.NONE:
        args.append("--unshare-net")
    args += ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", _TEMP_DIRECTORY]
    for path in resolved.writable:
        args += ["--bind", str(path), str(path)]

    # A path is bound read-only where it really is, so a symbolic link is
    # followed first. Only paths that exist can be mounted over.
    bound: set[Path] = set()
    for original in resolved.unwritable:
        target = original.resolve(strict=False)
        if not os.path.lexists(target):
            if original in resolved.denied_write:
                warnings.append(f"denied_write_paths entry does not exist and cannot be enforced here: {original}")
            continue
        if target in bound or not _is_inside(target, resolved.writable):
            continue
        bound.add(target)
        args += ["--ro-bind", str(target), str(target)]

    for original in resolved.denied_read:
        if not os.path.lexists(original):
            warnings.append(f"denied_read_paths entry does not exist and cannot be enforced here: {original}")
            continue
        if original.is_dir() and not original.is_symlink():
            args += ["--tmpfs", str(original), "--remount-ro", str(original)]
        else:
            args += ["--ro-bind", "/dev/null", str(original)]

    args.append("--clearenv")
    environment = {**env, "TMPDIR": _TEMP_DIRECTORY}
    for key, value in environment.items():
        args += ["--setenv", key, value]
    args += ["--chdir", str(cwd), "--", *argv]
    return args, warnings


class LinuxSandbox(Sandbox):
    """Confine commands on Linux with bubblewrap namespaces and read-only mounts.

    The platform class is the stable name; bubblewrap is how it is implemented.
    :func:`build_bubblewrap_arguments` renders the policy as a ``bwrap`` command
    line and each command runs behind it.

    The root filesystem is read-only; only the policy's writable paths and a
    private ``/tmp`` accept writes; the unwritable and protected paths are bound
    read-only again; credential paths are masked; the network namespace is new
    unless the policy opens the network; and process, IPC and UTS namespaces are
    private. Everything the process starts is inside the same boundary.

    Examples:
        Use it directly on Linux::

            sandbox = LinuxSandbox(SandboxPolicy.workspace_write("/work/project"))
            result = await sandbox.run(["python3", "build.py"], cwd="/work/project")

    .. note::
        Bubblewrap needs unprivileged user namespaces. :meth:`check` runs a
        probe once and, if it fails, says which of the usual causes to look at.

    .. warning::
        This backend adds no seccomp filter. With the network open, the process
        shares the host's network namespace and so can reach local services that
        listen there, including abstract Unix sockets; closing the network
        namespace (the default) isolates them. A protected path that does not
        exist yet cannot be masked, so it is only protected once it exists.
    """

    name = "linux"
    isolated = True

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        *,
        bwrap_path: str | os.PathLike[str] | None = None,
        memory_strategy: MemoryStrategy | str = MemoryStrategy.AUTO,
    ) -> None:
        super().__init__(policy, memory_strategy=memory_strategy)
        self._bwrap_path = None if bwrap_path is None else str(bwrap_path)
        self._executable: str | None = None

    def check(self) -> None:
        """Verify that ``bwrap`` exists and can create the namespaces this sandbox uses."""
        if self._executable is not None:
            return
        if not sys.platform.startswith("linux"):
            raise SandboxUnavailableError(
                "LinuxSandbox needs Linux. macOS uses MacOSSandbox; Windows uses WindowsSandbox, which does not isolate."
            )
        executable = self._bwrap_path or bwrap_path()
        if executable is None or not os.access(executable, os.X_OK):
            raise SandboxUnavailableError(
                "bubblewrap (bwrap) was not found, and this wheel carries no build for this architecture. "
                "Install it with your package manager, for example `apt install bubblewrap` or "
                "`dnf install bubblewrap`."
            )
        probe_arguments = [
            executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-user-try",
            "--unshare-pid",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "true",
        ]
        try:
            probe = subprocess.run(probe_arguments, capture_output=True, text=True, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxUnavailableError(f"bwrap could not be started: {error}") from error
        if probe.returncode != 0:
            detail = probe.stderr.strip() or f"exit status {probe.returncode}"
            raise SandboxUnavailableError(
                f"bwrap cannot create a sandbox here: {detail}. Common causes: unprivileged user namespaces are "
                "disabled (on Ubuntu 24.04, set kernel.apparmor_restrict_unprivileged_userns=0 or load the "
                "bwrap AppArmor profile); the process runs in a container that forbids namespaces (Docker needs "
                "relaxed seccomp/AppArmor or --privileged); or this is WSL1, which has no user namespaces."
            )
        self._executable = executable

    def _wrap(self, argv: tuple[str, ...], cwd: Path, env: dict[str, str]) -> SandboxedCommand:
        resolved = resolve_policy(self.policy)
        arguments, warnings = build_bubblewrap_arguments(resolved, argv, cwd, env, bwrap=self._executable or "bwrap")
        return SandboxedCommand(
            argv=tuple(arguments),
            # bwrap clears the sandbox environment itself; the launcher only needs a PATH to be found.
            env={**env, "TMPDIR": _TEMP_DIRECTORY},
            cwd=cwd,
            warnings=(*resolved.warnings, *warnings),
        )
