"""The external programs the sandbox launches, and what to do when a system lacks them.

A command given as a string runs through a shell. macOS and an ordinary Linux
image have ``/bin/sh``; a minimal container may not, so the wheel carries static
BusyBox builds for Linux x86_64 and aarch64 and runs the ``sh`` applet when the
system has no shell. Process counts and the process tree read ``/proc`` on Linux,
so no ``ps`` is needed there; macOS keeps using its own ``ps``.
"""

from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import SandboxUnavailableError

#: Directory holding one subdirectory of static binaries per Linux architecture.
BUNDLE_DIRECTORY = Path(__file__).resolve().parent.parent / "_bin"

#: ``platform.machine()`` spellings mapped to the bundle directory each one uses.
_LINUX_MACHINES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def bundled_busybox(machine: str | None = None, *, platform_name: str | None = None) -> Path | None:
    """Return the BusyBox shipped for this Linux machine, or None when there is none.

    Args:
        machine: ``platform.machine()`` value to resolve; this machine's by default.
        platform_name: ``sys.platform`` value to check; this machine's by default.
            Both exist so a test can ask what another platform would get.

    Only Linux consults the bundle: the other supported platforms always have the
    programs the runtime needs.
    """
    system = sys.platform if platform_name is None else platform_name
    if not system.startswith("linux"):
        return None
    resolved = _LINUX_MACHINES.get((platform.machine() if machine is None else machine).lower())
    if resolved is None:
        return None
    path = BUNDLE_DIRECTORY / f"linux-{resolved}" / "busybox"
    return path if path.is_file() else None


def shell_argv(shell: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """Return the command prefix that runs one shell command string.

    The order is: the shell the caller names, then ``sh`` on ``PATH``, then the
    bundled BusyBox applet on Linux. An explicit shell is returned as given, so a
    caller can pin ``/bin/bash`` or a multi-word launcher.

    Args:
        shell: Shell to use, or None to resolve one.

    Returns:
        The program and any fixed arguments, ready for ``-c`` and the command.

    Raises:
        SandboxUnavailableError: No shell is available to run a command string.
    """
    if shell is not None:
        return (shell,) if isinstance(shell, str) else tuple(shell)
    system = shutil.which("sh")
    if system is not None:
        return (system,)
    busybox = bundled_busybox()
    if busybox is not None:
        return (str(busybox), "sh")
    raise SandboxUnavailableError(
        "No shell is available to run a command string: install one on PATH, "
        "or install the zett-agent wheel that carries BusyBox for this architecture"
    )


__all__ = ["BUNDLE_DIRECTORY", "bundled_busybox", "shell_argv"]
