"""macOS sandbox built on Seatbelt, the kernel facility behind ``sandbox-exec``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
from pathlib import Path
from typing import Final

from .base import Sandbox, SandboxedCommand
from .errors import SandboxUnavailableError
from .policy import NetworkMode, ResolvedPolicy, SandboxPolicy, resolve_policy
from .resources import MemoryStrategy

__all__ = ["SEATBELT_EXECUTABLE", "SeatbeltSandbox", "build_seatbelt_profile"]

SEATBELT_EXECUTABLE: Final = "/usr/bin/sandbox-exec"

#: Everything a process needs to start and run ordinary programs, and nothing
#: else. The profile denies by default, so a capability absent here is a
#: capability the process does not have.
_BASE_PROFILE = """\
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target self))
(allow sysctl-read)
(allow file-read*)
(allow file-write-data
  (literal "/dev/null")
  (literal "/dev/zero")
  (literal "/dev/dtracehelper")
  (literal "/dev/tty"))
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.system.logger")
  (global-name "com.apple.logd")
  (global-name "com.apple.system.notification_center")
  (global-name "com.apple.CoreServices.coreservicesd"))
(allow ipc-posix-shm-read-data)
"""

#: Extra services a program needs once the network is open: name resolution,
#: network configuration, and certificate validation.
_NETWORK_PROFILE = """\
(allow network*)
(allow mach-lookup
  (global-name "com.apple.dnssd.service")
  (global-name "com.apple.SystemConfiguration.DNSConfiguration")
  (global-name "com.apple.SystemConfiguration.configd")
  (global-name "com.apple.networkd")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.trustd")
  (global-name "com.apple.ocspd")
  (global-name "com.apple.securityd"))
"""

#: Temporary directories every program expects to write.
_SYSTEM_TEMP = "/private/tmp"


#: ``_CS_DARWIN_USER_TEMP_DIR`` from ``<unistd.h>``. Python's ``os.confstr`` does not
#: know the name, so it is read from libc directly.
_CS_DARWIN_USER_TEMP_DIR = 65537


def _user_temp_directory() -> Path | None:
    """Return the per-user temporary directory (what ``$TMPDIR`` normally holds), resolved.

    It is asked of macOS rather than read from the environment, because a
    host ``$TMPDIR`` can name any directory and must not become writable by
    being named. Returns None when macOS does not report one.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
        buffer = ctypes.create_string_buffer(1024)
        if libc.confstr(_CS_DARWIN_USER_TEMP_DIR, buffer, len(buffer)) == 0:
            return None
        return Path(os.fsdecode(buffer.value)).resolve()
    except (OSError, AttributeError, ValueError):
        return None


def _rules(action: str, operation: str, prefix: str, paths: tuple[Path, ...], params: dict[str, str]) -> list[str]:
    """Emit one rule per path, passing each path as a parameter instead of quoting it."""
    lines = []
    for index, path in enumerate(paths):
        key = f"{prefix}{index}"
        params[key] = str(path)
        lines.append(f'({action} {operation} (subpath (param "{key}")))')
    return lines


def build_seatbelt_profile(
    resolved: ResolvedPolicy, *, temp_directories: tuple[Path, ...] = ()
) -> tuple[str, dict[str, str]]:
    """Render a resolved policy as a Seatbelt profile and its parameters.

    Paths never appear in the profile text. Each is a ``(param "KEY")`` that
    ``sandbox-exec -D KEY=value`` fills in, so a path holding a quote, a
    parenthesis or a newline cannot change what the profile means.

    Rules are ordered so that the last match wins: permissions first, then the
    denials that override them.

    Returns:
        The profile text and the ``KEY -> value`` parameters it refers to.
    """
    params: dict[str, str] = {}
    lines = [_BASE_PROFILE.rstrip("\n")]
    lines += _rules("allow", "file-write*", "WRITE", (*resolved.writable, *temp_directories), params)
    lines += _rules("deny", "file-write*", "NOWRITE", resolved.unwritable, params)
    lines += _rules("deny", "file-read*", "NOREAD", resolved.denied_read, params)
    if resolved.network is NetworkMode.FULL:
        lines.append(_NETWORK_PROFILE.rstrip("\n"))
    return "\n".join(lines) + "\n", params


class SeatbeltSandbox(Sandbox):
    """Confine commands with a Seatbelt profile (macOS).

    The kernel enforces the profile on the process and every process it starts.
    Reads are open except the paths the policy hides; writes work only in the
    policy's writable paths and the temporary directories; ``denied_write_paths``
    and the protected files stay unwritable even inside a writable path; and the
    network is closed unless the policy opens it. A write is judged by where it
    lands, so a symbolic link that points out of a writable path gives nothing.

    Examples:
        Use it directly when you know you are on macOS::

            sandbox = SeatbeltSandbox(SandboxPolicy.workspace_write("/work/project"))
            result = await sandbox.run("make test", cwd="/work/project")

    .. note::
        ``sandbox-exec`` is deprecated by Apple but ships with macOS, and it is
        what Chrome, Bazel and the other agent sandboxes use for this.

    .. warning::
        A process that is already inside a Seatbelt sandbox cannot start another.
        Running under a host that sandboxes you makes :meth:`check` fail with a
        message that says so.
    """

    name = "seatbelt"
    isolated = True

    def __init__(
        self, policy: SandboxPolicy | None = None, *, memory_strategy: MemoryStrategy | str = MemoryStrategy.AUTO
    ) -> None:
        super().__init__(policy, memory_strategy=memory_strategy)
        self._checked = False

    def check(self) -> None:
        """Verify that ``sandbox-exec`` exists and can apply a profile from this process."""
        if self._checked:
            return
        if not os.access(SEATBELT_EXECUTABLE, os.X_OK):
            raise SandboxUnavailableError(
                f"{SEATBELT_EXECUTABLE} was not found. The Seatbelt sandbox needs macOS; "
                "other systems use BubblewrapSandbox."
            )
        try:
            probe = subprocess.run(
                [SEATBELT_EXECUTABLE, "-p", "(version 1)(allow default)", "/usr/bin/true"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxUnavailableError(f"{SEATBELT_EXECUTABLE} could not be started: {error}") from error
        if probe.returncode != 0:
            detail = probe.stderr.strip() or f"exit status {probe.returncode}"
            raise SandboxUnavailableError(
                f"{SEATBELT_EXECUTABLE} could not apply a profile: {detail}. "
                "A process that is already inside a sandbox, such as one started by another agent "
                "or by an IDE with its own sandbox, cannot start a nested one."
            )
        self._checked = True

    def _wrap(self, argv: tuple[str, ...], cwd: Path, env: dict[str, str]) -> SandboxedCommand:
        resolved = resolve_policy(self.policy)
        temp_directories = tuple(
            path for path in (Path(_SYSTEM_TEMP).resolve(), _user_temp_directory()) if path is not None
        )
        profile, params = build_seatbelt_profile(resolved, temp_directories=temp_directories)
        launcher = [SEATBELT_EXECUTABLE, "-p", profile]
        for key, value in params.items():
            launcher += ["-D", f"{key}={value}"]
        return SandboxedCommand(
            argv=(*launcher, *argv),
            env=env,
            cwd=cwd,
            warnings=resolved.warnings,
        )
