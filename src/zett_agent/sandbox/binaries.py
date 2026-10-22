"""Where the Linux sandbox finds ``bwrap``, including the build the wheel carries.

A minimal Linux image may have no bubblewrap, and installing a system package is
not always possible. The wheel therefore carries static ``bwrap`` builds for Linux
x86_64 and aarch64; the resolver prefers an explicit path, then ``bwrap`` on
``PATH``, then the bundled one. macOS and Windows never consult the bundle: they
use their own sandbox mechanisms.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

#: Directory holding one subdirectory of static binaries per Linux architecture.
BUNDLE_DIRECTORY = Path(__file__).resolve().parent.parent / "_bin"

#: ``platform.machine()`` spellings mapped to the bundle directory each one uses.
_LINUX_MACHINES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def bundled_bwrap(machine: str | None = None, *, platform_name: str | None = None) -> Path | None:
    """Return the bubblewrap shipped for this Linux machine, or None when there is none.

    Args:
        machine: ``platform.machine()`` value to resolve; this machine's by default.
        platform_name: ``sys.platform`` value to check; this machine's by default.
            Both exist so a test can ask what another platform would get.

    Only Linux consults the bundle, and only an architecture with a build gets one.
    """
    system = sys.platform if platform_name is None else platform_name
    if not system.startswith("linux"):
        return None
    resolved = _LINUX_MACHINES.get((platform.machine() if machine is None else machine).lower())
    if resolved is None:
        return None
    path = BUNDLE_DIRECTORY / f"linux-{resolved}" / "bwrap"
    return path if path.is_file() else None


def bwrap_path(explicit: str | None = None) -> str | None:
    """Return the bubblewrap program to launch, or None when nothing can serve as one.

    Args:
        explicit: A path or command the caller pinned; returned as given when set.

    The order is: the pinned value, then ``bwrap`` on ``PATH``, then the bundled
    static build. A caller that receives None reports that bubblewrap is missing and
    names both ways to provide it.
    """
    if explicit is not None:
        return explicit
    found = shutil.which("bwrap")
    if found is not None:
        return found
    bundled = bundled_bwrap()
    return None if bundled is None else str(bundled)


__all__ = ["BUNDLE_DIRECTORY", "bundled_bwrap", "bwrap_path"]
