"""Choose the sandbox backend for this machine."""

from __future__ import annotations

import sys
import warnings

from .._compat import StrEnum
from .base import Sandbox
from .bubblewrap import BubblewrapSandbox
from .errors import SandboxPolicyError, SandboxUnavailableError
from .passthrough import PassthroughSandbox
from .policy import SandboxPolicy
from .resources import MemoryStrategy
from .seatbelt import SeatbeltSandbox

__all__ = ["SandboxBackend", "SandboxWarning", "create_sandbox", "default_backend"]


class SandboxBackend(StrEnum):
    """Which sandbox implementation to use."""

    #: Pick the one this platform has: Seatbelt on macOS, bubblewrap on Linux.
    AUTO = "auto"
    #: Seatbelt (macOS).
    SEATBELT = "seatbelt"
    #: bubblewrap (Linux).
    BUBBLEWRAP = "bubblewrap"
    #: No isolation; see :class:`~zett_agent.sandbox.passthrough.PassthroughSandbox`.
    PASSTHROUGH = "passthrough"


class SandboxWarning(UserWarning):
    """A sandbox was requested and this platform has none, so commands run unisolated."""


def default_backend() -> SandboxBackend:
    """Return the backend this platform uses when asked for ``AUTO``.

    Windows and every other platform report PASSTHROUGH, because no backend
    exists for them.
    """
    if sys.platform == "darwin":
        return SandboxBackend.SEATBELT
    if sys.platform.startswith("linux"):
        return SandboxBackend.BUBBLEWRAP
    return SandboxBackend.PASSTHROUGH


def create_sandbox(
    policy: SandboxPolicy | None = None,
    *,
    backend: SandboxBackend | str = SandboxBackend.AUTO,
    require_isolation: bool = False,
    memory_strategy: MemoryStrategy | str = MemoryStrategy.AUTO,
) -> Sandbox:
    """Create the sandbox for this machine, failing loudly rather than weakening it.

    A backend that exists but cannot run (bubblewrap missing, namespaces
    forbidden) raises :class:`SandboxUnavailableError`: the command is never run
    unsandboxed because of a problem the caller can fix. The one case that
    degrades is a platform with no backend at all, where ``AUTO`` returns a
    :class:`PassthroughSandbox` and emits a :class:`SandboxWarning`. Pass
    ``require_isolation=True`` to make that case an error too.

    Args:
        policy: The boundary to enforce. Defaults to a read-only policy.
        backend: A backend to force, or ``AUTO`` for this platform's own.
        require_isolation: Raise instead of returning a sandbox that isolates nothing.
        memory_strategy: How to enforce ``memory_bytes``; see
            :class:`~zett_agent.sandbox.resources.MemoryStrategy`.

    Raises:
        SandboxUnavailableError: If the backend cannot run here, or isolation is
            required and the platform has none.
        SandboxPolicyError: If ``backend`` is unknown or the policy is invalid.

    Examples:
        Take whatever this platform offers::

            sandbox = create_sandbox(SandboxPolicy.workspace_write("/work/project"))

        Refuse to run anywhere that cannot isolate::

            sandbox = create_sandbox(policy, require_isolation=True)
    """
    try:
        chosen = SandboxBackend(backend)
    except ValueError:
        raise SandboxPolicyError(f"unknown sandbox backend: {backend!r}") from None
    if chosen is SandboxBackend.AUTO:
        chosen = default_backend()
        if chosen is SandboxBackend.PASSTHROUGH:
            if require_isolation:
                raise SandboxUnavailableError(
                    f"No sandbox backend exists for platform {sys.platform!r}, and isolation was required."
                )
            warnings.warn(
                f"No sandbox backend exists for platform {sys.platform!r}; commands run without isolation.",
                SandboxWarning,
                stacklevel=2,
            )
    if chosen is SandboxBackend.PASSTHROUGH and require_isolation:
        raise SandboxUnavailableError("The passthrough sandbox does not isolate, and isolation was required.")
    implementations: dict[SandboxBackend, type[Sandbox]] = {
        SandboxBackend.SEATBELT: SeatbeltSandbox,
        SandboxBackend.BUBBLEWRAP: BubblewrapSandbox,
        SandboxBackend.PASSTHROUGH: PassthroughSandbox,
    }
    sandbox = implementations[chosen](policy, memory_strategy=memory_strategy)
    sandbox.check()
    return sandbox
