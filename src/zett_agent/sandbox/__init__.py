"""Run commands and agent-written programs inside an operating-system sandbox.

Start with :class:`~zett_agent.sandbox.session.SandboxSession`, which takes a
workspace and a budget, runs commands like :func:`subprocess.run`, and removes
what it created when the ``with`` block ends: ``from zett_agent.sandbox import
SandboxSession``.

Underneath, every sandbox implements the same
:class:`~zett_agent.sandbox.base.Sandbox` interface and is configured by one
:class:`~zett_agent.sandbox.policy.SandboxPolicy`, so a host can switch backends
without changing how it runs a command, for example
``from zett_agent.sandbox import create_sandbox``.

A name loads its defining module the first time it is used, so importing this
package, or the policy, never imports a backend. The modules stay importable for
everything the package does not list.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .._lazy import lazy_exports as _lazy_exports

_MODULES = {
    "base": ("Sandbox", "SandboxResult", "SandboxedCommand"),
    "binaries": ("shell_argv",),
    "errors": (
        "SandboxClosedError",
        "SandboxCommandError",
        "SandboxError",
        "SandboxPolicyError",
        "SandboxUnavailableError",
    ),
    "factory": ("SandboxBackend", "SandboxWarning", "create_sandbox", "default_backend"),
    "linux": ("LinuxSandbox",),
    "macos": ("MacOSSandbox",),
    "passthrough": ("PassthroughSandbox",),
    "policy": ("NetworkMode", "ResourceLimits", "SandboxPolicy"),
    "session": ("SandboxSession",),
    "windows": ("WindowsSandbox",),
}
_EXPORTS = {name: f"{__name__}.{module}" for module, names in _MODULES.items() for name in names}

__all__ = tuple(_EXPORTS)
__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS)

if _TYPE_CHECKING:
    from .base import Sandbox as Sandbox
    from .base import SandboxedCommand as SandboxedCommand
    from .base import SandboxResult as SandboxResult
    from .binaries import shell_argv as shell_argv
    from .errors import SandboxClosedError as SandboxClosedError
    from .errors import SandboxCommandError as SandboxCommandError
    from .errors import SandboxError as SandboxError
    from .errors import SandboxPolicyError as SandboxPolicyError
    from .errors import SandboxUnavailableError as SandboxUnavailableError
    from .factory import SandboxBackend as SandboxBackend
    from .factory import SandboxWarning as SandboxWarning
    from .factory import create_sandbox as create_sandbox
    from .factory import default_backend as default_backend
    from .linux import LinuxSandbox as LinuxSandbox
    from .macos import MacOSSandbox as MacOSSandbox
    from .passthrough import PassthroughSandbox as PassthroughSandbox
    from .policy import NetworkMode as NetworkMode
    from .policy import ResourceLimits as ResourceLimits
    from .policy import SandboxPolicy as SandboxPolicy
    from .session import SandboxSession as SandboxSession
    from .windows import WindowsSandbox as WindowsSandbox
