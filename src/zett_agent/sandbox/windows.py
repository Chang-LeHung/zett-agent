"""The Windows sandbox, which runs commands as they are because Windows has no backend yet."""

from __future__ import annotations

from .passthrough import PassthroughSandbox

__all__ = ["WindowsSandbox"]


class WindowsSandbox(PassthroughSandbox):
    """Run commands on Windows behind the sandbox interface, without isolation.

    Windows has no counterpart of Seatbelt or bubblewrap that this package
    drives, so the policy's paths and network setting are not enforced: the
    process can read and write whatever the user can and can use the network.
    What it still gives is everything the
    :class:`~zett_agent.sandbox.base.Sandbox` interface promises around the
    process: the filtered environment, the time limit, the output cap, and
    stopping the whole process tree. A host's code is then the same on every
    platform, and :attr:`~zett_agent.sandbox.base.Sandbox.isolated` tells the
    truth about the boundary.

    :func:`~zett_agent.sandbox.factory.create_sandbox` returns it on Windows and
    warns when it does; ``require_isolation=True`` turns that into an error.

    Examples:
        Run with no isolation, deliberately::

            sandbox = WindowsSandbox()
            result = await sandbox.run("echo hello")
            assert not result.isolated

    .. warning::
        This is not a sandbox. A command run through it has the full authority
        of the host process.
    """

    name = "windows"
