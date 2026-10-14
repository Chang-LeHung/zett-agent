"""The backend that runs commands as they are, for platforms with no sandbox."""

from __future__ import annotations

from pathlib import Path

from .base import Sandbox, SandboxedCommand

__all__ = ["PassthroughSandbox"]


class PassthroughSandbox(Sandbox):
    """Run commands with no isolation, behind the same interface as a real sandbox.

    The policy is not enforced: the process can read and write whatever the user
    can and can use the network. What it still gives is everything the
    :class:`~zett_agent.sandbox.base.Sandbox` interface promises around the
    process: the filtered environment, the time limit, the output cap, and
    stopping the whole process tree. That keeps a host's code identical on every
    platform, with :attr:`~Sandbox.isolated` telling the truth about the boundary.

    It is what :func:`~zett_agent.sandbox.factory.create_sandbox` returns on a
    platform with no backend, which is Windows today, and it warns when it does.
    It is never chosen in place of a backend that exists but cannot run.

    Examples:
        Run with no isolation, deliberately::

            sandbox = PassthroughSandbox()
            result = await sandbox.run("echo hello")
            assert not result.isolated

    .. warning::
        This is not a sandbox. A command run through it has the full authority
        of the host process.
    """

    name = "passthrough"
    isolated = False

    def check(self) -> None:
        """Always available."""

    def _wrap(self, argv: tuple[str, ...], cwd: Path, env: dict[str, str]) -> SandboxedCommand:
        return SandboxedCommand(
            argv=argv,
            env=env,
            cwd=cwd,
            warnings=("No sandbox is available: this command runs without isolation.",),
        )
