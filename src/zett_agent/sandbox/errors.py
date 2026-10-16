"""Errors raised by the sandbox layer."""

from ..exceptions import AgentError


class SandboxError(AgentError):
    """Base error for sandbox failures."""


class SandboxPolicyError(SandboxError, ValueError):
    """A policy is malformed or names something the sandbox cannot enforce.

    It is also a ``ValueError``, so code that validates user input with the
    standard exception keeps working.
    """


class SandboxUnavailableError(SandboxError):
    """The requested sandbox cannot run on this machine.

    The message says why and, where a fix exists, what to install or change.
    The sandbox layer never answers this by running the command unsandboxed:
    the caller chooses a sandbox that does not isolate, such as
    :class:`~zett_agent.sandbox.windows.WindowsSandbox`, explicitly, or fixes the
    machine.
    """


class SandboxClosedError(SandboxError):
    """The session was closed (or never opened), so it cannot run anything."""


class SandboxCommandError(SandboxError):
    """A command run with ``check=True`` did not succeed.

    Attributes:
        result: The full :class:`~zett_agent.sandbox.base.SandboxResult`, so the
            output and the resource report are still available to the handler.
    """

    def __init__(self, message: str, result: object) -> None:
        super().__init__(message)
        self.result = result
