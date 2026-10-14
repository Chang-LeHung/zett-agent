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
    the caller chooses :class:`~zett_agent.sandbox.passthrough.PassthroughSandbox`
    explicitly, or fixes the machine.
    """
