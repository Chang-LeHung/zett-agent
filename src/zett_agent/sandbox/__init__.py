"""Run commands and agent-written programs inside an operating-system sandbox.

Start with :class:`~zett_agent.sandbox.session.SandboxSession`, which takes a
workspace and a budget, runs commands like :func:`subprocess.run`, and removes
what it created when the ``with`` block ends: ``from zett_agent.sandbox.session
import SandboxSession``.

Underneath, every sandbox implements the same
:class:`~zett_agent.sandbox.base.Sandbox` interface and is configured by one
:class:`~zett_agent.sandbox.policy.SandboxPolicy`, so a host can switch backends
without changing how it runs a command. Each backend is imported from its own
module, for example ``from zett_agent.sandbox.factory import create_sandbox``.
Keeping this package free of re-exports means importing the policy never imports
a backend.
"""
