"""Shell execution exposed as its own Agent extension."""

from __future__ import annotations

from ..tools.coding import bash
from .tool import ToolExtension


class BashExtension(ToolExtension):
    """Register the ``bash`` tool and nothing else.

    The command runs with the host process's permissions in the current working
    directory, and ``timeout_seconds`` defaults to 60. Pair it with
    :class:`~zett_agent.extensions.shell_approval.ShellApprovalExtension` when a
    person must approve commands.

    Examples:
        Give a read-only agent a shell::

            extensions = [FileSystemExtension(read_only=True), BashExtension()]

    .. warning::
        This is a capability, not a sandbox: a command can change anything the
        process may change.
    """

    def __init__(self) -> None:
        super().__init__(bash)
