"""Local coding tools exposed as an Agent extension."""

from pathlib import Path

from ..agent import AgentRunContext
from ..tools.coding import run_shell
from .file_system import FileSystemExtension


class CodingExtension(FileSystemExtension):
    """Register writable filesystem tools followed by shell execution.

    The tools operate with host process permissions in the current working
    directory. Add ToolGuidelinesExtension separately when their prompt
    guidance should be included in model instructions.

    Examples:
        Combine coding tools with their model-facing guidance::

            agent = await Agent.create(
                model,
                config=AgentRunConfig(session_id="coding-session"),
                extensions=[CodingExtension(), ToolGuidelinesExtension()],
            )

    .. warning::
        ``run_shell`` and writable filesystem tools execute with the host
        process's permissions. This extension is a capability bundle, not a
        sandbox.

    .. seealso::
        :class:`~zett_agent.extensions.file_system.FileSystemExtension` supports read-only operation;
        :class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension` adds prompt guidance.
    """

    def __init__(self) -> None:
        super().__init__(read_only=False)

    def _working_directory_instructions(self, working_directory: Path) -> str:
        """Include the directory used by both file tools and shell commands."""
        return f"{super()._working_directory_instructions(working_directory)}\nShell commands run from this directory."

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register writable filesystem tools followed by shell execution."""
        await super().on_tool(context)
        context.register_tool(run_shell)
