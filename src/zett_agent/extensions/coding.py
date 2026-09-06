"""Local coding tools exposed as an Agent extension."""

from ..agent import AgentContext
from ..tools import run_shell
from .file_system import FileSystemExtension


class CodingExtension(FileSystemExtension):
    """Register writable filesystem tools followed by shell execution.

    The tools operate with host process permissions in the current working
    directory. Add ToolGuidelinesExtension separately when their prompt
    guidance should be included in model instructions.
    """

    def __init__(self) -> None:
        super().__init__(read_only=False)

    async def on_tool(self, context: AgentContext) -> None:
        """Register writable filesystem tools followed by shell execution."""
        await super().on_tool(context)
        context.register_tool(run_shell)
