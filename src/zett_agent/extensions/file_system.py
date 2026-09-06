"""Working-directory file tools exposed as an Agent extension."""

from ..agent import AgentContext
from ..tools import AgentTool, glob, grep, read_file, replace_in_file, write_file
from .base import AgentExtension


class FileSystemExtension(AgentExtension):
    """Register working-directory file tools with an optional read-only boundary.

    Read-only mode exposes read_file, glob, and grep. Writable mode additionally
    exposes write_file and replace_in_file. Shell execution is deliberately not
    a filesystem capability because arbitrary commands cannot guarantee that
    they will leave the workspace unchanged.

    Example:
        extension = FileSystemExtension(read_only=True)
        agent = await Agent.create(model, config=config, extensions=[extension])
    """

    def __init__(self, *, read_only: bool = False) -> None:
        self.read_only = read_only

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """Return the exact immutable registration set for the configured mode."""
        read_tools = (read_file, glob, grep)
        return read_tools if self.read_only else (*read_tools, write_file, replace_in_file)

    async def on_tool(self, context: AgentContext) -> None:
        """Register the selected filesystem tools for this request."""
        for registered in self.tools:
            context.register_tool(registered)
