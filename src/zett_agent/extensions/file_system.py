"""Filesystem tools exposed as an Agent extension."""

import json
from pathlib import Path

from ..agent import AgentContext
from ..messages import SystemMessage
from ..tools import AgentTool, delete_file, glob, grep, read_file, replace_in_file, write_file
from .base import AgentExtension


class FileSystemExtension(AgentExtension):
    """Register filesystem tools with an optional read-only boundary.

    Relative paths start from the current working directory; absolute paths are
    accepted. Read-only mode exposes read_file, glob, and grep. Writable mode
    additionally exposes write_file and replace_in_file. Shell execution is
    deliberately not a filesystem capability because arbitrary commands cannot
    guarantee that they will leave the workspace unchanged.

    Examples:
        Usage::

            extension = FileSystemExtension(read_only=True)
            agent = await Agent.create(model, config=config, extensions=[extension])
    """

    def __init__(self, *, read_only: bool = False) -> None:
        self.read_only = read_only

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """Return the exact immutable registration set for the configured mode."""
        read_tools = (read_file, glob, grep)
        return read_tools if self.read_only else (*read_tools, write_file, replace_in_file, delete_file)

    async def on_tool(self, context: AgentContext) -> None:
        """Register the selected filesystem tools for this request."""
        for registered in self.tools:
            context.register_tool(registered)

    def _working_directory_instructions(self, working_directory: Path) -> str:
        """Describe the path base shared by this extension's tools."""
        rendered = json.dumps(str(working_directory), ensure_ascii=False)
        return (
            "# Filesystem environment\n"
            f"Current working directory: {rendered}.\n"
            "Relative filesystem paths are resolved from this directory. Absolute paths are also allowed."
        )

    async def on_state(self, context: AgentContext) -> None:
        """Inject the request's resolved working directory into model context."""
        state = context.state
        message = SystemMessage(content=self._working_directory_instructions(Path.cwd().resolve()))
        instructions = [item for item in state.messages if isinstance(item, SystemMessage)]
        dialogue = [item for item in state.messages if not isinstance(item, SystemMessage)]
        state.messages[:] = [*instructions, message, *dialogue]
