"""Filesystem tools exposed as an Agent extension."""

import json
from pathlib import Path

from ..agent import AgentRunContext
from ..messages import SystemMessage
from ..tools.base import AgentTool
from ..tools.coding import (
    delete_file,
    edit,
    glob,
    grep,
    read,
    read_file,
    write_file,
)
from ..tools.images import view_image
from .base import AgentExtension
from .tool import ToolExtension


class ReadFileExtension(ToolExtension):
    """Register ``read_file``, which returns a line range of a UTF-8 text file."""

    def __init__(self) -> None:
        super().__init__(read_file)


class ReadExtension(ToolExtension):
    """Register ``read``, which returns UTF-8 text or a local image.

    Text comes back as a line range with a continuation hint, exactly like
    ``read_file``. A PNG, JPEG, GIF, or WebP file comes back as an image plus a
    one-line description, so one tool covers both without the model choosing
    between them. It is separate from :class:`FileSystemExtension`, so add it
    next to (or instead of) :class:`ReadFileExtension` when the model should
    read images through the same tool.
    """

    def __init__(self) -> None:
        super().__init__(read)


class ViewImageExtension(ToolExtension):
    """Register ``view_image`` for local images and HTTP(S) image URLs."""

    def __init__(self) -> None:
        super().__init__(view_image)


class GlobExtension(ToolExtension):
    """Register ``glob`` to find paths by pattern."""

    def __init__(self) -> None:
        super().__init__(glob)


class GrepExtension(ToolExtension):
    """Register ``grep`` to search file contents by regular expression."""

    def __init__(self) -> None:
        super().__init__(grep)


class WriteFileExtension(ToolExtension):
    """Register ``write_file`` to create or replace a whole text file."""

    def __init__(self) -> None:
        super().__init__(write_file)


class EditExtension(ToolExtension):
    """Register ``edit`` to change an existing file by exact-text replacement."""

    def __init__(self) -> None:
        super().__init__(edit)


class DeleteFileExtension(ToolExtension):
    """Register ``delete_file`` to remove one regular file."""

    def __init__(self) -> None:
        super().__init__(delete_file)


class FileSystemExtension(AgentExtension):
    """Register filesystem tools with an optional read-only boundary.

    Relative paths start from the current working directory; absolute paths are
    accepted. Read-only mode exposes read_file, view_image, glob, and grep. Writable mode
    additionally exposes write_file, edit, and delete_file. Shell execution is
    deliberately not a filesystem capability because arbitrary commands cannot
    guarantee that they will leave the workspace unchanged.

    Each tool also has its own extension (:class:`ReadFileExtension`,
    :class:`EditExtension`, and so on) for an application that wants a subset.

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
        read_tools = (read_file, view_image, glob, grep)
        return read_tools if self.read_only else (*read_tools, write_file, edit, delete_file)

    async def on_tool(self, context: AgentRunContext) -> None:
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

    async def on_state(self, context: AgentRunContext) -> None:
        """Inject the request's resolved working directory into model context."""
        state = context.state
        message = SystemMessage(content=self._working_directory_instructions(Path.cwd().resolve()))
        instructions = [item for item in state.messages if isinstance(item, SystemMessage)]
        context.add_message(message, index=len(instructions))
