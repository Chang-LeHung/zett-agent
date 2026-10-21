"""Tool framework and built-in tools.

The common names import straight from the package, for example
``from zett_agent.tools import tool, read_file``. Each name loads its defining module the first
time it is used, so importing this package imports no tool, no schema library, and no SDK. The
defining modules, such as ``zett_agent.tools.base`` and ``zett_agent.tools.coding``, stay importable
for everything the package does not list.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .._lazy import lazy_exports as _lazy_exports

_MODULES = {
    "base": ("AgentTool", "ToolExecutionMode", "ToolResult", "tool"),
    "coding": ("bash", "delete_file", "edit", "glob", "grep", "read", "read_file", "write_file"),
    "images": ("view_image",),
}
_EXPORTS = {name: f"{__name__}.{module}" for module, names in _MODULES.items() for name in names}

__all__ = tuple(_EXPORTS)
__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS)

if _TYPE_CHECKING:
    from .base import AgentTool as AgentTool
    from .base import ToolExecutionMode as ToolExecutionMode
    from .base import ToolResult as ToolResult
    from .base import tool as tool
    from .coding import bash as bash
    from .coding import delete_file as delete_file
    from .coding import edit as edit
    from .coding import glob as glob
    from .coding import grep as grep
    from .coding import read as read
    from .coding import read_file as read_file
    from .coding import write_file as write_file
    from .images import view_image as view_image
