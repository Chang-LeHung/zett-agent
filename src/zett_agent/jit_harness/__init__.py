"""Session-local tools and Markdown notes that a model keeps in a repository on disk.

The common names import straight from the package, for example
``from zett_agent.jit_harness import JITHarnessExtension``. A name loads its defining
module the first time it is used, so importing this package imports neither the runtime
nor a provisioning class until one is needed.

A repository declares itself in ``MANIFEST`` (name, version, description, and optionally
the tools it promises) and may list forbidden tool names in ``BAN_TOOLS``. Both are
ordinary files the model reads and writes; :func:`inspect_repository` checks them without
importing anything or starting a session.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .._lazy import lazy_exports as _lazy_exports

_MODULES = {
    "enums": ("ToolCallStatus", "ToolOrigin"),
    "errors": ("ToolErrorRecord", "ToolErrorReport"),
    "extension": (
        "JIT_TOOL_COMPLETED_EVENT_NAME",
        "JIT_TOOL_STARTED_EVENT_NAME",
        "JITHarnessExtension",
        "JITToolCompletedEvent",
        "JITToolStartedEvent",
    ),
    "host": (
        "HarnessCalls",
        "HostedRepo",
        "InProcessToolHost",
        "JITHostCrashed",
        "JITHostError",
        "JITHostTimeout",
        "JITToolHost",
        "SubprocessToolHost",
        "ToolNotSerializable",
        "ToolSpec",
    ),
    "profiling": ("JITHarnessProfile", "ToolCallProfile", "ToolProfileStats"),
    "manifest": (
        "BAN_TOOLS_FILE",
        "MANIFEST_FILE",
        "Manifest",
        "RepositoryPolicy",
        "inspect_repository",
        "is_repository_name",
        "is_tool_name",
        "read_banned_tools",
        "read_manifest",
    ),
    "repo": ("TextKind",),
    "results": (
        "RepoResult",
        "ResourceInfo",
        "ToolCatalogEntry",
        "ToolMutationResult",
        "ToolStatsReport",
        "ToolStatsRow",
        "ToolStatus",
    ),
    "runtime": ("JITHarness",),
}
_EXPORTS = {name: f"{__name__}.{module}" for module, names in _MODULES.items() for name in names}

__all__ = tuple(_EXPORTS)
__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS)

if _TYPE_CHECKING:
    from .enums import ToolCallStatus as ToolCallStatus
    from .enums import ToolOrigin as ToolOrigin
    from .errors import ToolErrorRecord as ToolErrorRecord
    from .errors import ToolErrorReport as ToolErrorReport
    from .extension import JIT_TOOL_COMPLETED_EVENT_NAME as JIT_TOOL_COMPLETED_EVENT_NAME
    from .extension import JIT_TOOL_STARTED_EVENT_NAME as JIT_TOOL_STARTED_EVENT_NAME
    from .extension import JITHarnessExtension as JITHarnessExtension
    from .extension import JITToolCompletedEvent as JITToolCompletedEvent
    from .extension import JITToolStartedEvent as JITToolStartedEvent
    from .host import HarnessCalls as HarnessCalls
    from .host import HostedRepo as HostedRepo
    from .host import InProcessToolHost as InProcessToolHost
    from .host import JITHostCrashed as JITHostCrashed
    from .host import JITHostError as JITHostError
    from .host import JITHostTimeout as JITHostTimeout
    from .host import JITToolHost as JITToolHost
    from .host import SubprocessToolHost as SubprocessToolHost
    from .host import ToolNotSerializable as ToolNotSerializable
    from .host import ToolSpec as ToolSpec
    from .manifest import BAN_TOOLS_FILE as BAN_TOOLS_FILE
    from .manifest import MANIFEST_FILE as MANIFEST_FILE
    from .manifest import Manifest as Manifest
    from .manifest import RepositoryPolicy as RepositoryPolicy
    from .manifest import inspect_repository as inspect_repository
    from .manifest import is_repository_name as is_repository_name
    from .manifest import is_tool_name as is_tool_name
    from .manifest import read_banned_tools as read_banned_tools
    from .manifest import read_manifest as read_manifest
    from .profiling import JITHarnessProfile as JITHarnessProfile
    from .profiling import ToolCallProfile as ToolCallProfile
    from .profiling import ToolProfileStats as ToolProfileStats
    from .repo import TextKind as TextKind
    from .results import RepoResult as RepoResult
    from .results import ResourceInfo as ResourceInfo
    from .results import ToolCatalogEntry as ToolCatalogEntry
    from .results import ToolMutationResult as ToolMutationResult
    from .results import ToolStatsReport as ToolStatsReport
    from .results import ToolStatsRow as ToolStatsRow
    from .results import ToolStatus as ToolStatus
    from .runtime import JITHarness as JITHarness
