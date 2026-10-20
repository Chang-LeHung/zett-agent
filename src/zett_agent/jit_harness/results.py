"""JSON-ready results returned by harness APIs and their management tools.

These are TypedDicts rather than models because they are small answers that go
straight into a tool message. The full measurements stay with
``JITHarness.profile`` and ``profile_stats``; :class:`ToolStatsReport` is the short
summary of them that a model reads to decide what to keep, fix, or build.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .._compat import TypeAliasType
from .enums import ToolOrigin
from .repo import TextKind

ToolStatus = TypeAliasType("ToolStatus", Literal["loaded", "unloaded", "missing", "unavailable"])


class ToolMutationResult(TypedDict):
    """Minimal acknowledgement of a successful tool state change."""

    name: str
    #: State after unload or load.
    status: ToolStatus


class ToolCatalogEntry(TypedDict):
    """One row of a session's tool catalog."""

    name: str
    #: Registry that serves the tool; see ToolOrigin.
    origin: ToolOrigin
    #: ``loaded`` when callable now, ``unloaded`` when hidden but retained,
    #: ``missing`` when no implementation is bound, and ``unavailable`` when a
    #: later policy extension removed a loaded tool from the current request.
    status: ToolStatus
    #: Empty when no implementation is bound to describe.
    description: str


class ResourceInfo(TypedDict):
    """One Markdown note of the repository, as far as its front matter says.

    The note is a file, not a tool: the model opens ``path`` with its file tools when
    ``description`` says the note applies.
    """

    name: str
    #: ``workflow``, ``skill``, ``instruction`` or ``memory``.
    kind: TextKind
    description: str
    #: Absolute path of the file.
    path: str


class RepoResult(TypedDict):
    """What the repository tools report back to the model."""

    #: The repository's name.
    repo: str
    #: Absolute directory of the repository; edit its files with file tools.
    path: str
    #: ``created`` right after a new repository was made, ``loaded`` while its
    #: tools are registered, and ``unloaded`` while they are hidden.
    status: Literal["created", "loaded", "unloaded"]
    #: Names of the Python tools the repository currently provides. Their schemas are in
    #: the request already, so only the names are repeated here.
    tools: list[str]
    #: The Markdown notes under ``resources/``, by name. They are listed while the
    #: repository is hidden too, because hiding tools does not touch files.
    resources: list[ResourceInfo]


class ToolStatsRow(TypedDict):
    """What one tool has done so far, in the few numbers that decide whether to keep it."""

    name: str
    #: Registry that serves the tool; see ToolOrigin.
    origin: ToolOrigin
    #: ``loaded`` when callable now, ``unloaded`` while hidden.
    status: ToolStatus
    #: Dispatch attempts, whatever their outcome.
    calls: int
    #: ``completed / (completed + failed)``, or None before the first finished call.
    #: Rejected and cancelled attempts say nothing about the tool and are left out.
    success_rate: float | None
    failed: int
    #: Mean wall time of an attempt in milliseconds, nested calls included.
    avg_ms: float | None
    #: Mean size of what a call handed back, in characters: what it costs to read. The
    #: message of a failed call counts, since the model reads that too.
    avg_output_chars: float | None
    #: Most recent distinct errors as ``Type: message``, shortened. One per tool in the
    #: overview, up to three for a tool asked for by name.
    recent_errors: list[str]


class ToolStatsReport(TypedDict):
    """How the asked-about tools are doing, or a bounded overview when none were named."""

    tools: list[ToolStatsRow]
    #: Tools left out of an overview to stay within the response budget; name one to see it.
    omitted: int
