"""JSON-ready results returned by harness APIs and their management tools.

These are TypedDicts rather than models because they are small acknowledgements
that go straight into a tool message. Measurements are not here:
``JITHarness.profile`` and ``profile_stats`` own every one of them.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .._compat import TypeAliasType
from .enums import ToolOrigin

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


class RepoResult(TypedDict):
    """What the repository tools report back to the model."""

    #: The repository's name.
    repo: str
    #: Absolute directory of the repository; edit its files with file tools.
    path: str
    #: ``created`` right after a new repository was made, ``loaded`` while its
    #: tools are registered, and ``unloaded`` while they are hidden.
    status: Literal["created", "loaded", "unloaded"]
    #: Python tools and workflows the repository currently provides.
    tools: list[str]


class ToolTextPage(TypedDict):
    """One exact workflow fragment, without unrelated metadata."""

    name: str
    #: Repository-relative file this fragment came from.
    path: str
    content: str
    total_chars: int
    next_offset: int | None
    #: Content hash; compare between pages if the workflow was reloaded.
    content_id: str
