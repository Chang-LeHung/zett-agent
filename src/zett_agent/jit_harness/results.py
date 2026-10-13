"""JSON-ready results returned by harness APIs and their management tools.

These are TypedDicts rather than models because the payloads carry conditional
keys: source, entry points, and schema may not exist for every tool. Model query
payloads separate catalog, definitions, text fragments, statistics, calls, and
errors; the application's ``ToolInspection`` carries structure only, while
``JITHarness.profile`` and ``profile_stats`` own every measurement.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from .._compat import TypeAliasType
from .enums import ToolCallStatus, ToolOrigin

ToolStatus = TypeAliasType("ToolStatus", Literal["loaded", "unloaded", "missing", "unavailable"])


class ToolMutationResult(TypedDict):
    """Minimal acknowledgement of a successful tool state change."""

    name: str
    #: State after creation, update, unload, or load.
    status: ToolStatus


class ToolCatalogEntry(TypedDict):
    """One row of a session's tool catalog, as the model receives it."""

    name: str
    #: Registry that serves the tool; see ToolOrigin.
    origin: ToolOrigin
    #: ``loaded`` when callable now, ``unloaded`` when hidden but retained,
    #: ``missing`` when no implementation is bound, and ``unavailable`` when a
    #: later policy extension removed a loaded tool from the current request.
    status: ToolStatus
    #: Empty when no implementation is bound to describe.
    description: str


class ToolInspection(ToolCatalogEntry, total=False):
    """The application's detail payload from JITHarness.inspect_tool.

    Source and definitions depend on tool availability. The model's
    inspect_tool instead returns ToolDefinitionView without source. Measurements
    are not repeated here; read profile() or profile_stats() for them.
    """

    #: Retained Python or Markdown source; only dynamic tools have it.
    source: str
    #: Module variable holding the entry point; only dynamic Python tools have it.
    entrypoint: str
    #: JSON Schema of the arguments, from the stored source or the bound tool.
    parameters: dict[str, Any]
    #: Model-facing guidance, from the stored source or the bound tool.
    guidelines: list[str]
    #: Characters per package file, entry point included; only extra files add it.
    files: dict[str, int]


class ToolListPage(TypedDict):
    """A bounded model-facing catalog page, without profiling or definitions."""

    tools: list[ToolCatalogEntry]
    total_tools: int
    next_offset: int | None


class ToolDefinitionView(ToolCatalogEntry, total=False):
    """Model-facing definition; large schemas are read separately in chunks."""

    entrypoint: str
    parameters: dict[str, Any]
    #: True when the schema exceeded the response budget; use read_tool_schema.
    parameters_omitted: bool
    parameters_chars: int
    #: Characters per package file, entry point included; absent for one module.
    files: dict[str, int]


class ToolTextPage(TypedDict):
    """One exact source or schema fragment, without unrelated metadata."""

    name: str
    #: File this fragment came from; the entry point unless path was given.
    path: str
    content: str
    total_chars: int
    next_offset: int | None
    #: Source hash or schema hash; compare between pages if the tool was updated.
    content_id: str


class ToolCallView(TypedDict):
    """Required fields of one compact call measurement; the page names its tool."""

    call_id: str
    status: ToolCallStatus
    duration_ms: float
    input_chars: int | None
    output_chars: int | None


class ToolCallDetails(ToolCallView, total=False):
    """Context that appears only when it carries information.

    A top-level call has no composing caller, and a successful call has no error
    type; omitting those keys keeps the model-facing page small.
    """

    #: Tool whose handler composed this call; absent at the top level.
    caller_tool_name: str
    #: Exception class name for failures and rejections; absent on success.
    error_type: str


class ToolCallsPage(TypedDict):
    """A model-facing page of compact call records."""

    name: str
    calls: list[ToolCallDetails]
    total_calls: int
    next_offset: int | None


class ToolErrorView(TypedDict):
    """Actionable recent error information, without cache configuration."""

    tool_name: str
    error_type: str
    message: str | None
    occurrences: int
    last_call_id: str
    message_truncated: bool


class ToolErrorsPage(TypedDict):
    """A bounded model-facing page; full error reports remain application-side."""

    errors: list[ToolErrorView]
    total_matches: int
    next_offset: int | None
