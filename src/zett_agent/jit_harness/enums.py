"""Shared vocabulary for profiling and recent-error records.

These enums live in their own module because both :mod:`~zett_agent.jit_harness.profiling`
and :mod:`~zett_agent.jit_harness.errors` need them, while profiling imports the
error cache. Members are strings, so JSON export and comparison keep the plain
values.
"""

from __future__ import annotations

from .._compat import StrEnum


class ToolKind(StrEnum):
    """Representation of a reusable dynamic capability.

    Python tools execute trusted code. Text tools return Markdown instructions
    for the model to follow; reading them never executes their content.
    """

    PYTHON = "python"
    TEXT = "text"


class InspectionToolExposure(StrEnum):
    """How the model sees the harness's inspection and measurement tools.

    Those tools read retained source, schemas, statistics, call records, and
    recent errors. The capability-building loop does not need their definitions
    on every request, so an application can choose how much context they cost.

    ``DEFERRED``
        Offer them while letting a provider that supports deferred loading
        fetch each definition on demand. Protocols that ignore the flag
        (Chat Completions, Anthropic, Google, and Ollama) still receive every
        definition, so no capability is lost. This is the default.
    ``ALWAYS``
        Send every inspection definition with the first request, exactly as if
        the harness did not tier them.
    ``HIDDEN``
        Never expose them to the model. The application keeps the full
        ``JITHarness`` query API, but the model cannot read source, schemas,
        measurements, or error history on its own.
    """

    DEFERRED = "deferred"
    ALWAYS = "always"
    HIDDEN = "hidden"


class ToolOrigin(StrEnum):
    """Which registry served one dispatch attempt.

    The value records how the harness resolved the tool name at call time. It
    stays a plain string in JSON so exported profiles and model-facing results
    keep the values ``"base"``, ``"python"``, ``"text"``, ``"management"``, and ``"unknown"``.

    ``BASE``
        An application-owned tool: a constructor ``tools=[...]`` entry or one a
        tool-providing extension registered for the request. The harness keeps no
        Python source for it, so its implementation cannot be updated or saved.
        With ``manage_base_tools=True`` it can be unloaded/reloaded and profiled;
        snapshots retain its name and unload state, not its implementation.
    ``PYTHON``
        A session-local dynamic tool the model created with ``create_tool`` or
        replaced with ``update_tool``. An unloaded dynamic tool is still
        ``PYTHON``; the call status then reads ``rejected``.
    ``TEXT``
        A reusable Markdown workflow created with ``create_tool(kind="text")``.
        Calling it reads a bounded text page, not executable Python.
    ``MANAGEMENT``
        One of the reserved harness control functions: ``create_tool``,
        ``update_tool``, ``unload_tool``, ``load_tool``, ``list_tools``,
        ``inspect_tool``, source/schema readers, statistics/call queries, or
        ``inspect_tool_errors``. These belong to the harness
        rather than the application's toolbox.
    ``UNKNOWN``
        The name was in no registry when the attempt was recorded, usually a
        defensive fallback for a rejected call. It does not describe another kind
        of tool.
    """

    BASE = "base"
    PYTHON = "python"
    TEXT = "text"
    MANAGEMENT = "management"
    UNKNOWN = "unknown"


class ToolCallStatus(StrEnum):
    """How one dispatch attempt ended.

    The value stays a plain string in JSON, so exported profiles and recent-error
    reports keep ``"completed"``, ``"failed"``, ``"cancelled"``, and
    ``"rejected"``.

    ``COMPLETED``
        The handler returned. A later serialization or lifecycle-hook failure
        does not change this record; ``completed`` means the tool itself worked.
    ``FAILED``
        The tool started and raised: argument validation, handler code, or a
        nested dependency error the handler did not catch.
    ``CANCELLED``
        The awaiting caller was cancelled, for example by request cancellation or
        ``asyncio`` task cancellation. A synchronous worker may still be running
        because Python cannot forcibly stop it.
    ``REJECTED``
        The harness refused the call before any handler body ran: the tool is
        unloaded, missing its base binding, no longer registered in the current
        request, or the nested call exceeded ``max_call_depth``.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
