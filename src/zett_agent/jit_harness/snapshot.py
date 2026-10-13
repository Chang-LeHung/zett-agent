"""Versioned, non-executable snapshots of a session's JIT tool environment."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..tools.base import ToolExecutionMode

_MANAGEMENT_NAMES = frozenset(
    {
        "create_tool",
        "update_tool",
        "unload_tool",
        "load_tool",
        "list_tools",
        "inspect_tool",
        "inspect_tool_errors",
        "read_tool_source",
        "read_tool_schema",
        "inspect_tool_stats",
        "inspect_tool_calls",
    }
)
_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
#: The entry point module is the package root, so it is never listed in ``files``.
_ENTRYPOINT_FILE = "__init__.py"
_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MAX_PACKAGE_FILES = 32
_MAX_FILE_CHARS = 200_000
_MAX_PATH_CHARS = 96
_MAX_PATH_DEPTH = 4


def _validate_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError("Tool names need 1–64 letters, digits, underscores, or hyphens")
    if name in _MANAGEMENT_NAMES:
        raise ValueError(f"Tool name {name!r} is reserved for JIT management")
    return name


def _validate_file_path(path: str) -> str:
    """Validate one extra module path of a tool package.

    Paths are relative module paths, because a sibling is reached with
    ``from .helpers import value``. Every segment must be importable, so names
    with hyphens, spaces, ``..``, or another suffix cannot be registered.
    """
    if not isinstance(path, str) or not path or len(path) > _MAX_PATH_CHARS:
        raise ValueError(f"Package file paths need 1–{_MAX_PATH_CHARS} characters")
    if path.startswith("/") or "\\" in path or path.strip() != path:
        raise ValueError(f"Package file path must be a relative path: {path!r}")
    segments = path.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ValueError(f"Package file path must not contain empty or relative segments: {path!r}")
    if len(segments) > _MAX_PATH_DEPTH:
        raise ValueError(f"Package file path nests deeper than {_MAX_PATH_DEPTH}: {path!r}")
    if not segments[-1].endswith(".py"):
        raise ValueError(f"Package files must be Python modules: {path!r}")
    for segment in segments[:-1]:
        if not _SEGMENT.fullmatch(segment):
            raise ValueError(f"Package directory names must be Python identifiers: {path!r}")
    if path == _ENTRYPOINT_FILE:
        raise ValueError("The entry point is already this package's __init__ module")
    stem = segments[-1][: -len(".py")]
    if not _SEGMENT.fullmatch(stem) and not (stem == "__init__" and len(segments) > 1):
        raise ValueError(f"Package module names must be Python identifiers: {path!r}")
    return path


def _validate_files(files: Mapping[str, str]) -> dict[str, str]:
    """Validate one tool package's extra modules and return a plain copy."""
    if len(files) > _MAX_PACKAGE_FILES:
        raise ValueError(f"A tool package holds at most {_MAX_PACKAGE_FILES} files")
    validated: dict[str, str] = {}
    modules: dict[str, str] = {}
    for path, content in files.items():
        _validate_file_path(path)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Package file {path!r} cannot be blank")
        if len(content) > _MAX_FILE_CHARS:
            raise ValueError(f"Package file {path!r} exceeds {_MAX_FILE_CHARS} characters")
        module = path[: -len(".py")].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        if module in modules:
            raise ValueError(f"Package files {modules[module]!r} and {path!r} name the same module")
        modules[module] = path
        validated[path] = content
    return validated


class PythonToolSource(BaseModel):
    """Trusted source and the tool definition validated when it was registered.

    Reading this record does not execute Python. Activation recompiles the
    source and regenerates the schema; stored metadata is descriptive only.
    ``source`` is the entry point module, which is also the package root, so
    ``files`` holds the other modules of the package, imported relative to it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source: str = Field(min_length=1)
    entrypoint: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    guidelines: tuple[str, ...] = Field(min_length=1)
    snippet: str = ""
    execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL
    #: Additional modules, keyed by relative path such as ``helpers.py``.
    files: dict[str, str] = Field(default_factory=dict)

    _name = field_validator("name")(_validate_name)

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_files(value)

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError("entrypoint must be a Python identifier")
        return value


class TextToolSource(BaseModel):
    """Reusable Markdown workflow, stored verbatim and never executed as code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    #: Markdown steps, applicability, dependencies, pitfalls, and checks.
    source: str = Field(min_length=1, strict=True)
    #: Short selection hint exposed to the model; the workflow stays on demand.
    description: str = Field(min_length=1, max_length=512, strict=True)

    _name = field_validator("name")(_validate_name)

    @field_validator("source", "description")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text tool source and description cannot be blank")
        return value


class JITHarnessSnapshot(BaseModel):
    """Portable tool definitions and unload state, separate from history.

    Snapshots never contain callables, credentials, connections, or Python
    globals. Each activation uses the application's supplied base tools;
    missing base tools do not prevent loading.

    Examples:
        Read a snapshot without executing its Python::

            snapshot = JITHarnessSnapshot.load("harness.json")
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 1 stores single-module tools; 2 allows extra package files per tool.
    format_version: Literal[1, 2] = 1
    tools: tuple[PythonToolSource, ...] = ()
    #: Markdown workflows; older Python-only snapshots default to no text tools.
    text_tools: tuple[TextToolSource, ...] = ()
    base_tools: tuple[str, ...] = ()
    #: Retained names hidden from dispatch until load_tool restores them.
    unloaded_tools: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_names(self) -> JITHarnessSnapshot:
        if self.format_version == 1 and any(tool.files for tool in self.tools):
            raise ValueError("Multi-file tools require format_version 2")
        names = [tool.name for tool in (*self.tools, *self.text_tools)] + list(self.base_tools)
        if len(set(names)) != len(names):
            raise ValueError("Snapshot tool names must be unique")
        for name in names:
            _validate_name(name)
        if len(set(self.unloaded_tools)) != len(self.unloaded_tools):
            raise ValueError("Unloaded tool names must be unique")
        if not set(self.unloaded_tools) <= set(names):
            raise ValueError("Unloaded names must refer to snapshot tools")
        return self

    def to_json(self) -> str:
        """Serialize stable, readable JSON without running tool source."""
        payload = self.model_dump(mode="json")
        for tool in payload["tools"]:
            # A single-module tool keeps the shape it had before packages existed.
            if not tool["files"]:
                del tool["files"]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, value: str) -> JITHarnessSnapshot:
        """Validate JSON and its format version without running source."""
        return cls.model_validate_json(value)

    def save(self, path: str | Path) -> None:
        """Atomically replace an explicit file; its parent must already exist.

        A failed write leaves the old file intact. Saving is controlled by the
        application, never an automatic model operation.
        """
        destination = Path(path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self.to_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> JITHarnessSnapshot:
        """Read and validate a UTF-8 snapshot; Python is not executed."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
