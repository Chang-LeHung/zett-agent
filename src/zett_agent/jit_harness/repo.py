"""Load a session's tool repository from disk through Python's import system.

A repository is an ordinary Python package in a directory the model writes with
its file tools. This module owns the three things the harness adds to a normal
import: a finder that maps a throwaway package name onto that directory, a
loader that injects the ``harness`` global, and the discovery of every tool and
Markdown workflow the package defines.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import spec_from_file_location
from pathlib import Path
from threading import RLock
from types import CodeType, ModuleType
from typing import TYPE_CHECKING, Any

from ..ids import new_uuid7
from ..tools.base import AgentTool

if TYPE_CHECKING:
    from .runtime import JITHarness

#: Names the harness reserves for the model-facing repository tools.
MANAGEMENT_NAMES = frozenset({"repo_create", "repo_load", "repo_unload"})

_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
_REPO_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
_SESSION_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9_-])?\Z")

#: The package root; every repository has one.
INIT_FILE = "__init__.py"
#: Markdown workflows live here, one ``<name>.md`` file each.
WORKFLOWS_DIRECTORY = "workflows"
MAX_MODULES = 64
MAX_FILE_BYTES = 200_000
MAX_DESCRIPTION_CHARS = 512
MAX_FAILURE_CHARS = 2000
#: Traceback frames inside the repository that a failure report keeps.
MAX_FAILURE_FRAMES = 5

#: Loading installs a process-global import hook, so one thread loads at a time.
_LOAD_LOCK = RLock()

SKELETON = '"""Reusable tools for this session. ``repo_load`` registers every ``@tool`` defined in this package."""\n'


def validate_name(name: str) -> str:
    """Check a tool or workflow name and keep the reserved ones free."""
    if not _NAME.fullmatch(name):
        raise ValueError("Tool names need 1–64 letters, digits, underscores, or hyphens")
    if name in MANAGEMENT_NAMES:
        raise ValueError(f"Tool name {name!r} is reserved for JIT management")
    return name


def validate_repo_name(repo: str) -> str:
    """Check a repository directory name chosen by the model."""
    if not isinstance(repo, str) or not _REPO_NAME.fullmatch(repo):
        raise ValueError("Repository names need 1–64 letters, digits, underscores, or hyphens, starting with a letter")
    return repo


def session_directory_name(session_id: str) -> str:
    """Map any session id to one safe directory name.

    A conventional id is used as is. Anything else, such as a path separator, a
    leading dot, or a very long id, becomes a readable stem plus a digest, so two
    different ids never share a directory and none can leave the root.
    """
    if _SESSION_NAME.fullmatch(session_id):
        return session_id
    digest = sha256(session_id.encode()).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id).strip("_")[:48] or "session"
    return f"{stem}-{digest}"


@dataclass(frozen=True)
class Workflow:
    """One Markdown workflow read from ``workflows/<name>.md``."""

    name: str
    #: Short selection hint the model sees beside the reader tool.
    description: str
    #: Markdown body after the front matter.
    content: str
    #: Repository-relative path, for pages that quote the file.
    path: str


def parse_workflow(name: str, text: str) -> Workflow:
    """Split a workflow file into its ``description`` front matter and body."""
    validate_name(name)
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Workflow {name!r} must start with a front matter block: ---, description: ..., ---")
    end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end is None:
        raise ValueError(f"Workflow {name!r} has a front matter block that is never closed with ---")
    description: str | None = None
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or key.strip() != "description":
            raise ValueError(f"Workflow {name!r} front matter supports only 'description: ...', got {line!r}")
        description = value.strip()
        if len(description) > 1 and description[0] == description[-1] and description[0] in "\"'":
            description = description[1:-1].strip()
    if not description or len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"Workflow {name!r} needs a description of 1–{MAX_DESCRIPTION_CHARS} characters")
    content = "\n".join(lines[end + 1 :]).strip("\n")
    if not content.strip():
        raise ValueError(f"Workflow {name!r} has no content after its front matter")
    return Workflow(name, description, content, f"{WORKFLOWS_DIRECTORY}/{name}.md")


class RepoLoader(SourceFileLoader):
    """Internal: execute one repository module with ``harness`` in its globals."""

    def __init__(self, fullname: str, path: str, harness: JITHarness) -> None:
        super().__init__(fullname, path)
        self.harness = harness

    def get_code(self, fullname: str) -> CodeType:
        """Compile straight from the file, with no bytecode cache.

        A cached ``.pyc`` is keyed on the source's modification time and size, so
        two quick edits of the same length could load stale code, and the cache
        would also write into a directory the model reads.
        """
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)

    def exec_module(self, module: ModuleType) -> None:
        """Bind the harness before the body runs, so import-time code can use it."""
        module.__dict__["harness"] = self.harness
        super().exec_module(module)


class RepoFinder(MetaPathFinder):
    """Internal: resolve one throwaway package name onto the repository directory."""

    def __init__(self, alias: str, directory: Path, harness: JITHarness) -> None:
        self.alias = alias
        self.directory = directory
        self.harness = harness

    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> ModuleSpec | None:
        if fullname != self.alias and not fullname.startswith(self.alias + "."):
            return None
        relative = fullname[len(self.alias) + 1 :].split(".") if fullname != self.alias else []
        base = self.directory.joinpath(*relative)
        init = base / INIT_FILE
        if init.is_file():
            return spec_from_file_location(
                fullname,
                init,
                loader=RepoLoader(fullname, str(init), self.harness),
                submodule_search_locations=[str(base)],
            )
        module = base.parent / f"{base.name}.py"
        if relative and module.is_file():
            return spec_from_file_location(fullname, module, loader=RepoLoader(fullname, str(module), self.harness))
        if relative and base.is_dir():
            # A directory without __init__.py is a namespace package.
            spec = ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [str(base)]
            return spec
        return None


@dataclass
class LoadedRepo:
    """The imported repository: its package name, hook, tools, and workflows."""

    alias: str
    finder: RepoFinder
    tools: dict[str, AgentTool]
    workflows: dict[str, Workflow]

    def close(self) -> None:
        """Drop the import hook and every module it imported."""
        discard(self.alias, self.finder)


def discard(alias: str, finder: RepoFinder) -> None:
    """Remove a package's modules and hook, leaving no residue in the interpreter."""
    with _LOAD_LOCK:
        for name in [name for name in sys.modules if name == alias or name.startswith(alias + ".")]:
            del sys.modules[name]
        try:
            sys.meta_path.remove(finder)
        except ValueError:  # pragma: no cover - only reachable after external mutation
            pass


def _module_names(directory: Path) -> list[str]:
    """List the dotted names of every module below the package root."""
    if not (directory / INIT_FILE).is_file():
        raise ValueError(f"The repository has no {INIT_FILE}; it must stay a Python package")
    names: list[str] = []
    for current, directories, files in os.walk(directory):
        directories[:] = sorted(item for item in directories if not item.startswith(".") and item != "__pycache__")
        relative = Path(current).relative_to(directory)
        for file in sorted(files):
            if not file.endswith(".py") or file.startswith("."):
                continue
            path = Path(current) / file
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"{path.relative_to(directory).as_posix()} exceeds {MAX_FILE_BYTES} bytes")
            parts = [*relative.parts, file.removesuffix(".py")]
            if parts[-1] == "__init__":
                parts.pop()
            if not parts:
                continue
            if not all(part.isidentifier() for part in parts):
                raise ValueError(f"{path.relative_to(directory).as_posix()} is not an importable module name")
            names.append(".".join(parts))
    if len(names) > MAX_MODULES:
        raise ValueError(f"A repository holds at most {MAX_MODULES} modules")
    return names


def _read_workflows(directory: Path) -> dict[str, Workflow]:
    folder = directory / WORKFLOWS_DIRECTORY
    if not folder.is_dir():
        return {}
    workflows: dict[str, Workflow] = {}
    for path in sorted(folder.glob("*.md")):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{WORKFLOWS_DIRECTORY}/{path.name} exceeds {MAX_FILE_BYTES} bytes")
        workflows[path.stem] = parse_workflow(path.stem, path.read_text(encoding="utf-8"))
    return workflows


def _collect_tools(alias: str) -> dict[str, AgentTool]:
    """Gather the tools the package defines, ignoring ones it merely imported."""
    found: dict[str, AgentTool] = {}
    for name in sorted(name for name in sys.modules if name == alias or name.startswith(alias + ".")):
        for value in list(vars(sys.modules[name]).values()):
            if not isinstance(value, AgentTool):
                continue
            home = getattr(value.handler, "__module__", "")
            if home != alias and not home.startswith(alias + "."):
                continue
            validate_name(value.name)
            if value.local_tool_search or value.deferred:
                raise ValueError("JIT tools must be ordinary, non-deferred function tools")
            if found.setdefault(value.name, value) is not value:
                raise ValueError(f"Two tools are named {value.name!r}")
    return found


def describe_failure(error: Exception, directory: Path) -> str:
    """Summarize a failed load by where it failed inside the repository.

    The model has file tools, so a repository-relative file and line is all it
    needs; frames from the harness and the standard library are noise.
    """
    prefix = str(directory) + os.sep
    lines: list[str] = []
    frames = [frame for frame in traceback.extract_tb(error.__traceback__) if frame.filename.startswith(prefix)]
    for frame in frames[-MAX_FAILURE_FRAMES:]:
        where = f"{frame.filename[len(prefix) :]}:{frame.lineno} in {frame.name}"
        lines.append(f"{where}: {frame.line}" if frame.line else where)
    if isinstance(error, SyntaxError) and error.text and error.filename and error.filename.startswith(prefix):
        lines.append(f"{error.filename[len(prefix) :]}:{error.lineno}: {error.text.strip()}")
    lines.append(f"{type(error).__name__}: {str(error).replace(prefix, '')}")
    return "\n".join(lines)[:MAX_FAILURE_CHARS]


def load_repo(directory: Path, harness: JITHarness) -> LoadedRepo:
    """Import every module of a repository and collect its tools and workflows.

    Module bodies run here, with the application's privileges. On any failure
    the half-imported package is removed and a ``ValueError`` names the file and
    line, so a caller can keep serving the previous version.
    """
    modules = _module_names(directory)
    workflows = _read_workflows(directory)
    alias = f"zett_jit_{new_uuid7().replace('-', '_')}"
    finder = RepoFinder(alias, directory, harness)
    with _LOAD_LOCK:
        sys.meta_path.insert(0, finder)
        try:
            import_module(alias)
            for dotted in modules:
                import_module(f"{alias}.{dotted}")
            tools = _collect_tools(alias)
            clash = sorted(tools.keys() & workflows.keys())
            if clash:
                raise ValueError(f"A tool and a workflow share the name {clash[0]!r}")
        except BaseException as failure:
            discard(alias, finder)
            if isinstance(failure, Exception):
                raise ValueError(describe_failure(failure, directory)) from failure
            raise
    return LoadedRepo(alias, finder, tools, workflows)
