"""Load a session's tool repository from disk through Python's import system.

A repository is an ordinary Python package in a directory the model writes with
its file tools. This module owns the three things the harness adds to a normal
import: a finder that maps a throwaway package name onto that directory, a
loader that injects the ``harness`` global, and the discovery of every tool the
package defines. Markdown notes (workflow, skill, instruction, memory) beside it
are only described, never imported: the model reads them with its file tools.
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

from .._compat import StrEnum
from ..ids import new_uuid7
from ..tools.base import AgentTool
from .manifest import (
    MANAGEMENT_NAMES,
    TOOL_NAME,
    RepositoryPolicy,
    _tool_problems,
    inspect_repository,
    is_repository_name,
    is_tool_name,
)

if TYPE_CHECKING:
    from .host import HarnessCalls

#: Names the harness reserves for its model-facing management tools, re-exported here.
# (no reassignment: the import above puts the name in this module's namespace)

_NAME = TOOL_NAME
_REPO_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
_SESSION_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9_-])?\Z")

#: The package root; every repository has one.
INIT_FILE = "__init__.py"
#: Every throwaway package name starts with this, which is how the finder tells its own imports apart.
ALIAS_PREFIX = "zett_jit_"
#: Markdown resources live here, one ``<name>.md`` file each.
RESOURCES_DIRECTORY = "resources"
MAX_MODULES = 64
#: Notes a listing carries. Each shows a description of up to 512 characters, so the cap keeps
#: the answer of ``repo_load`` within what a model reads at once.
MAX_RESOURCES = 50
MAX_FILE_BYTES = 200_000
MAX_DESCRIPTION_CHARS = 512
MAX_FAILURE_CHARS = 2000
#: Traceback frames inside the repository that a failure report keeps.
MAX_FAILURE_FRAMES = 5

#: Loading installs a process-global import hook, so one thread loads at a time.
_LOAD_LOCK = RLock()

SKELETON = '"""Reusable tools for this session. ``repo_load`` registers every ``@tool`` defined in this package."""\n'
#: A new repository starts with a manifest that loads: the model rewrites it as the
#: repository grows, and ``repo_load`` refuses a repository without one.
MANIFEST_SKELETON = 'name = "{name}"\nversion = "0.1.0"\ndescription = "Reusable tools for this session."\n'


def validate_name(name: str) -> str:
    """Check a tool name and keep the reserved ones free.

    The pattern lives in :mod:`~zett_agent.jit_harness.manifest`, so a name the loader
    accepts is also a name a manifest can declare and a ban list can forbid.
    """
    if not is_tool_name(name):
        raise ValueError("Tool names need 1–64 letters, digits, underscores, or hyphens")
    if name in MANAGEMENT_NAMES:
        raise ValueError(f"Tool name {name!r} is reserved for JIT management")
    return name


def validate_repo_name(repo: str) -> str:
    """Check a repository directory name chosen by the model."""
    if not is_repository_name(repo):
        raise ValueError(
            "Repository names need 1–64 letters, digits, underscores, or hyphens, starting with a letter or digit"
        )
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


class TextKind(StrEnum):
    """What a Markdown resource is for, declared by ``kind:`` in its front matter.

    The kind tells the model how far to trust the text and what to do with it.

    ``WORKFLOW``
        An ordered procedure that worked, to follow step by step after checking
        that it applies. This is the default.
    ``SKILL``
        Know-how for a tool, library, or technique: how to use it well and which
        mistakes to avoid. Applied where it fits, not followed as a sequence.
    ``INSTRUCTION``
        A standing rule or preference to honor whenever its description matches
        the work at hand.
    ``MEMORY``
        Distilled facts and key points from earlier work, such as findings,
        decisions, and locations. It can be out of date, so verify before relying
        on it.
    """

    WORKFLOW = "workflow"
    SKILL = "skill"
    INSTRUCTION = "instruction"
    MEMORY = "memory"


@dataclass(frozen=True)
class TextResource:
    """What the front matter of ``resources/<name>.md`` says about the note.

    The harness reads the metadata and nothing else: the model opens the file itself
    with its file tools, so the body is never held or injected.
    """

    name: str
    kind: TextKind
    #: Short hint of when the note applies; what a listing shows beside the name.
    description: str
    #: Path relative to the repository root.
    path: str


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def parse_resource(name: str, text: str) -> TextResource:
    """Read the ``kind`` and ``description`` front matter of a resource, and check it has a body.

    A resource is a note, not a tool, so its name may be any file name of 1–64 letters,
    digits, underscores, or hyphens; it does not have to avoid the reserved tool names.
    """
    if not _NAME.fullmatch(name):
        raise ValueError(f"Resource {name!r}: names need 1–64 letters, digits, underscores, or hyphens")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"Resource {name!r} must start with a front matter block: ---, kind: ..., description: ..., ---"
        )
    end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end is None:
        raise ValueError(f"Resource {name!r} has a front matter block that is never closed with ---")
    description: str | None = None
    kind = TextKind.WORKFLOW
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or key not in {"description", "kind"}:
            raise ValueError(
                f"Resource {name!r} front matter supports only 'kind: ...' and 'description: ...', got {line!r}"
            )
        if key == "description":
            description = _unquote(value)
            continue
        try:
            kind = TextKind(_unquote(value).lower())
        except ValueError:
            allowed = ", ".join(item.value for item in TextKind)
            raise ValueError(f"Resource {name!r} has kind {_unquote(value)!r}; use one of: {allowed}") from None
    if not description or len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"Resource {name!r} needs a description of 1–{MAX_DESCRIPTION_CHARS} characters")
    if not "\n".join(lines[end + 1 :]).strip():
        raise ValueError(f"Resource {name!r} has no content after its front matter")
    return TextResource(name, kind, description, f"{RESOURCES_DIRECTORY}/{name}.md")


class RepoLoader(SourceFileLoader):
    """Internal: execute one repository module with ``harness`` in its globals."""

    def __init__(self, fullname: str, path: str, harness: HarnessCalls) -> None:
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
    """Internal: resolve the throwaway package name of every loaded repository.

    One instance serves the whole process. ``sys.meta_path`` is consulted for every
    import the interpreter makes, so a finder per repository would make each import
    slower with every session, and one that was never closed would stay for good.
    Here the cost is a prefix test and a dictionary lookup however many repositories
    are loaded, and the finder is on ``sys.meta_path`` only while one is.
    """

    def __init__(self) -> None:
        self._roots: dict[str, tuple[Path, HarnessCalls]] = {}

    @property
    def empty(self) -> bool:
        """Whether no repository is mounted."""
        return not self._roots

    def mount(self, alias: str, directory: Path, harness: HarnessCalls) -> None:
        """Serve ``alias`` and its submodules from ``directory``."""
        self._roots[alias] = (directory, harness)

    def unmount(self, alias: str) -> None:
        """Stop serving ``alias``."""
        self._roots.pop(alias, None)

    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> ModuleSpec | None:
        if not fullname.startswith(ALIAS_PREFIX):
            return None
        alias, _, rest = fullname.partition(".")
        root = self._roots.get(alias)
        if root is None:
            return None
        directory, harness = root
        relative = rest.split(".") if rest else []
        base = directory.joinpath(*relative)
        init = base / INIT_FILE
        if init.is_file():
            return spec_from_file_location(
                fullname,
                init,
                loader=RepoLoader(fullname, str(init), harness),
                submodule_search_locations=[str(base)],
            )
        module = base.parent / f"{base.name}.py"
        if relative and module.is_file():
            return spec_from_file_location(fullname, module, loader=RepoLoader(fullname, str(module), harness))
        if relative and base.is_dir():
            # A directory without __init__.py is a namespace package.
            spec = ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [str(base)]
            return spec
        return None


@dataclass
class LoadedRepo:
    """The imported repository: its package name, its tools, and what its sources looked like."""

    alias: str
    tools: dict[str, AgentTool]
    #: Digest of every Python source at import time; equal digests mean nothing needs importing.
    fingerprint: str

    def close(self) -> None:
        """Drop every module this repository imported, and the import hook if it was the last."""
        discard(self.alias)


_FINDER = RepoFinder()


def _mount(alias: str, directory: Path, harness: HarnessCalls) -> None:
    """Serve a repository, installing the shared finder when it is the first. Hold ``_LOAD_LOCK``."""
    _FINDER.mount(alias, directory, harness)
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def discard(alias: str) -> None:
    """Remove a package's modules and, with the last one, the hook: no residue in the interpreter."""
    with _LOAD_LOCK:
        for name in [name for name in sys.modules if name == alias or name.startswith(alias + ".")]:
            del sys.modules[name]
        _FINDER.unmount(alias)
        if _FINDER.empty:
            try:
                sys.meta_path.remove(_FINDER)
            except ValueError:  # pragma: no cover - only reachable after external mutation
                pass


def _scan_modules(directory: Path) -> tuple[list[str], str]:
    """List the dotted names of every module below the package root, and digest their sources.

    The digest covers the root ``__init__.py`` as well, and the content rather than the
    modification time: two quick edits of the same length must not look like no change.
    """
    if not (directory / INIT_FILE).is_file():
        raise ValueError(f"The repository has no {INIT_FILE}; it must stay a Python package")
    names: list[str] = []
    digest = sha256()
    for current, directories, files in os.walk(directory):
        directories[:] = sorted(item for item in directories if not item.startswith(".") and item != "__pycache__")
        relative = Path(current).relative_to(directory)
        for file in sorted(files):
            if not file.endswith(".py") or file.startswith("."):
                continue
            path = Path(current) / file
            shown = path.relative_to(directory).as_posix()
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"{shown} exceeds {MAX_FILE_BYTES} bytes")
            digest.update(shown.encode() + b"\0" + path.read_bytes() + b"\0")
            parts = [*relative.parts, file.removesuffix(".py")]
            if parts[-1] == "__init__":
                parts.pop()
            if not parts:
                continue
            if not all(part.isidentifier() for part in parts):
                raise ValueError(f"{shown} is not an importable module name")
            names.append(".".join(parts))
    if len(names) > MAX_MODULES:
        raise ValueError(f"A repository holds at most {MAX_MODULES} modules")
    return names, digest.hexdigest()


def _read_resources(directory: Path) -> dict[str, TextResource]:
    folder = directory / RESOURCES_DIRECTORY
    if not folder.is_dir():
        return {}
    resources: dict[str, TextResource] = {}
    for path in sorted(folder.glob("*.md")):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{RESOURCES_DIRECTORY}/{path.name} exceeds {MAX_FILE_BYTES} bytes")
        resources[path.stem] = parse_resource(path.stem, path.read_text(encoding="utf-8"))
    if len(resources) > MAX_RESOURCES:
        raise ValueError(f"A repository lists at most {MAX_RESOURCES} resources; merge related notes")
    return resources


@dataclass(frozen=True)
class RepoScan:
    """What the files of a repository say right now, found without importing anything.

    ``policy`` is the declaration read in this scan. The loader compares it with the tools
    the import produces instead of reading the files again, so a module body that rewrites
    them changes the next load and not the one it is part of.
    """

    modules: list[str]
    fingerprint: str
    resources: dict[str, TextResource]
    policy: RepositoryPolicy


def scan_repo(directory: Path) -> RepoScan:
    """Check the files of a repository and describe them.

    Raises:
        ValueError: If a file is too large, misnamed, a resource has bad front matter, the
            manifest is missing, or either policy file is malformed. A missing ban list is
            allowed and bans nothing. Nothing is imported here, so a repository that cannot
            declare itself never runs a line of its code.
    """
    modules, fingerprint = _scan_modules(directory)
    policy = inspect_repository(directory)
    if not policy.ok:
        raise ValueError("; ".join(policy.problems))
    return RepoScan(modules, fingerprint, _read_resources(directory), policy)


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


def load_repo(directory: Path, harness: HarnessCalls, scan: RepoScan) -> LoadedRepo:
    """Import every module of a repository and collect its tools.

    Module bodies run here, with the application's privileges. On any failure
    the half-imported package is removed and a ``ValueError`` names the file and
    line when the error is in Python, or the declaration it rejects, so a caller can
    keep serving the previous version.

    Args:
        directory: The repository root.
        harness: Bound into every module's globals as ``harness``.
        scan: The result of :func:`scan_repo` for this directory.
    """
    alias = f"{ALIAS_PREFIX}{new_uuid7().replace('-', '_')}"
    with _LOAD_LOCK:
        _mount(alias, directory, harness)
        try:
            import_module(alias)
            for dotted in scan.modules:
                import_module(f"{alias}.{dotted}")
            tools = _collect_tools(alias)
            # Judged against the declarations this scan read, never against the disk again:
            # a module body that rewrites MANIFEST or BAN_TOOLS changes the next load.
            # ``scan_repo`` refuses a file without a manifest, so the policy always has one;
            # the assert states that for a caller that built a ``RepoScan`` by hand.
            assert scan.policy.manifest is not None
            problems = _tool_problems(scan.policy.manifest, scan.policy.banned, tools)
            if problems:
                raise ValueError("; ".join(problems))
        except BaseException as failure:
            discard(alias)
            if isinstance(failure, Exception):
                raise ValueError(describe_failure(failure, directory)) from failure
            raise
    return LoadedRepo(alias, tools, scan.fingerprint)
