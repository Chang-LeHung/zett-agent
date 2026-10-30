"""What a repository declares about itself, and what it may not offer.

A repository is loadable only when `MANIFEST` sits beside its `__init__.py` and says
what it is: a name, a version, and a description. `BAN_TOOLS` names tools the
repository may not define, one per line.

Nothing here imports a repository, touches the harness, or knows how a tool is built.
Reading and checking are ordinary file work, so an application can call
:func:`inspect_repository` on a directory without a running session.

What the check can and cannot promise: the files are read before any repository code
runs, but the names a repository defines are only known after its modules execute. The
loader therefore decides on the declarations read *before* the import, and compares
them with what that import produced. A repository that rewrites its own files while
its modules run has changed the next load, not this one. The files are ordinary,
model-writable text: this is a recorded refusal plus a consistency check at load time,
not a sandbox and not a policy the application controls.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - the standard library has it from Python 3.11
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore[import-not-found, no-redef]  # 3.10-only dependency, absent here

#: Names the harness reserves for its own model-facing tools; a repository cannot use them.
MANAGEMENT_NAMES = frozenset({"repo_create", "repo_load", "repo_unload", "profile_tools"})

#: The file that declares a repository, beside its ``__init__.py``.
MANIFEST_FILE = "MANIFEST"
#: The file that names tools the repository may not define, one per line.
BAN_TOOLS_FILE = "BAN_TOOLS"
#: Keys `MANIFEST` may hold, in the order the documentation lists them.
MANIFEST_KEYS = ("name", "version", "description", "tools")
#: A repository's description is shown with it; long ones cost every listing.
MAX_DESCRIPTION_CHARS = 512
#: Both files are hand-written or model-written text; a large one is a mistake.
MAX_FILE_BYTES = 200_000

#: A tool name, as the loader accepts it: this is the one rule, and `repo` uses it too.
TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
#: A repository directory name; it starts with a letter or digit so it is a usable module name.
REPOSITORY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_VERSION = re.compile(r"\d+(?:\.\d+){0,2}\Z")

_BOM = "\ufeff"


def is_tool_name(name: object) -> bool:
    """Whether ``name`` is a tool name the harness accepts."""
    return isinstance(name, str) and TOOL_NAME.fullmatch(name) is not None


def is_repository_name(name: object) -> bool:
    """Whether ``name`` can be a repository directory name."""
    return isinstance(name, str) and REPOSITORY_NAME.fullmatch(name) is not None


@dataclass(frozen=True)
class Manifest:
    """What `MANIFEST` says a repository is."""

    name: str
    version: str
    description: str
    #: Tools the repository promises to define, exactly. Empty means `MANIFEST` has no
    #: ``tools`` key; an empty list is an error, not a promise of nothing.
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositoryPolicy:
    """The raw declarations of one directory, and everything wrong with them.

    This is a report, not an enforced policy: the caller decides what to do with it.
    """

    manifest: Manifest | None
    banned: tuple[str, ...]
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether the declarations allow this repository to load."""
        return not self.problems


def read_manifest(directory: Path) -> Manifest:
    """Read and check `MANIFEST` in ``directory``.

    The directory is resolved first, so ``"."`` names the directory it stands for
    rather than the empty string.

    Raises:
        ValueError: The file is missing, unreadable, too large, not TOML, or does not
            name a repository the way this module requires.
    """
    directory = Path(directory).absolute()
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise ValueError(
            f"The repository has no {MANIFEST_FILE}; write name = {directory.name!r}, version and description there"
        )
    text = _read_text(path, MANIFEST_FILE)
    try:
        parsed = _toml.loads(text)
    except _toml.TOMLDecodeError as error:
        raise ValueError(f"{MANIFEST_FILE} is not valid TOML: {error}") from error
    unknown = [key for key in parsed if key not in MANIFEST_KEYS]
    if unknown:
        raise ValueError(f"{MANIFEST_FILE} has an unknown key {unknown[0]!r}; it takes {', '.join(MANIFEST_KEYS)}")
    problems = _manifest_problems(parsed, directory.name)
    if problems:
        raise ValueError("; ".join(problems))
    name = parsed["name"]
    version = parsed["version"]
    description = parsed["description"]
    declared = parsed.get("tools", [])
    return Manifest(
        name=name.strip(),
        version=version.strip(),
        description=description.strip(),
        tools=tuple(item.strip() for item in declared),
    )


def read_banned_tools(directory: Path) -> tuple[str, ...]:
    """Read `BAN_TOOLS`: one tool name per line, blank lines ignored.

    A missing file bans nothing. The file is ordinary text, so the same write the
    model uses on any other file edits it.
    """
    path = directory / BAN_TOOLS_FILE
    if not path.is_file():
        return ()
    banned: list[str] = []
    for line in _read_text(path, BAN_TOOLS_FILE).splitlines():
        name = line.strip()
        if not name:
            continue
        if not is_tool_name(name):
            raise ValueError(f"{BAN_TOOLS_FILE} names {name!r}, which is not a tool name")
        if name in banned:
            raise ValueError(f"{BAN_TOOLS_FILE} names {name!r} more than once")
        banned.append(name)
    return tuple(banned)


def inspect_repository(directory: str | Path, *, tools: Iterable[str] | None = None) -> RepositoryPolicy:
    """Read the repository's raw declarations and report every problem with them.

    Nothing is imported: this is file work, so it answers for a directory no session
    has opened.

    Args:
        directory: Repository root, the directory holding ``__init__.py``.
        tools: Names the repository defines. ``None`` means they are not known yet, so
            only the rules that depend on the files are checked; pass the names — an
            empty iterable included — to check them against the bans and to compare them
            with the declaration.

    Returns:
        A :class:`RepositoryPolicy` holding the parsed manifest, the ban list, and one
        message per problem. :attr:`RepositoryPolicy.ok` is false when there are any.
    """
    root = Path(directory)
    problems: list[str] = []
    manifest: Manifest | None = None
    banned: tuple[str, ...] = ()
    try:
        banned = read_banned_tools(root)
    except ValueError as error:
        problems.append(str(error))
    try:
        manifest = read_manifest(root)
    except ValueError as error:
        problems.append(str(error))
    if manifest is not None:
        problems.extend(_tool_problems(manifest, banned, tools))
    return RepositoryPolicy(manifest, banned, tuple(problems))


def _tool_problems(manifest: Manifest, banned: Iterable[str], tools: Iterable[str] | None = None) -> tuple[str, ...]:
    """Check the tools a repository defines against its declaration and the bans.

    The caller supplies the declarations read *before* the import and the names the
    import produced, so a module body cannot change the verdict by rewriting the files
    while it runs. ``tools=None`` checks only what the two files say between themselves,
    which already catches a declaration that names a banned tool.
    """
    forbidden = set(banned)
    declared = set(manifest.tools)
    defined = None if tools is None else set(tools)
    problems: list[str] = []
    for name in sorted(declared & forbidden):
        problems.append(f"{MANIFEST_FILE} declares {name!r}, which {BAN_TOOLS_FILE} bans")
    if defined is not None:
        for name in sorted((defined & forbidden) - declared):
            problems.append(f"The repository defines {name!r}, which {BAN_TOOLS_FILE} bans")
        if declared:
            # A banned name is already reported; calling it missing as well would point
            # the reader at the wrong problem.
            missing = sorted(declared - defined - forbidden)
            if missing:
                problems.append(f"{MANIFEST_FILE} declares {missing[0]!r}, which the repository does not define")
            undeclared = sorted(defined - declared - forbidden)
            if undeclared:
                problems.append(f"The repository defines {undeclared[0]!r}, which {MANIFEST_FILE} does not declare")
    return tuple(problems)


def _read_text(path: Path, label: str) -> str:
    """Read one policy file as UTF-8 text, without a byte-order mark."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{label} exceeds {MAX_FILE_BYTES} bytes")
        return path.read_text(encoding="utf-8").lstrip(_BOM)
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error
    except OSError as error:
        raise ValueError(f"{label} cannot be read: {error}") from error


def _manifest_problems(parsed: dict[str, object], directory_name: str) -> list[str]:
    """Check the fields of one parsed manifest, naming every problem found."""
    problems: list[str] = []
    for key in ("name", "version", "description"):
        if key not in parsed:
            problems.append(f"{MANIFEST_FILE} needs {key}")
    name = parsed.get("name")
    if name is not None and not isinstance(name, str):
        problems.append(f"{MANIFEST_FILE} name must be text")
        name = None
    elif isinstance(name, str) and not is_repository_name(name.strip()):
        problems.append(f"{MANIFEST_FILE} name {name.strip()!r} is not a repository name")
    elif isinstance(name, str) and name.strip() != directory_name:
        problems.append(f"{MANIFEST_FILE} name {name.strip()!r} must match the directory {directory_name!r}")
    version = parsed.get("version")
    if version is not None and not isinstance(version, str):
        problems.append(f"{MANIFEST_FILE} version must be text")
    elif isinstance(version, str) and not _VERSION.fullmatch(version.strip()):
        problems.append(f"{MANIFEST_FILE} version {version.strip()!r} must look like 1 or 1.2.3")
    description = parsed.get("description")
    if description is not None and not isinstance(description, str):
        problems.append(f"{MANIFEST_FILE} description must be text")
    elif isinstance(description, str) and not 1 <= len(description.strip()) <= MAX_DESCRIPTION_CHARS:
        problems.append(f"{MANIFEST_FILE} description must be 1–{MAX_DESCRIPTION_CHARS} characters")
    declared = parsed.get("tools")
    if declared is not None:
        if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
            problems.append(f"{MANIFEST_FILE} tools must be a list of tool names")
        elif not declared:
            problems.append(f"{MANIFEST_FILE} tools is empty; list at least one tool name, or remove the key")
        else:
            names = [item.strip() for item in declared]
            bad = [item for item in names if not is_tool_name(item)]
            if bad:
                problems.append(f"{MANIFEST_FILE} tools names {bad[0]!r}, which is not a tool name")
            reserved = [item for item in names if item in MANAGEMENT_NAMES]
            if reserved:
                problems.append(f"{MANIFEST_FILE} tools names {reserved[0]!r}, which is reserved for JIT management")
            duplicates = sorted({item for item in names if names.count(item) > 1})
            if duplicates:
                problems.append(f"{MANIFEST_FILE} tools names {duplicates[0]!r} more than once")
    return problems


__all__ = [
    "BAN_TOOLS_FILE",
    "MANIFEST_FILE",
    "Manifest",
    "RepositoryPolicy",
    "inspect_repository",
    "is_repository_name",
    "is_tool_name",
    "read_banned_tools",
    "read_manifest",
]
