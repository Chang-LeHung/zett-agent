"""Walk the tree, run every rule over every module, and report what they found."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from .rules import RULES, Violation, check_source

#: Where source lives, relative to the repository root. Tests are read too: they must follow the alias rule.
SOURCE_DIRECTORIES = ("src", "tests", "examples", "docs/_examples", "static_check")

#: Directory names a source file is never in: a virtual environment, a cache, or build output.
BUILD_OUTPUT = frozenset({".venv", "__pycache__", "_build", "node_modules", "dist", "build"})


def repository_root(start: Path | None = None) -> Path:
    """The repository root, found from ``start`` (or the working directory) by looking for ``pyproject.toml``."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml above {here}")


def python_files(root: Path) -> list[Path]:
    """Every module under the source directories, sorted, ignoring build output and the virtual environment."""
    found: list[Path] = []
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            found.extend(path for path in base.rglob("*.py") if not _generated(path, root))
    return sorted(found)


def _generated(path: Path, root: Path) -> bool:
    """Whether ``path`` sits inside build output, a virtual environment or a cache, rather than in source.

    A directory named ``build`` holds the package's own build output; ``tests`` is a source directory and
    must not be mistaken for it, so the names are compared as whole path components.
    """
    return any(part in BUILD_OUTPUT for part in path.relative_to(root).parts[:-1])


def scan(root: Path) -> list[Violation]:
    """Every violation in the tree, in file and line order."""
    violations: list[Violation] = []
    for path in python_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
        except (OSError, SyntaxError) as error:  # a file that cannot be read or parsed is not this tool's report
            raise ValueError(f"cannot check {relative}: {error}") from error
        violations.extend(check_source(relative, source))
    return violations


def rule_names() -> Iterator[str]:
    """The rules this tool enforces, in id order."""
    return iter(sorted(RULES))
