"""Helpers for writing a JIT repository's files in tests, the way a model would."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from zett_agent.jit_harness import JITHarness
from zett_agent.jit_harness.repo import MANIFEST_SKELETON, SKELETON, session_directory_name

#: Test sources use ``@tool`` and ``ToolExecutionMode`` without importing them.
HEADER = "from zett_agent.tools.base import ToolExecutionMode, tool\n"


def seed_repo(root: Path, session: str, modules: Mapping[str, str], name: str = "repo") -> Path:
    """Leave a repository on disk, as an earlier process would have.

    ``JITHarnessExtension(root)`` adopts and loads it on the session's first
    request. ``modules`` maps a module name to its source.
    """
    directory = Path(root) / session_directory_name(session) / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(SKELETON, encoding="utf-8")
    (directory / "MANIFEST").write_text(MANIFEST_SKELETON.format(name=name), encoding="utf-8")
    for module, source in modules.items():
        (directory / f"{module}.py").write_text(HEADER + source, encoding="utf-8")
    return directory


def repo_dir(harness: JITHarness, name: str = "repo") -> Path:
    """Return the harness's repository directory, creating the repository once."""
    if harness.repo is None:
        harness.create_repo(name)
    path = harness.repo_path
    assert path is not None
    return path


def write_module(harness: JITHarness, module: str, source: str, *, header: bool = True) -> Path:
    """Write ``<module>.py`` into the repository without loading it."""
    path = repo_dir(harness) / f"{module}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((HEADER if header else "") + source, encoding="utf-8")
    return path


def put_tool(harness: JITHarness, source: str, module: str = "tool", *, reload: bool = True) -> None:
    """Write one module and, by default, reload the repository."""
    write_module(harness, module, source)
    if reload:
        harness.load_repo()


def put_resource(
    harness: JITHarness, name: str, description: str, content: str, *, kind: str | None = None, reload: bool = True
) -> None:
    """Write ``resources/<name>.md`` and, by default, reload the repository.

    Without ``kind`` the front matter omits it, which means a workflow.
    """
    folder = repo_dir(harness) / "resources"
    folder.mkdir(exist_ok=True)
    header = f"kind: {kind}\n" if kind else ""
    (folder / f"{name}.md").write_text(f"---\n{header}description: {description}\n---\n{content}", encoding="utf-8")
    if reload:
        harness.load_repo()
