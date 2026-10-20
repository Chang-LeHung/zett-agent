"""JIT instructions load as package data, independent of checkout paths."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest

from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.jit_harness.extension import _INSTRUCTIONS
from zett_agent.jit_harness.repo import MANAGEMENT_NAMES

ROOT = Path(__file__).resolve().parents[1] / "src"
RESOURCES = ("zett_agent/jit_harness/instructions.md",)
#: Tool names the repository API removed or made Python-only; the prompt may not advertise them.
REMOVED_TOOLS = (
    "create_tool",
    "update_tool",
    "unload_tool",
    "load_tool",
    "list_tools",
    "inspect_tool",
    "read_tool_source",
    "read_tool_schema",
    "inspect_tool_stats",
    "inspect_tool_calls",
    "inspect_tool_errors",
)
#: Identifiers the core prompt must teach, so the model can use what it is offered.
CORE_CODES = (
    "repo_create(repo=name)",
    "repo_load",
    "repo_unload",
    "profile_tools",
    "@tool",
    "Guidelines:",
    "__init__.py",
    "harness",
    "def",
    "async def",
    "await harness.call_tool(name, **arguments)",
    "harness.call_tool_sync(name, **arguments)",
    "resources/<name>.md",
    "description",
    "kind",
    "SKILL",
)


def test_jit_instructions_are_loaded_from_the_markdown_package_resources():
    root = files("zett_agent.jit_harness")
    assert _INSTRUCTIONS == root.joinpath("instructions.md").read_text(encoding="utf-8")
    assert _INSTRUCTIONS.startswith("# JIT tools and reusable notes\n")
    assert "fewest total tokens and shortest end-to-end" in _INSTRUCTIONS
    assert "trusted, not sandboxed" in _INSTRUCTIONS
    assert "substantial time, tokens" in _INSTRUCTIONS
    assert "`SKILL`" in _INSTRUCTIONS
    assert _INSTRUCTIONS.endswith("\n")


def test_guidance_names_exactly_the_tools_the_session_offers(tmp_path):
    """The prompt never advertises a tool the model cannot call."""
    harness = JITHarness(tmp_path)
    extension = JITHarnessExtension(tmp_path)
    offered = {registered.name for registered in extension._management_tools(harness)}
    assert offered == MANAGEMENT_NAMES
    prompt = extension._guidance(harness, "")
    for name in offered:
        assert f"`{name}`" in prompt, name
    for code in CORE_CODES:
        assert f"`{code}`" in prompt
    for name in REMOVED_TOOLS:
        assert f"`{name}`" not in prompt, name


def test_guidance_keeps_the_stable_instruction_prefix_first(tmp_path):
    """Repository line, then tool guidance; the cacheable instructions come first."""
    harness = JITHarness(tmp_path)
    extension = JITHarnessExtension(tmp_path)
    assert extension._guidance(harness, "") == _INSTRUCTIONS
    created = harness.create_repo("repo")
    guidance = "Prefer `repo_load` after every edit."
    prompt = extension._guidance(harness, guidance)
    assert prompt.startswith(_INSTRUCTIONS)
    assert f"Your repository is `repo` at `{created['path']}`." in prompt
    assert prompt.endswith("\n" + guidance)


@pytest.mark.parametrize("include_resources", [True, False])
def test_jit_instructions_load_from_zipped_package_without_a_source_checkout(tmp_path, include_resources):
    """No __file__ assumptions or silent fallback to a stale embedded prompt."""
    archive_path = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted((ROOT / "zett_agent").rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".md"):
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative not in RESOURCES or include_resources:
                archive.write(path, relative)
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from importlib.resources import files\n"
        "import zett_agent.jit_harness.extension as extension\n"
        "assert extension.__file__.startswith(sys.argv[1])\n"
        "root = files('zett_agent.jit_harness')\n"
        "core = root.joinpath('instructions.md').read_text(encoding='utf-8')\n"
        "assert extension._INSTRUCTIONS == core\n"
        "assert core.startswith('# JIT tools and reusable notes\\n')\n"
        "print('Packaged instructions loaded')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(archive_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if include_resources:
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "Packaged instructions loaded"
    else:
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr and "instructions.md" in result.stderr
