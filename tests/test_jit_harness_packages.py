"""Multi-module JIT tool packages: relative imports, per-file edits, portability."""

from __future__ import annotations

import asyncio
import json
import sys
from textwrap import dedent

import pytest
from pydantic import ValidationError

from zett_agent.jit_harness import JITHarness, JITHarnessSnapshot, PythonToolSource
from zett_agent.jit_harness.queries import ToolQueries
from zett_agent.jit_harness.snapshot import _validate_file_path, _validate_files
from zett_agent.tools.base import tool

PACKAGE_SOURCE = dedent('''\
    from .helpers import doubling

    @tool(guidelines="Use to compose a package-local helper.")
    def twice(value: int) -> int:
        """Double through a helper module."""
        return doubling(value)
''')
FILES = {
    "helpers.py": "def doubling(value):\n    return value * 2\n",
    "nested/__init__.py": "def label():\n    return 'nested'\n",
}
SINGLE_SOURCE = dedent('''\
    @tool(guidelines="Use to double one integer.")
    def twice(value: int) -> int:
        """Double one integer."""
        return value * 2
''')
COMPOSED_SOURCE = dedent('''\
    from .batch import total

    @tool(guidelines="Use to sum a range through composed base calls.")
    async def sum_range(count: int) -> int:
        """Add every integer from one to count in code, not in the model."""
        return await total(harness, count)
''')
COMPOSED_FILES = {
    "batch.py": (
        "async def total(harness, count):\n"
        "    running = 0\n"
        "    for value in range(1, count + 1):\n"
        "        running = await harness.call_tool('add', left=running, right=value)\n"
        "    return running\n"
    )
}


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def test_package_paths_must_stay_importable_and_unambiguous():
    assert _validate_file_path("helpers.py") == "helpers.py"
    assert _validate_file_path("deep/nested/helpers.py") == "deep/nested/helpers.py"
    for invalid in (
        "__init__.py",
        "/abs.py",
        "../escape.py",
        "helpers.txt",
        "helpers",
        "my-helpers.py",
        "sub//helpers.py",
        "sub/../helpers.py",
        "sub/",
        "",
        " ",
        f"sub/{'x' * 96}.py",
    ):
        with pytest.raises(ValueError):
            _validate_file_path(invalid)
    assert _validate_files(FILES) == FILES
    with pytest.raises(ValueError, match="same module"):
        _validate_files({"helpers.py": "value = 1", "helpers/__init__.py": "value = 2"})
    with pytest.raises(ValueError, match="blank"):
        _validate_files({"helpers.py": "   \n"})
    with pytest.raises(ValueError, match="at most"):
        _validate_files({f"helper_{index}.py": "value = 1" for index in range(33)})
    with pytest.raises(ValueError, match="exceeds"):
        _validate_files({"helpers.py": "#" * 200_001})


def test_package_files_round_trip_through_the_snapshot_at_version_two():
    harness = JITHarness()
    harness.create_tool(PACKAGE_SOURCE, entrypoint="twice", files=FILES)
    snapshot = harness.snapshot()
    assert snapshot.format_version == 2
    assert snapshot.tools[0].files == FILES
    restored = JITHarness(JITHarnessSnapshot.from_json(snapshot.to_json()))
    assert restored.source("twice", "helpers.py") == FILES["helpers.py"]
    assert restored.snapshot().tools[0].files == FILES
    # A single-module tool keeps the older, byte-stable snapshot version.
    plain = JITHarness()
    plain.create_tool(SINGLE_SOURCE, entrypoint="twice")
    assert plain.snapshot().format_version == 1
    assert "files" not in json.loads(plain.snapshot().to_json())["tools"][0]
    with pytest.raises(ValidationError, match="format_version 2"):
        JITHarnessSnapshot(
            format_version=1, tools=(plain.snapshot().tools[0].model_copy(update={"files": dict(FILES)}),)
        )


def test_relative_imports_resolve_helpers_and_subpackages_without_leaving_modules_behind():
    harness = JITHarness()
    harness.create_tool(PACKAGE_SOURCE, entrypoint="twice", files=FILES)
    assert harness.inspect_tool("twice")["files"] == {
        "__init__.py": len(PACKAGE_SOURCE),
        "helpers.py": len(FILES["helpers.py"]),
        "nested/__init__.py": len(FILES["nested/__init__.py"]),
    }
    assert asyncio.run(harness.call_tool("twice", value=21)) == 42
    queries = ToolQueries(harness)
    page = queries.read_source("twice", "helpers.py")
    assert page["path"] == "helpers.py" and page["content"] == FILES["helpers.py"]
    assert page["next_offset"] is None and len(page["content_id"]) == 64
    assert queries.read_source("twice")["path"] == "__init__.py"
    assert harness.source("twice", "__init__.py") == PACKAGE_SOURCE
    with pytest.raises(ValueError, match="Unknown package file"):
        harness.source("twice", "missing.py")
    with pytest.raises(ValueError, match="Python identifiers"):
        harness.source("twice", "my-helpers.py")
    # Loading is temporary: no session modules or import hooks survive it.
    assert not [name for name in sys.modules if name.startswith("zett_jit_")]


def test_a_broken_helper_file_leaves_the_previous_version_installed():
    harness = JITHarness()
    harness.create_tool(PACKAGE_SOURCE, entrypoint="twice", files=FILES)
    before = harness.snapshot()
    with pytest.raises(SyntaxError):
        harness.update_tool("twice", files={"helpers.py": "def doubling(value):\n"})
    with pytest.raises(ImportError):
        harness.update_tool("twice", files={"helpers.py": "value = 1\n"})
    assert harness.snapshot() == before
    assert harness.source("twice", "helpers.py") == FILES["helpers.py"]
    assert asyncio.run(harness.call_tool("twice", value=21)) == 42


def test_per_file_updates_merge_add_and_remove_without_resending_unchanged_files():
    harness = JITHarness()
    harness.create_tool(PACKAGE_SOURCE, entrypoint="twice", files=FILES)
    changed = "def doubling(value):\n    return value * 3\n"
    assert harness.update_tool("twice", files={"helpers.py": changed}) == {"name": "twice", "status": "loaded"}
    assert harness.source("twice", "nested/__init__.py") == FILES["nested/__init__.py"]
    assert harness.source("twice") == PACKAGE_SOURCE
    assert asyncio.run(harness.call_tool("twice", value=21)) == 63
    assert harness.update_tool("twice", files={"more.py": "value = 1\n"})["status"] == "loaded"
    assert sorted(harness.inspect_tool("twice")["files"]) == [
        "__init__.py",
        "helpers.py",
        "more.py",
        "nested/__init__.py",
    ]
    assert harness.update_tool("twice", remove=["more.py"])["status"] == "loaded"
    assert sorted(harness.inspect_tool("twice")["files"]) == ["__init__.py", "helpers.py", "nested/__init__.py"]
    with pytest.raises(ValueError, match="Unknown package file"):
        harness.update_tool("twice", remove=["more.py"])
    with pytest.raises(ValueError, match="source, files, or remove"):
        harness.update_tool("twice")
    with pytest.raises(ValueError, match="must be decorated"):
        harness.update_tool("twice", source="value = 1\n", entrypoint="twice")
    assert harness.snapshot().tools[0].files == {
        "helpers.py": changed,
        "nested/__init__.py": FILES["nested/__init__.py"],
    }


def test_text_tools_reject_package_operations():
    harness = JITHarness()
    with pytest.raises(ValueError, match="single document"):
        harness.create_tool("# steps", kind="text", name="notes", description="Use for notes.", files=FILES)
    harness.create_tool("# steps", kind="text", name="notes", description="Use for notes.")
    with pytest.raises(ValueError, match="single document"):
        harness.update_tool("notes", files={"helpers.py": "value = 1\n"})
    with pytest.raises(ValueError, match="single document"):
        harness.update_tool("notes", "# better steps", remove=["helpers.py"])
    assert harness.update_tool("notes", "# better steps")["status"] == "loaded"
    assert harness.source("notes") == "# better steps"
    with pytest.raises(ValueError, match="single document"):
        harness.source("notes", "helpers.py")


def test_unloaded_packages_keep_their_files_and_reload_without_recompiling():
    harness = JITHarness()
    harness.create_tool(PACKAGE_SOURCE, entrypoint="twice", files=FILES)
    harness.unload_tool("twice")
    assert harness.inspect_tool("twice")["status"] == "unloaded"
    assert harness.source("twice", "helpers.py") == FILES["helpers.py"]
    assert harness.snapshot().tools[0].files == FILES
    assert harness.load_tool("twice") == {"name": "twice", "status": "loaded"}
    assert asyncio.run(harness.call_tool("twice", value=21)) == 42


def test_a_package_helper_composes_base_tools_inside_one_model_call():
    """Programmatic tool calling: one outer call, many nested calls, one result."""
    harness = JITHarness(tools=[add])
    harness.create_tool(COMPOSED_SOURCE, entrypoint="sum_range", files=COMPOSED_FILES)
    assert asyncio.run(harness.call_tool("sum_range", count=100)) == 5050
    # Every composed call is measured, and the model sees only the outer result.
    assert harness.profile_stats("add").call_count == 100
    assert harness.profile_stats("sum_range").call_count == 1
    record = PythonToolSource.model_validate(harness.snapshot().tools[0].model_dump())
    assert record.files == COMPOSED_FILES and record.entrypoint == "sum_range"
