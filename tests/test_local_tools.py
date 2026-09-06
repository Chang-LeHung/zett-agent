import asyncio
import importlib

import pytest
from pydantic import ValidationError

from kcs_agent import glob, grep, read_file, replace_in_file, run_shell, write_file


async def test_file_tools_write_read_and_replace_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    created = await write_file({"path": "notes/example.txt", "content": "one\ntwo\nthree\nfour\n"})
    assert created.path == "notes/example.txt"
    assert created.created is True

    page = await read_file({"path": "notes/example.txt", "start_line": 2, "line_count": 2})
    assert page.content == "two\nthree\n"
    assert (page.start_line, page.end_line, page.total_lines, page.has_more) == (2, 3, 4, True)

    replaced = await replace_in_file({"path": "notes/example.txt", "old_text": "three", "new_text": "THREE"})
    assert replaced.replacements == 1
    assert (await read_file({"path": "notes/example.txt"})).content == "one\ntwo\nTHREE\nfour\n"

    overwritten = await write_file({"path": "notes/example.txt", "content": "same\nsame\n"})
    assert overwritten.created is False
    replaced_all = await replace_in_file(
        {"path": "notes/example.txt", "old_text": "same", "new_text": "changed", "replace_all": True}
    )
    assert replaced_all.replacements == 2
    assert (tmp_path / "notes/example.txt").read_text() == "changed\nchanged\n"


async def test_file_tools_reject_unsafe_or_ambiguous_operations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool_module = importlib.import_module("kcs_agent.tools.coding")
    monkeypatch.setattr(tool_module, "MAX_FILE_BYTES", 16)
    await write_file({"path": "existing.txt", "content": "same same"})
    (tmp_path / "directory").mkdir()

    with pytest.raises(ValueError, match="escapes"):
        await read_file({"path": "../outside.txt"})
    with pytest.raises(ValueError, match="relative"):
        await read_file({"path": str(tmp_path / "existing.txt")})
    with pytest.raises(ValueError, match="does not exist"):
        await read_file({"path": "missing.txt"})
    with pytest.raises(ValueError, match="does not exist"):
        await replace_in_file({"path": "missing.txt", "old_text": "old", "new_text": "new"})
    with pytest.raises(ValueError, match="not a file"):
        await write_file({"path": "directory", "content": "text"})
    with pytest.raises(ValueError, match="already exists"):
        await write_file({"path": "existing.txt", "content": "text", "overwrite": False})
    with pytest.raises(ValueError, match="not unique"):
        await replace_in_file({"path": "existing.txt", "old_text": "same", "new_text": "changed"})
    with pytest.raises(ValueError, match="not found"):
        await replace_in_file({"path": "existing.txt", "old_text": "missing", "new_text": "changed"})
    with pytest.raises(ValueError, match="byte limit"):
        await write_file({"path": "large.txt", "content": "x" * 17})

    (tmp_path / "oversized.txt").write_text("x" * 17)
    assert (await read_file({"path": "oversized.txt"})).content == "x" * 17


async def test_local_tools_export_typed_schemas():
    tools = {
        registered.name: registered for registered in (glob, grep, read_file, write_file, replace_in_file, run_shell)
    }
    assert set(tools) == {"glob", "grep", "read_file", "write_file", "replace_in_file", "run_shell"}
    assert tools["read_file"].parameters["properties"]["start_line"]["minimum"] == 1
    assert tools["read_file"].parameters["properties"]["line_count"]["maximum"] == 2_000
    assert tools["read_file"].parameters["properties"]["path"]["description"].startswith("Relative path")
    assert tools["read_file"].snippet.startswith("read_file(")
    assert tools["read_file"].guidelines == (
        "Use line ranges for large files.",
        "Inspect the current content before editing a file.",
        "Continue with next_line and next_column when has_more is true.",
    )

    with pytest.raises(ValidationError):
        await tools["read_file"]({"path": "file.txt", "start_line": 0})


async def test_glob_finds_sorted_working_directory_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "src" / "z.py").write_text("")
    (tmp_path / "src" / "nested" / "a.py").write_text("")
    (tmp_path / "src" / "ignored.txt").write_text("")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("outside")
    (tmp_path / "src" / "outside.py").symlink_to(outside)

    result = await glob({"pattern": "src/**/*.py"})
    assert result.paths == ["src/nested/a.py", "src/z.py"]
    assert result.truncated is False

    limited = await glob({"pattern": "src/**/*.py", "max_results": 1})
    assert limited.paths == ["src/nested/a.py"]
    assert limited.truncated is True

    with pytest.raises(ValueError, match="escape"):
        await glob({"pattern": "../*.py"})
    with pytest.raises(ValueError, match="relative"):
        await glob({"pattern": str(tmp_path / "*.py")})
    outside.unlink()


async def test_grep_searches_text_files_and_reports_locations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "first.py").write_text("class Agent:\n    pass\n")
    (tmp_path / "src" / "second.py").write_text("agent = 'lowercase'\nclass Other:\n")
    (tmp_path / "src" / "binary.py").write_bytes(b"\xff\xfe")

    result = await grep({"pattern": "agent", "file_pattern": "src/**/*.py", "case_sensitive": False, "max_results": 10})
    assert [(match.path, match.line_number, match.column, match.text) for match in result.matches] == [
        ("src/first.py", 1, 7, "class Agent:"),
        ("src/second.py", 1, 1, "agent = 'lowercase'"),
    ]
    assert result.files_searched == 2
    assert result.truncated is False

    limited = await grep({"pattern": "class|agent", "file_pattern": "src/**/*.py", "max_results": 1})
    assert len(limited.matches) == 1
    assert limited.truncated is True

    with pytest.raises(ValueError, match="Invalid regular expression"):
        await grep({"pattern": "[", "file_pattern": "src/**/*.py"})
    with pytest.raises(ValueError, match="escape"):
        await grep({"pattern": "Agent", "file_pattern": "../**/*"})

    tool_module = importlib.import_module("kcs_agent.tools.coding")
    monkeypatch.setattr(tool_module, "MAX_FILE_BYTES", 4)
    oversized = await grep({"pattern": "Agent", "file_pattern": "src/first.py"})
    assert oversized.files_searched == 0
    assert oversized.matches == []


async def test_shell_tool_captures_status_output_and_truncation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_module = importlib.import_module("kcs_agent.tools.output")
    monkeypatch.setattr(output_module, "MAX_OUTPUT_BYTES", 100)
    result = await run_shell({"command": "printf START; printf '%0200d' 0; printf END; printf 'error' >&2; exit 3"})

    assert result.exit_code == 3
    assert result.stdout.startswith("START")
    assert result.stdout.endswith("END")
    assert len(result.stdout.encode()) <= 100
    assert result.stderr == "error"
    assert (tmp_path / result.stdout_path).read_text().endswith("END")
    assert result.output_truncated is True
    assert result.timed_out is False

    with pytest.raises(ValueError, match="blank"):
        await run_shell({"command": "   "})


async def test_shell_tool_terminates_after_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await run_shell({"command": "while :; do :; done", "timeout_seconds": 1})
    assert result.timed_out is True
    assert result.exit_code != 0


async def test_shell_tool_terminates_when_its_task_is_cancelled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = asyncio.create_task(run_shell({"command": "while :; do :; done"}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
