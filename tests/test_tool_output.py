"""Bounded previews preserve diagnostics and provide lossless read cursors."""

import asyncio
import importlib
import re
import shlex
import sys

import pytest

from zett_agent import glob, grep, read_file, run_shell
from zett_agent.tools import output as tool_output

RESUME = re.compile(r"start_line=(\d+)(?:, start_column=(\d+))?\]")


@pytest.mark.parametrize("text", ["", "abc", "中文" * 4000, "row\n" * 4000])
def test_shell_preview_respects_both_budgets(tmp_path, text):
    path = tmp_path / "output"
    path.write_text(text, encoding="utf-8")
    preview, truncated = tool_output.shell_preview(path)
    assert len(preview.encode("utf-8")) <= tool_output.MAX_OUTPUT_BYTES
    assert len(preview.splitlines()) <= tool_output.MAX_OUTPUT_LINES
    if truncated:
        assert "middle omitted" in preview
    else:
        assert preview == text
    assert path.read_text() == text


async def test_long_unicode_line_can_be_read_without_losing_characters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("zett_agent.tools.coding")
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 101)
    text = "中文" * 120 + "\nsecond\nthird\n"
    (tmp_path / "long.txt").write_text(text, encoding="utf-8")
    parts = []
    line, column = 1, 1
    for _ in range(30):
        result = await read_file({"path": "long.txt", "start_line": line, "start_column": column, "line_count": 1})
        content, _, hint = result.partition("\n... [")
        assert len(content.encode("utf-8")) <= 101
        parts.append(content)
        if not hint:
            break
        match = RESUME.search(hint)
        assert match is not None
        next_line, next_column = int(match.group(1)), int(match.group(2) or 1)
        assert (next_line, next_column) > (line, column)
        line, column = next_line, next_column
    else:
        pytest.fail("Read cursor failed to reach EOF")
    assert "".join(parts) == text


async def test_read_large_file_and_out_of_range_cursor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "x" * (3 * 1024 * 1024) + "\nend\n"
    (tmp_path / "large.txt").write_text(text)
    page = await read_file({"path": "large.txt"})
    assert page.startswith("x" * 4096)
    content, _, hint = page.partition("\n... [")
    assert len(content.encode("utf-8")) <= tool_output.MAX_OUTPUT_BYTES
    match = RESUME.search(hint)
    assert match is not None
    assert int(match.group(1)) == 1
    assert int(match.group(2)) > 1
    end = await read_file({"path": "large.txt", "start_line": 2})
    assert end == "end\n"
    empty = await read_file({"path": "large.txt", "start_line": 50})
    assert empty == ""


async def test_search_caps_output_and_preserves_distant_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("zett_agent.tools.coding")
    (tmp_path / "long.txt").write_text("x" * 10000 + "TARGET" + "y" * 10000)
    result = await grep({"pattern": "TARGET"})
    assert result.startswith("long.txt:1: ")
    assert "TARGET" in result
    assert len(result.encode()) <= tool_output.MAX_MATCH_BYTES + len("long.txt:1: ")
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 4)
    assert "truncated" in (await grep({"pattern": "TARGET"}))
    assert "truncated" in (await glob({"pattern": "*.txt"}))


async def test_shell_full_output_is_readable_after_preview_truncation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = "import sys; print('START'); print('x'*3000000); print('END'); print('ERROR', file=sys.stderr)"
    result = await run_shell({"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"})
    assert result.startswith("START")
    assert "END\n" in result
    assert "[stderr]\nERROR" in result
    stdout_files = list((tmp_path / ".zett-tool-output").rglob("stdout.txt"))
    assert len(stdout_files) == 1
    assert stdout_files[0].stat().st_size > 3000000
    stdout_path = stdout_files[0].relative_to(tmp_path).as_posix()
    assert stdout_path in result
    page = await read_file({"path": stdout_path, "start_line": 3})
    assert page == "END\n"
    stderr_path = stdout_files[0].with_name("stderr.txt").relative_to(tmp_path).as_posix()
    assert await read_file({"path": stderr_path}) == "ERROR\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group behavior")
async def test_timeout_stops_child_processes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = "import time; from pathlib import Path; time.sleep(2); Path('survived').touch()"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)} & wait"
    with pytest.raises(RuntimeError, match="timed out"):
        await asyncio.wait_for(run_shell({"command": command, "timeout_seconds": 1}), timeout=5)
    await asyncio.sleep(1.2)
    assert not (tmp_path / "survived").exists()
