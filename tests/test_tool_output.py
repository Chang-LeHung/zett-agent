"""Bounded previews preserve diagnostics and provide lossless read cursors."""

import asyncio
import importlib
import shlex
import sys

import pytest

from kcs_agent import glob, grep, read_file, run_shell
from kcs_agent.tools import output as tool_output


@pytest.mark.parametrize("text", ["", "abc", "\u4e2d\u6587" * 4000, "row\n" * 4000])
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
    module = importlib.import_module("kcs_agent.tools.coding")
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 101)
    text = "\u4e2d\u6587" * 120 + "\nsecond\nthird\n"
    (tmp_path / "long.txt").write_text(text, encoding="utf-8")
    parts = []
    line, column = 1, 1
    for _ in range(30):
        result = await read_file({"path": "long.txt", "start_line": line, "start_column": column, "line_count": 1})
        assert len(result.content.encode("utf-8")) <= 101
        assert result.total_lines == 3
        parts.append(result.content)
        if not result.has_more:
            break
        assert (result.next_line, result.next_column) > (line, column)
        line, column = result.next_line, result.next_column
    else:
        pytest.fail("Read cursor failed to reach EOF")
    assert "".join(parts) == text


async def test_read_large_file_and_out_of_range_cursor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "x" * (3 * 1024 * 1024) + "\nend\n"
    (tmp_path / "large.txt").write_text(text)
    page = await read_file({"path": "large.txt"})
    assert page.truncated
    assert page.next_line == 1
    end = await read_file({"path": "large.txt", "start_line": 2})
    assert end.content == "end\n"
    assert not end.has_more
    empty = await read_file({"path": "large.txt", "start_line": 50})
    assert empty.content == ""
    assert not empty.has_more


async def test_search_caps_output_and_preserves_distant_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("kcs_agent.tools.coding")
    (tmp_path / "long.txt").write_text("x" * 10000 + "TARGET" + "y" * 10000)
    result = await grep({"pattern": "TARGET"})
    match = result.matches[0]
    assert "TARGET" in match.text
    assert match.text_truncated
    assert match.text_start_column <= match.column
    assert len(match.text.encode()) <= tool_output.MAX_MATCH_BYTES
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 4)
    assert (await grep({"pattern": "TARGET"})).truncated
    assert (await glob({"pattern": "*.txt"})).truncated


async def test_shell_full_output_is_readable_after_preview_truncation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = "import sys; print('START'); print('x'*3000000); print('END'); print('ERROR', file=sys.stderr)"
    result = await run_shell({"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"})
    assert result.exit_code == 0
    assert result.output_truncated
    assert result.stdout.startswith("START")
    assert result.stdout.endswith("END\n")
    assert result.stderr == "ERROR\n"
    assert (tmp_path / result.stdout_path).stat().st_size > 3000000
    page = await read_file({"path": result.stdout_path, "start_line": 3})
    assert page.content == "END\n"
    assert (await read_file({"path": result.stderr_path})).content == "ERROR\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group behavior")
async def test_timeout_stops_child_processes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = "import time; from pathlib import Path; time.sleep(2); Path('survived').touch()"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)} & wait"
    result = await asyncio.wait_for(run_shell({"command": command, "timeout_seconds": 1}), timeout=5)
    assert result.timed_out
    await asyncio.sleep(1.2)
    assert not (tmp_path / "survived").exists()
