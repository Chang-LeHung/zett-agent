"""The preview a bounded tool result carries: what comes back, when it is cut, and that the file is untouched."""

from __future__ import annotations

from pathlib import Path

import pytest

from zett_agent.tools.output import (
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_LINES,
    OMISSION,
    Preview,
    stream_preview,
    utf8_prefix,
)


def test_a_short_stream_is_returned_whole(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("hello\nworld\n", newline="")

    assert stream_preview(path) == Preview("hello\nworld\n", False)


def test_a_stream_of_only_blank_lines_is_returned_whole(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("\n\n\n", newline="")

    assert stream_preview(path) == Preview("\n\n\n", False)


def test_a_tool_written_stream_is_returned_as_it_is_on_every_platform(tmp_path: Path):
    """A command on Windows ends its lines with `\\r\\n`; the preview hands back what the command wrote."""
    path = tmp_path / "out.txt"
    path.write_bytes(b"hello\r\nworld\r\n")

    assert stream_preview(path) == Preview("hello\r\nworld\r\n", False)


def test_a_long_stream_keeps_its_start_and_its_end_and_marks_the_middle(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("".join(f"line {n}\n" for n in range(1, 20_001)), newline="")

    preview = stream_preview(path)

    assert preview.truncated and OMISSION in preview.text
    assert preview.text.startswith("line 1\n") and preview.text.endswith("line 20000\n")
    assert len(preview.text.encode()) <= MAX_OUTPUT_BYTES
    assert len(preview.text.splitlines()) <= MAX_OUTPUT_LINES
    assert path.read_bytes().count(b"\n") == 20_000, "the file itself is never changed"


def test_a_stream_of_few_but_huge_lines_is_cut_by_bytes(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("x" * (MAX_OUTPUT_BYTES * 3))

    preview = stream_preview(path)

    assert preview.truncated and len(preview.text.encode()) <= MAX_OUTPUT_BYTES


def test_a_character_cut_in_half_at_a_window_edge_is_dropped_not_garbled(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("\u00e9" * (MAX_OUTPUT_BYTES * 2), encoding="utf-8")

    preview = stream_preview(path)

    assert "\ufffd" not in preview.text and preview.truncated


def test_a_caller_marker_stands_where_the_cut_is_and_is_counted_against_the_budgets(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("".join(f"line {n}\n" for n in range(1, 20_001)), newline="")
    marker = "\n... [kept elsewhere] ...\n"

    preview = stream_preview(path, marker)

    assert preview.text.count(marker) == 1 and OMISSION not in preview.text
    assert len(preview.text.encode()) <= MAX_OUTPUT_BYTES
    assert len(preview.text.splitlines()) <= MAX_OUTPUT_LINES


@pytest.mark.parametrize(
    ("text", "budget", "expected"), [("a\u00e9b", 2, "a"), ("a\u00e9b", 3, "a\u00e9"), ("abc", 0, "")]
)
def test_a_prefix_never_ends_inside_a_character(text: str, budget: int, expected: str):
    assert utf8_prefix(text, budget) == expected
