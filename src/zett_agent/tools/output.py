"""Bounded UTF-8 previews: a long text is cut to a budget, and the full output stays in a file for later reads."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

#: The most bytes, lines and bytes-per-matching-line any one tool result may carry.
MAX_OUTPUT_BYTES = 50 * 1024
MAX_OUTPUT_LINES = 2000
MAX_MATCH_BYTES = 2000

#: What stands where the middle of a long output was left out.
OMISSION = "\n... [middle omitted; read the output file for full content] ...\n"


class Preview(NamedTuple):
    """A bounded view of one output stream.

    Attributes:
        text: What fits the budget.
        truncated: Whether the stream was longer than that, so the full text has to be kept.
    """

    text: str
    truncated: bool


def utf8_prefix(text: str, budget: int) -> str:
    """Return the longest prefix of ``text`` whose UTF-8 encoding fits in ``budget`` bytes, never half a character."""
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def stream_preview(path: Path, marker: str = OMISSION) -> Preview:
    """Return a bounded preview of a stream written to ``path``: its start and its end, with the middle left out.

    A quarter of the byte budget goes to the start (what a command printed while it started up) and the rest to
    the end (its final results). Only bounded windows are read from disk, so a gigabyte of output costs what fifty
    kilobytes does, and a partial UTF-8 character at either edge of a window is dropped. The line limit applies to
    the text returned, the marker included. The file is never changed, which is what lets the model read what was
    omitted.

    Args:
        path: The stream that was written.
        marker: What stands between the two windows. A caller that knows where the whole text is kept puts that in
            the marker; it is written once, where the cut is, so a copy of the default marker inside the text cannot
            be mistaken for the cut.
    """
    with path.open("rb") as source:
        data = source.read(MAX_OUTPUT_BYTES + 1)
        text = data.decode("utf-8", errors="replace")
        if (
            len(data) <= MAX_OUTPUT_BYTES
            and len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES
            and len(text.splitlines()) <= MAX_OUTPUT_LINES
        ):
            return Preview(text, False)
        budget = max(2, MAX_OUTPUT_BYTES - len(marker.encode("utf-8")))
        head_budget = budget // 4
        tail_budget = budget - head_budget
        head = data[:head_budget].decode("utf-8", errors="ignore")
        source.seek(max(0, path.stat().st_size - tail_budget))
        tail = source.read(tail_budget).decode("utf-8", errors="ignore")
    # What the marker takes is held back so the line limit covers the text that comes back: a caller's marker is a
    # sentence or two, and the floor keeps a long one from eating the whole preview.
    lines = max(2, MAX_OUTPUT_LINES - max(1, marker.count("\n")) - 1)
    head_lines = max(1, lines // 4)
    head = "".join(head.splitlines(keepends=True)[:head_lines])
    tail = "".join(tail.splitlines(keepends=True)[-(lines - head_lines) :])
    return Preview(head + marker + tail, True)


__all__ = [
    "MAX_MATCH_BYTES",
    "MAX_OUTPUT_BYTES",
    "MAX_OUTPUT_LINES",
    "OMISSION",
    "Preview",
    "stream_preview",
    "utf8_prefix",
]
