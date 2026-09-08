"""Bounded UTF-8 previews; full shell output stays in files for later reads."""

from pathlib import Path

MAX_OUTPUT_BYTES = 50 * 1024
MAX_OUTPUT_LINES = 2000
MAX_MATCH_BYTES = 2000


def utf8_prefix(text: str, budget: int) -> str:
    """Keep a valid UTF-8 prefix without exceeding the byte budget."""
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def shell_preview(path: Path) -> tuple[str, bool]:
    """Keep a quarter of the budget for startup and the rest for final results.

    Read only bounded windows from disk. Line and byte limits apply to the
    returned text, including the omission marker. UTF-8 boundary fragments are
    dropped. Full output remains available even when errors occur in the middle.

    Preview construction (the file on disk is never changed)::

        +------------------------------------------------------------+
        | FULL OUTPUT FILE                                           |
        | START .............................................. END   |
        +------------------------------------------------------------+
                                      |
                                      v
        +------------------------------------------------------------+          +---------------------------+
        | PROBE                                                      |-- yes -->| RETURN FULL TEXT          |
        | Read at most MAX_OUTPUT_BYTES + 1 bytes                    |          | truncated = False         |
        | Does all text fit BOTH byte and line limits?               |          +---------------------------+
        +------------------------------------------------------------+
                                      |
                                      | no
                                      v
        +------------------------------------------------------------+
        | RESERVE MARKER SPACE                                       |
        | B = byte limit - marker size                               |
        +------------------------------------------------------------+
                                      |
                                      |
                                      v
        +------------------------------------------------------------+
        | READ BOUNDED BYTE WINDOWS                                  |
        | Head: B // 4 bytes from START                              |
        | Tail: remaining bytes by seeking to END                    |
        | Middle: omitted from preview                               |
        +------------------------------------------------------------+
                                      |
                                      |
                                      v
        +------------------------------------------------------------+
        | DECODE AND TRIM                                            |
        | Drop partial UTF-8 characters at window boundaries         |
        | Reserve 4 lines for marker and boundary newlines           |
        | Keep first ~1/4 of remaining lines from head               |
        | Keep last ~3/4 of remaining lines from tail                |
        +------------------------------------------------------------+
                                      |
                                      |
                                      v
        +------------------------------------------------------------+
        | ASSEMBLE PREVIEW                                           |
        | Head preview + omission marker + tail preview              |
        | Return combined text, truncated = True                     |
        +------------------------------------------------------------+

    Byte windows are selected before line trimming; unused space is not
    redistributed. The 1:3 split is a budget allocation, not a guarantee of
    final text lengths. Window edges may contain partial lines. Consumers
    can read the original file to recover omitted text.
    """
    marker = "\n... [middle omitted; read the output file for full content] ...\n"
    with path.open("rb") as source:
        data = source.read(MAX_OUTPUT_BYTES + 1)
        text = data.decode("utf-8", errors="replace")
        if len(data) <= MAX_OUTPUT_BYTES and len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES:
            if len(text.splitlines()) <= MAX_OUTPUT_LINES:
                return text, False
        budget = MAX_OUTPUT_BYTES - len(marker.encode("utf-8"))
        head_budget = budget // 4
        tail_budget = budget - head_budget
        head = data[:head_budget].decode("utf-8", errors="ignore")
        source.seek(max(0, path.stat().st_size - tail_budget))
        tail = source.read(tail_budget).decode("utf-8", errors="ignore")
    # Reserve lines for the marker and boundary newlines.
    lines = max(2, MAX_OUTPUT_LINES - 4)
    head_lines = max(1, lines // 4)
    head = "".join(head.splitlines(keepends=True)[:head_lines])
    tail = "".join(tail.splitlines(keepends=True)[-(lines - head_lines) :])
    return head + marker + tail, True
