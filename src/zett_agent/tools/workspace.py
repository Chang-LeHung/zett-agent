"""The file system as the tools see it: paths taken from the working directory, text read in ranges, written whole."""

from __future__ import annotations

import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .output import MAX_OUTPUT_BYTES, utf8_prefix

#: The largest text file a tool reads whole, writes, or searches.
MAX_FILE_BYTES = 2 * 1024 * 1024

#: How much of a line is read at a time, so one enormous line cannot use the memory of a whole file.
FRAGMENT_CHARACTERS = 4096


# -- where a path points ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Located:
    """A path a tool was asked about, resolved against the directory the process is in.

    Attributes:
        given: The path as the model wrote it, which is what an error message quotes back.
        directory: The working directory, resolved.
        path: Where ``given`` points: absolute paths and ``~`` are taken as they are, anything else is relative to
            ``directory``; symbolic links are followed.
    """

    given: str
    directory: Path
    path: Path

    @classmethod
    def of(cls, given: str) -> Located:
        """Resolve ``given`` against the working directory."""
        directory = Path.cwd().resolve()
        candidate = Path(given).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (directory / candidate).resolve()
        return cls(given, directory, target)

    @property
    def shown(self) -> str:
        """Return the path as a result should name it: relative to the working directory for a relative ``given``."""
        if Path(self.given).expanduser().is_absolute():
            return self.path.as_posix()
        return Path(os.path.relpath(self.path, self.directory)).as_posix()

    def require_file(self) -> Path:
        """Return the path, which has to be an existing regular file.

        Raises:
            ValueError: With the path as given, when there is no such file.
        """
        if not self.path.is_file():
            raise ValueError(f"File does not exist: {self.given}")
        return self.path


@dataclass(frozen=True, slots=True)
class Pattern:
    """A glob pattern, split into the directory to search from and the expression to match below it.

    Attributes:
        root: Where the search starts: the filesystem root for an absolute pattern, else the working directory.
        expression: What is matched below ``root``, as ``Path.glob`` takes it.
        absolute: Whether matches are reported as absolute paths (the pattern was one) or relative to ``directory``.
        directory: The working directory, resolved.
    """

    root: Path
    expression: str
    absolute: bool
    directory: Path

    @classmethod
    def parse(cls, pattern: str) -> Pattern:
        """Split ``pattern`` the way ``glob`` and ``grep`` take it."""
        directory = Path.cwd().resolve()
        candidate = Path(pattern).expanduser()
        if candidate.is_absolute():
            root = Path(candidate.anchor)
            return cls(root, candidate.relative_to(root).as_posix(), True, directory)
        return cls(directory, candidate.as_posix(), False, directory)

    def matches(self) -> list[Path]:
        """Return everything the pattern matches, in sorted order."""
        return sorted(self.root.glob(self.expression))

    def render(self, match: Path) -> str:
        """Return a match in the style the pattern was written in."""
        if self.absolute:
            return match.absolute().as_posix()
        return Path(os.path.relpath(match.absolute(), self.directory)).as_posix()


# -- reading and writing text ------------------------------------------------------------------------------------


def read_text(located: Located) -> str:
    """Return a whole text file.

    Raises:
        ValueError: When there is no such file, or it is over :data:`MAX_FILE_BYTES`; the message quotes the path
            as the model gave it.
        UnicodeDecodeError: When it is not UTF-8.
    """
    path = located.require_file()
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte limit: {located.given}")
    return path.read_text(encoding="utf-8")


def _open_temporary(directory: Path, mode: int) -> tuple[Path, int]:
    """Create a file nobody else holds in ``directory`` and return its path and open descriptor.

    ``O_EXCL`` on a name of our own means the file is new, so the kernel applies the process umask to ``mode`` in
    the call that creates it: that is the one way to get a created file's mode without reading the umask, and the
    one way to read the umask is to set it — a pair of calls that two threads can interleave into a leaked 0.

    ``mode`` is the mode the file it will replace has, so a secret being rewritten is never readable while it is
    being written. For a new file the caller passes ``0o666``, the mode a plain create gives.
    """
    for _ in range(10):
        candidate = directory / f"{secrets.token_hex(8)}.tmp"
        try:
            return candidate, os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        except FileExistsError:
            continue
    raise ValueError(f"Cannot make a temporary file in {directory}")


def write_text(path: Path, content: str) -> None:
    """Replace a file with ``content`` atomically: it is written beside the file and renamed over it.

    A reader of the file sees the old text or the new text, never half of it, and a failed write leaves the old
    text and no stray file behind. Missing parent directories are made.

    The replacement keeps what a plain write would: an existing file keeps its permission bits, minus the
    set-user-ID and set-group-ID bits a write clears anyway, and a new file gets the mode a create would give it.
    The file it replaces is never more readable than it was while the new text is written: the temporary carries
    that mode from the moment it exists. The text is written with ``\\n`` line breaks, so a CRLF file comes back
    with LF endings and an edit of one line reaches the rest of the file too.

    Raises:
        ValueError: When the text is over :data:`MAX_FILE_BYTES`.
    """
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(f"Content exceeds the {MAX_FILE_BYTES}-byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = None
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode) & ~(stat.S_ISUID | stat.S_ISGID)
    temporary, descriptor = _open_temporary(path.parent, 0o666 if mode is None else mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
        # The permissions go on the temporary file, before it takes the place of the old text: a chmod that fails
        # then leaves the file as it was instead of committing the new text and reporting a failure.
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_lines(located: Located, offset: int, limit: int) -> str:
    """Return ``limit`` lines of a UTF-8 file from line ``offset``, with a note on how to continue when more remains.

    Each line is prefixed with its 1-based number and a tab (``<number>\\t<content>``); the prefix is for reference
    and is not part of the file.

    The result is held to :data:`MAX_OUTPUT_BYTES`. A line longer than that is cut, and the note says so: the
    reader has no column to resume from, because one 2 MB line is not what the tool is for — the shell reads it.
    A cut inside an ordinary line is not that: the budget was used up before it, and the note then names the line
    to start again from. The file is read a fragment at a time and never whole.

    Args:
        located: The file; it has to exist.
        offset: One-based first line.
        limit: Most lines to return.

    Raises:
        ValueError: When the file is not UTF-8 text.
    """
    selected: list[str] = []
    number, used, line_start, emitted = 1, 0, 0, 0
    next_offset: int | None = None
    cut_at: int | None = None
    oversized = False
    try:
        with located.path.open(encoding="utf-8") as source:
            while fragment := source.readline(FRAGMENT_CHARACTERS):
                if number >= offset + limit:
                    next_offset = number
                    break
                if number >= offset:
                    # The prefix belongs to the first fragment of a line; a long line is read in pieces, and
                    # repeating the number for each would put it in the middle of the content.
                    prefix = "" if number == emitted else f"{number}\t"
                    numbered = prefix + fragment
                    part = utf8_prefix(numbered, MAX_OUTPUT_BYTES - used)
                    if len(part) <= len(prefix):
                        # The budget ran out inside the line number, or exactly at it. A bare number would read
                        # as content, so emit none of the line and send the reader back to it.
                        cut_at = number
                        oversized = _line_longer_than_budget(source, used - line_start, numbered, MAX_OUTPUT_BYTES)
                        break
                    selected.append(part)
                    used += len(part.encode("utf-8"))
                    emitted = number
                    if len(part) < len(numbered):
                        # Whether this line could fill a call of its own decides what the note says, and only
                        # what is left of the line knows that. Reading on is bounded by one budget.
                        cut_at = number
                        oversized = _line_longer_than_budget(
                            source, used - line_start, numbered[len(part) :], MAX_OUTPUT_BYTES
                        )
                        break
                if fragment.endswith("\n"):
                    number, line_start = number + 1, used
    except UnicodeDecodeError as error:
        raise ValueError(f"File is not UTF-8 text: {located.given}") from error
    content = "".join(selected)
    if cut_at is not None and oversized:
        content += (
            f"\n... [line truncated; line {cut_at} is longer than the {MAX_OUTPUT_BYTES // 1024} KB output limit "
            "and was cut here; read it with the shell]"
        )
    elif cut_at is not None:
        content += f"\n... [output truncated; continue with offset={cut_at}]"
    elif next_offset is not None:
        content += f"\n... [more lines; continue with offset={next_offset}]"
    return content


def _line_longer_than_budget(source: TextIO, seen: int, tail: str, budget: int) -> bool:
    """Return whether the line continues for more than ``budget`` bytes in all.

    Args:
        source: The file, positioned just after the fragment the cut happened in.
        seen: How many bytes of this line are already in the result.
        tail: The rest of the fragment the cut happened in, which is not in the result yet.
        budget: What one result may carry.

    The line is read on to its end, but never past ``budget`` bytes: past that point the answer is known, and
    the rest of a runaway line is what the shell is for.
    """
    fragment = tail
    while True:
        seen += len(fragment.encode("utf-8"))
        if seen > budget:
            return True
        if not fragment or fragment.endswith("\n"):
            return False
        fragment = source.readline(FRAGMENT_CHARACTERS)


# -- editing ------------------------------------------------------------------------------------------------------


def apply_edit(content: str, old_string: str, new_string: str, replace_all: bool = False) -> tuple[str, int]:
    """Replace ``old_string`` in ``content`` once, or everywhere when ``replace_all`` says so.

    By default ``old_string`` must appear exactly once in the text, so include the surrounding characters that
    make it unique; ``new_string`` may be empty, which deletes the match.

    Returns:
        The new text and how many occurrences were replaced.

    Raises:
        ValueError: Saying what to do, when the old text is not found or is not unique. Nothing has been written
            by then, since the caller writes once, after the replacement has passed.
    """
    matches = content.count(old_string)
    if matches == 0:
        raise ValueError(f"Text to replace was not found: {_excerpt(old_string)}. {_miss_hint(content, old_string)}")
    if matches > 1 and not replace_all:
        raise ValueError(
            f"Text to replace is not unique; found {matches} matches "
            f"(include more surrounding lines, or set replace_all to change every one): {_excerpt(old_string)}"
        )
    updated = content.replace(old_string, new_string, -1 if replace_all else 1)
    return updated, matches if replace_all else 1


def _miss_hint(content: str, old_string: str) -> str:
    """Say why an exact match failed, so the model fixes ``old_string`` instead of leaving the tool."""
    if _numbered_copy_matches(content, old_string):
        return (
            "A read prefixes each line with its number and a tab, and those are not in the file; "
            "copy the lines again without them. Nothing was written."
        )
    if " ".join(content.split()).count(" ".join(old_string.split())) > 0:
        return (
            "A match exists that differs only in whitespace or indentation; "
            "read the file again and copy the lines exactly. Nothing was written."
        )
    return "Read the file again and copy old_string from its current content. Nothing was written."


#: A line a read returned: its 1-based number, a tab, then the file's own text.
_NUMBERED_LINE = re.compile(r"^[ \t]*\d+\t", re.MULTILINE)


def _numbered_copy_matches(content: str, old_string: str) -> bool:
    """Return whether ``old_string`` matches once a read's line numbers and tabs are removed."""
    stripped = _NUMBERED_LINE.sub("", old_string)
    return stripped != old_string and stripped in content


def _excerpt(value: str, *, limit: int = 60) -> str:
    """Shorten one matched snippet for an error message."""
    collapsed = " ".join(value.split())
    return repr(f"{collapsed[:limit]}\u2026" if len(collapsed) > limit else collapsed)


__all__ = [
    "FRAGMENT_CHARACTERS",
    "MAX_FILE_BYTES",
    "Located",
    "Pattern",
    "apply_edit",
    "read_lines",
    "read_text",
    "write_text",
]
