"""Working-directory tools used by CodingExtension."""

import asyncio
import os
import re
import signal
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .base import tool
from .output import MAX_MATCH_BYTES, MAX_OUTPUT_BYTES, shell_preview, utf8_prefix

FilePath = Annotated[
    str,
    Field(min_length=1),
]
StartLine = Annotated[int, Field(ge=1)]
LineCount = Annotated[int, Field(ge=1, le=2_000)]
TimeoutSeconds = Annotated[int, Field(ge=1, le=300)]
MaxResults = Annotated[int, Field(ge=1, le=5_000)]


MAX_FILE_BYTES = 2 * 1024 * 1024


def _resolve_file_path(path: str) -> tuple[Path, Path]:
    """Resolve an absolute path or a path relative to the current directory."""
    working_directory = Path.cwd().resolve()
    candidate = Path(path).expanduser()
    target = candidate.resolve() if candidate.is_absolute() else (working_directory / candidate).resolve()
    return working_directory, target


def _resolve_glob_pattern(pattern: str) -> tuple[Path, str, bool]:
    """Return the search root, relative glob expression, and output path style."""
    candidate = Path(pattern).expanduser()
    if candidate.is_absolute():
        root = Path(candidate.anchor)
        return root, candidate.relative_to(root).as_posix(), True
    return Path.cwd().resolve(), candidate.as_posix(), False


def _result_path(path: str, working_directory: Path, target: Path) -> str:
    """Preserve relative results for relative inputs and absolute results otherwise."""
    if Path(path).expanduser().is_absolute():
        return target.as_posix()
    return Path(os.path.relpath(target, working_directory)).as_posix()


def _matched_path(candidate: Path, working_directory: Path, *, absolute: bool) -> str:
    """Render a glob match using the same relative/absolute style as its pattern."""
    if absolute:
        return candidate.absolute().as_posix()
    return Path(os.path.relpath(candidate.absolute(), working_directory)).as_posix()


def _read_working_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte limit")
    return path.read_text(encoding="utf-8")


def _write_working_text(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(f"Content exceeds the {MAX_FILE_BYTES}-byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temporary:
            temporary.write(encoded)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@tool
def glob(pattern: Annotated[str, Field(min_length=1)], max_results: MaxResults = 200) -> str:
    """Find paths matching a relative or absolute glob pattern.

    Args:
        pattern: Glob pattern such as **/*.py, ../tests/*.py, or /work/src/**/*.py.
        max_results: Maximum number of sorted paths to return.

    Snippet:
        glob(pattern="src/**/*.py", max_results=200)

    Guidelines:
        - Use glob to discover files before reading or searching them.
        - Narrow the pattern when the result is truncated.
        - The result lists one path per line and is empty when nothing matches.
    """
    working_directory = Path.cwd().resolve()
    root, resolved_pattern, absolute = _resolve_glob_pattern(pattern)
    paths: list[str] = []
    output_bytes = 0
    truncated = False
    for candidate in sorted(root.glob(resolved_pattern)):
        rendered = _matched_path(candidate, working_directory, absolute=absolute)
        size = len(rendered.encode("utf-8"))
        if len(paths) == max_results or output_bytes + size > MAX_OUTPUT_BYTES:
            truncated = True
            break
        paths.append(rendered)
        output_bytes += size
    result = "\n".join(paths)
    if truncated:
        result += "\n... [truncated; narrow the pattern or increase max_results]"
    return result


@tool
def grep(
    pattern: Annotated[str, Field(min_length=1)],
    file_pattern: Annotated[str, Field(min_length=1)] = "**/*",
    case_sensitive: bool = True,
    max_results: MaxResults = 200,
) -> str:
    """Search UTF-8 text files selected by a relative or absolute glob pattern.

    Args:
        pattern: Python regular expression to search for on each line.
        file_pattern: Relative or absolute glob selecting files to search.
        case_sensitive: Whether letter case must match exactly.
        max_results: Maximum number of matching lines to return.

    Snippet:
        grep(pattern="class\\s+Agent", file_pattern="src/**/*.py", max_results=200)

    Guidelines:
        - Use a narrow file_pattern to avoid scanning unrelated files.
        - Escape regular-expression characters when searching for literal text.
        - Each match is one path:line:text line; use read_file for surrounding context.
    """
    working_directory = Path.cwd().resolve()
    root, resolved_file_pattern, absolute = _resolve_glob_pattern(file_pattern)
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        expression = re.compile(pattern, flags)
    except re.error as error:
        raise ValueError(f"Invalid regular expression: {error}") from error

    lines: list[str] = []
    output_bytes = 0
    truncated = False
    for candidate in sorted(root.glob(resolved_file_pattern)):
        if not candidate.is_file():
            continue
        if candidate.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            match = expression.search(line)
            if match is None:
                continue
            if len(lines) == max_results:
                truncated = True
                break
            text_start = max(0, match.start() - 100) if len(line.encode("utf-8")) > MAX_MATCH_BYTES else 0
            preview = utf8_prefix(line[text_start:], MAX_MATCH_BYTES)
            rendered = f"{_matched_path(candidate, working_directory, absolute=absolute)}:{line_number}: {preview}"
            size = len(rendered.encode("utf-8"))
            if output_bytes + size > MAX_OUTPUT_BYTES:
                truncated = True
                break
            lines.append(rendered)
            output_bytes += size
        if truncated:
            break
    result = "\n".join(lines)
    if truncated:
        result += "\n... [truncated; narrow the search or increase max_results]"
    return result


@tool
def read_file(
    path: FilePath, start_line: StartLine = 1, line_count: LineCount = 200, start_column: StartLine = 1
) -> str:
    """Read a line range from a UTF-8 text file.

    Args:
        path: Absolute path or a path relative to the current working directory.
        start_line: One-based first line to return.
        line_count: Maximum number of lines to return.
        start_column: One-based character column on start_line, used to resume a long line.

    Snippet:
        read_file(path="src/app.py", start_line=1, line_count=200)

    Guidelines:
        - Use line ranges for large files.
        - Inspect the current content before editing a file.
        - A trailing hint gives the next start_line and start_column when more content remains.
    """
    _, target = _resolve_file_path(path)
    if not target.is_file():
        raise ValueError(f"File does not exist: {path}")
    selected: list[str] = []
    line_number, column = 1, 1
    used = 0
    next_line = next_column = None
    truncated = False
    with target.open(encoding="utf-8") as source:
        while fragment := source.readline(4096):
            if line_number >= start_line and next_line is None:
                offset = max(0, start_column - column) if line_number == start_line else 0
                available = fragment[offset:]
                if line_number >= start_line + line_count:
                    next_line = line_number
                elif available:
                    part = utf8_prefix(available, MAX_OUTPUT_BYTES - used)
                    selected.append(part)
                    used += len(part.encode("utf-8"))
                    if len(part) < len(available):
                        next_line, next_column = line_number, column + offset + len(part)
                        truncated = True
            if fragment.endswith("\n"):
                line_number, column = line_number + 1, 1
            else:
                column += len(fragment)
    content = "".join(selected)
    if truncated:
        content += f"\n... [line truncated; continue with start_line={next_line}, start_column={next_column}]"
    elif next_line is not None:
        content += f"\n... [more lines; continue with start_line={next_line}]"
    return content


@tool
def write_file(path: FilePath, content: str, overwrite: bool = True) -> str:
    """Atomically write a UTF-8 text file.

    Args:
        path: Absolute path or a path relative to the current working directory.
        content: Complete UTF-8 text to write.
        overwrite: Whether an existing file may be replaced.

    Snippet:
        write_file(path="notes/plan.md", content="# Plan\\n")

    Guidelines:
        - Use for new files or intentional full-file replacement.
        - Prefer replace_in_file for a small change to an existing file.
    """
    working_directory, target = _resolve_file_path(path)
    if target.exists() and not target.is_file():
        raise ValueError(f"Path is not a file: {path}")
    created = not target.exists()
    if not overwrite and not created:
        raise ValueError(f"File already exists: {path}")
    _write_working_text(target, content)
    result_path = _result_path(path, working_directory, target)
    return f"Created {result_path}" if created else f"Wrote {result_path}"


class FileEdit(BaseModel):
    """One exact-text edit applied to a file.

    By default ``old_text`` must appear exactly once in the current text, so
    include the surrounding characters that make it unique. Set ``replace_all``
    when every occurrence should change. ``new_text`` may be empty to delete the
    matched text.
    """

    model_config = ConfigDict(extra="forbid")

    old_text: Annotated[str, Field(min_length=1, description="Exact text to find")]
    new_text: str = Field(default="", description="Replacement text; an empty value deletes the match")
    replace_all: bool = Field(default=False, description="Whether every exact match should be replaced")


@tool
def replace_in_file(
    path: FilePath,
    edits: Annotated[list[FileEdit], Field(min_length=1, description="Ordered exact-text edits applied in one call")],
) -> str:
    """Replace exact text in a UTF-8 file with one or more edits.

    Args:
        path: Absolute path or a path relative to the current working directory.
        edits: Ordered edits applied in one call; each one replaces exact text once, or every
            match when it sets replace_all. Pass a single-element list for one change.

    Snippet:
        replace_in_file(path="notes/plan.md", edits=[{"old_text": "Draft", "new_text": "Final"}])
        replace_in_file(path="notes/plan.md", edits=[{"old_text": "First", "new_text": "1st"}, {"old_text": "Second", "new_text": "2nd", "replace_all": True}])

    Guidelines:
        - Send every change you already know in one call: each edit sees the result of the previous one and the file is written once.
        - Keep replace_all false unless every occurrence should change.
        - Read the file first when the target text may be ambiguous.
    """
    working_directory, target = _resolve_file_path(path)
    if not target.is_file():
        raise ValueError(f"File does not exist: {path}")
    content = _read_working_text(target)
    updated, replacements = _apply_edits(content, edits)
    # One write for the whole batch: a failing edit leaves the file untouched.
    _write_working_text(target, updated)
    result_path = _result_path(path, working_directory, target)
    return f"Replaced {replacements} occurrence{'s' if replacements != 1 else ''} in {result_path}"


def _apply_edits(content: str, edits: Sequence[FileEdit]) -> tuple[str, int]:
    """Apply exact-text edits in order, validating each against the current text."""
    replacements = 0
    for edit in edits:
        matches = content.count(edit.old_text)
        if matches == 0:
            raise ValueError(f"Text to replace was not found: {_excerpt(edit.old_text)}")
        if matches > 1 and not edit.replace_all:
            raise ValueError(
                f"Text to replace is not unique; found {matches} matches "
                f"(set replace_all to change every one): {_excerpt(edit.old_text)}"
            )
        content = content.replace(edit.old_text, edit.new_text, -1 if edit.replace_all else 1)
        replacements += matches if edit.replace_all else 1
    return content, replacements


def _excerpt(value: str, *, limit: int = 60) -> str:
    """Shorten one matched snippet for an error message."""
    collapsed = " ".join(value.split())
    return repr(f"{collapsed[:limit]}…" if len(collapsed) > limit else collapsed)


@tool
def delete_file(path: FilePath) -> str:
    """Delete one regular file by relative or absolute path.

    Args:
        path: Absolute path or a path relative to the current working directory.

    Snippet:
        delete_file(path="notes/obsolete.md")

    Guidelines:
        - Delete a file only when the task explicitly requires its removal.
        - Inspect or confirm the exact path before deleting it.
        - This tool never deletes directories or recursively expands patterns.
    """
    working_directory, target = _resolve_file_path(path)
    if not target.is_file():
        raise ValueError(f"File does not exist: {path}")
    result_path = _result_path(path, working_directory, target)
    target.unlink()
    return f"Deleted {result_path}"


@tool
async def run_shell(
    command: Annotated[str, Field(min_length=1)],
    timeout_seconds: TimeoutSeconds = 30,
) -> str:
    """Run a shell command with captured output from the current working directory.

    Args:
        command: Shell command to execute.
        timeout_seconds: Maximum command runtime in seconds.

    Snippet:
        run_shell(command="git status --short", timeout_seconds=30)

    Guidelines:
        - Use only for bounded commands in a trusted working directory.
        - A non-zero exit code or a timeout is reported as an error carrying the output.
        - Output previews keep startup and final results within byte and line limits.
        - When output is truncated, read the retained files named in the hint.
    """
    if not command.strip():
        raise ValueError("Shell command cannot be blank")
    working_directory, output_root = _resolve_file_path(".zett-tool-output")
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="shell-", dir=output_root))
    stdout_file, stderr_file = directory / "stdout.txt", directory / "stderr.txt"
    timed_out = False
    # Direct file descriptors keep memory bounded regardless of output volume.
    # The two streams remain separate; their cross-stream ordering is not retained.
    with stdout_file.open("wb") as stdout_target, stderr_file.open("wb") as stderr_target:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=working_directory,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=os.name == "posix",
        )
        try:
            async with asyncio.timeout(timeout_seconds):
                await process.wait()
        except (TimeoutError, asyncio.CancelledError) as error:
            # On POSIX stop the entire process group, including child commands.
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            if isinstance(error, asyncio.CancelledError):
                raise
            timed_out = True
    stdout, stdout_truncated = shell_preview(stdout_file)
    stderr, stderr_truncated = shell_preview(stderr_file)
    # Keep both original streams when either preview is incomplete.
    sections: list[str] = []
    if stdout:
        sections.append(stdout)
    if stderr:
        sections.append(f"[stderr]\n{stderr}")
    composed = "\n".join(sections)
    if stdout_truncated or stderr_truncated:
        stdout_path = stdout_file.relative_to(working_directory).as_posix()
        stderr_path = stderr_file.relative_to(working_directory).as_posix()
        composed += f"\n... [output truncated; full streams kept at {stdout_path} and {stderr_path}]"
    else:
        stdout_file.unlink()
        stderr_file.unlink()
        directory.rmdir()
    if timed_out:
        plural = "" if timeout_seconds == 1 else "s"
        raise RuntimeError(f"Command timed out after {timeout_seconds} second{plural}\n{composed}")
    exit_code = process.returncode if process.returncode is not None else -1
    if exit_code != 0:
        raise RuntimeError(f"Command exited with code {exit_code}\n{composed}")
    return composed
