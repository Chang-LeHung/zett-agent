"""The shell tool: run one command in a directory it names, bounded in time and in what comes back.

The command's two streams go to files and not into memory, so the volume of output costs nothing, and what is
returned is a preview of each (see ``stream_preview`` in :mod:`zett_agent.tools.output`). When a preview had to cut
something, the files are kept and the result ends with a note naming them, which the model can read; otherwise they
are removed. A command that fails or runs out of time is reported as an error that carries its output.
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field

from .._compat import TypeAliasType
from .base import tool
from .output import Preview, stream_preview
from .workspace import Located

#: Where a command's kept output goes, inside the working directory.
OUTPUT_ROOT = ".zett-tool-output"

#: The runtime a command is given unless the model asks for another, and the most it may ask for.
DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600

Command = TypeAliasType("Command", Annotated[str, Field(min_length=1)])
WorkDir = TypeAliasType("WorkDir", Annotated[str, Field(min_length=1)])
TimeoutSeconds = TypeAliasType("TimeoutSeconds", Annotated[int, Field(ge=1, le=MAX_TIMEOUT_SECONDS)])


class CommandFailed(RuntimeError):
    """A command that exited with a code other than 0, or ran out of time. The message carries what it printed."""


def _compose(stdout: Preview, stderr: Preview, kept: tuple[Path, Path] | None, directory: Path) -> str:
    """Join the two previews into the text a result carries, with the note that names the kept files if there are any."""
    sections: list[str] = []
    if stdout.text:
        sections.append(stdout.text)
    if stderr.text:
        sections.append(f"[stderr]\n{stderr.text}")
    composed = "\n".join(sections)
    if kept is not None:
        first, second = (path.relative_to(directory).as_posix() for path in kept)
        composed += f"\n... [output truncated; full streams kept at {first} and {second}]"
    return composed


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Kill a command and every process it started, and wait for it to be gone."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


async def run_command(command: str, timeout_seconds: int, directory: Path | None = None) -> str:
    """Run ``command`` in ``directory``, or in the working directory, and return what it printed.

    Raises:
        ValueError: The command is blank.
        CommandFailed: It exited with a code other than 0, or did not finish in ``timeout_seconds``; the message
            carries its output either way.
    """
    if not command.strip():
        raise ValueError("Shell command cannot be blank")
    root = Located.of(OUTPUT_ROOT)
    folder, stdout_file, stderr_file = _start_folder(root.path)
    timed_out = False
    # File descriptors rather than pipes keep memory bounded however much is printed. The two streams stay
    # separate: how they interleaved is not kept.
    try:
        with stdout_file.open("wb") as stdout_target, stderr_file.open("wb") as stderr_target:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=directory if directory is not None else root.directory,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout_target,
                stderr=stderr_target,
                start_new_session=os.name == "posix",
            )
            try:
                await asyncio.wait_for(process.wait(), timeout_seconds)
            # ``asyncio.TimeoutError`` is the builtin from 3.11 on and a class of its own before.
            except (TimeoutError, asyncio.TimeoutError):
                await _stop(process)
                timed_out = True
            except asyncio.CancelledError:
                await _stop(process)
                raise
    except BaseException:
        # Nothing came back that a caller could name: a command that could not start and a command that was
        # cancelled keep no folder. Removed here rather than inside the ``with``, because a stream this process
        # still holds open cannot be unlinked on Windows, and that error would hide the one that mattered.
        _discard(folder, stdout_file, stderr_file)
        _discard_root(root.path)
        raise
    stdout, stderr = stream_preview(stdout_file), stream_preview(stderr_file)
    truncated = stdout.truncated or stderr.truncated
    composed = _compose(stdout, stderr, (stdout_file, stderr_file) if truncated else None, root.directory)
    if not truncated:
        _discard(folder, stdout_file, stderr_file)
        _discard_root(root.path)
    if timed_out:
        plural = "" if timeout_seconds == 1 else "s"
        raise CommandFailed(f"Command timed out after {timeout_seconds} second{plural}\n{composed}")
    code = process.returncode if process.returncode is not None else -1
    if code != 0:
        raise CommandFailed(f"Command exited with code {code}\n{composed}")
    return composed


def _start_folder(root: Path) -> tuple[Path, Path, Path]:
    """Make the folder one command keeps its streams in, under :data:`OUTPUT_ROOT`.

    Another process sharing this directory removes the root as soon as its own command keeps nothing, which can
    happen between the ``mkdir`` and the ``mkdtemp`` here. That is a race worth losing a microsecond to and not a
    command that never ran: the retry makes the folder and tries once more.
    """
    root.mkdir(parents=True, exist_ok=True)
    try:
        folder = Path(tempfile.mkdtemp(prefix="shell-", dir=root))
    except FileNotFoundError:
        root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="shell-", dir=root))
    return folder, folder / "stdout.txt", folder / "stderr.txt"


def _discard(folder: Path, stdout_file: Path, stderr_file: Path) -> None:
    """Remove the two kept streams and the folder that holds them, if they are still there.

    Best-effort like :func:`_discard_root`: this runs while another error is on its way to the caller, and a
    cleanup that raises would replace the failure that mattered with its own.
    """
    try:
        stdout_file.unlink(missing_ok=True)
        stderr_file.unlink(missing_ok=True)
        folder.rmdir()
    except OSError:
        pass


def _discard_root(root: Path) -> None:
    """Remove :data:`OUTPUT_ROOT` once nothing is kept in it, so a project does not grow an empty folder.

    Another command may be running in it right now, and a folder that still holds a kept stream stays: the reason
    this is a best-effort ``rmdir`` and not a recursive delete.
    """
    try:
        root.rmdir()
    except OSError:
        pass


@tool
async def bash(
    command: Command, work_dir: WorkDir | None = None, timeout_seconds: TimeoutSeconds = DEFAULT_TIMEOUT_SECONDS
) -> str:
    """Run a shell command with captured output, in the working directory or in ``work_dir``.

    Args:
        command: Shell command to execute.
        work_dir: Directory to run in: absolute, a `~` path, or relative to the working directory. Omit it to run in the working directory.
        timeout_seconds: Maximum command runtime in seconds; the default is 60.

    Returns:
        The command's captured output, with startup and final results kept within the byte and line limits; a
        truncation hint names the retained files.

    Snippet:
        bash(command="git status --short", timeout_seconds=60)
        bash(command="npm test", work_dir="clients/vscode")

    Guidelines:
        - Use for builds, tests, git, and searching the code: rg or grep for text, find or ls for paths. Prefer read_file, write_file and edit for reading and changing a file.
        - Set work_dir instead of cd: every call starts a new shell, so a cd lasts only that one command.
        - Search with a narrow pattern and a path, and pipe through head when the result may be long.
        - Use only for bounded commands in a trusted working directory.
        - A non-zero exit code or a timeout is reported as an error carrying the output.
        - Output previews keep startup and final results within byte and line limits.
        - When output is truncated, read the retained files named in the hint.
    """
    if not command.strip():
        raise ValueError("Shell command cannot be blank")
    located = None if work_dir is None else Located.of(work_dir)
    if located is not None and not located.path.is_dir():
        raise ValueError(f"Not a directory: {work_dir}")
    return await run_command(command, timeout_seconds, None if located is None else located.path)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "OUTPUT_ROOT",
    "CommandFailed",
    "bash",
    "run_command",
]
