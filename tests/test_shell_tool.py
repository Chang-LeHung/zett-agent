"""The shell tool: where a command runs, what comes back, and what stays on disk when it is cut."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from zett_agent.tools.output import OMISSION
from zett_agent.tools.shell import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    OUTPUT_ROOT,
    CommandFailed,
    bash,
    run_command,
)

posix = pytest.mark.skipif(sys.platform == "win32", reason="drives the tool with POSIX shell commands")


@pytest.fixture
def here(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


# -- running a command -------------------------------------------------------------------------------------------


@posix
async def test_a_command_returns_what_it_printed_and_leaves_nothing_behind(here: Path):
    assert (await run_command("echo hi", 10)).strip() == "hi"
    assert not (here / OUTPUT_ROOT).exists(), "a short output keeps no files, not even the folder"


@posix
async def test_the_output_folder_stays_when_a_stream_was_kept(here: Path):
    result = await run_command("yes line | head -n 20000", 30)
    note = result.rsplit("\n", 1)[1]
    kept = here / note.split("kept at ")[1].split(" and ")[0]

    assert kept.is_file() and kept.read_text().count("line\n") == 20_000
    # A later short command must not take the folder that still holds the kept stream with it.
    assert (await run_command("echo hi", 10)).strip() == "hi"
    assert kept.is_file(), "a folder another command still names stays"


@posix
async def test_a_command_starts_in_the_working_directory(here: Path):
    assert (await run_command("pwd", 10)).strip() == str(here.resolve())


@posix
async def test_a_command_can_be_run_in_another_directory(here: Path):
    elsewhere = here / "sub" / "deeper"
    elsewhere.mkdir(parents=True)

    assert (await run_command("pwd", 10, elsewhere)).strip() == str(elsewhere.resolve())


@posix
async def test_the_tool_runs_a_command_in_the_directory_it_is_given(here: Path):
    (here / "sub").mkdir()
    (here / "sub" / "marker.txt").write_text("here\n")

    assert (await bash({"command": "pwd", "work_dir": "sub"})).strip() == str((here / "sub").resolve())
    assert (await bash({"command": "cat marker.txt", "work_dir": str(here / "sub")})).strip() == "here"
    assert (await bash({"command": "pwd"})).strip() == str(here.resolve()), (
        "a call that names no directory runs in the working directory"
    )


@posix
async def test_a_work_dir_that_is_not_a_directory_is_refused_before_the_command_runs(here: Path):
    (here / "a.txt").write_text("one\n")

    with pytest.raises(ValueError, match="Not a directory: a.txt"):
        await bash({"command": "touch ran", "work_dir": "a.txt"})
    with pytest.raises(ValueError, match="Not a directory: nope"):
        await bash({"command": "touch ran", "work_dir": "nope"})
    assert not (here / "ran").exists()


@posix
async def test_stderr_is_labelled_and_comes_after_stdout(here: Path):
    result = await run_command("echo out; echo err >&2", 10)

    assert result == "out\n\n[stderr]\nerr\n"


@posix
async def test_a_command_that_prints_only_to_stderr_is_just_labelled(here: Path):
    assert await run_command("echo err >&2", 10) == "[stderr]\nerr\n"


@posix
async def test_a_failing_command_is_an_error_that_carries_its_output(here: Path):
    with pytest.raises(CommandFailed) as raised:
        await run_command("echo partial; exit 3", 10)

    assert str(raised.value) == "Command exited with code 3\npartial\n"
    assert isinstance(raised.value, RuntimeError)


@posix
async def test_a_command_that_runs_out_of_time_is_stopped_with_everything_it_started(here: Path):
    marker = here / "survived"

    with pytest.raises(CommandFailed, match=r"timed out after 1 second\n"):
        await run_command(f"(sleep 1.5; touch {marker}) & echo begun; wait", 1)

    await asyncio.sleep(1.7)
    assert not marker.exists(), "the child of the command was killed with it"


@posix
async def test_a_cancelled_command_is_stopped_too(here: Path):
    marker = here / "survived"
    task = asyncio.ensure_future(run_command(f"sleep 1; touch {marker}", 30))
    await asyncio.sleep(0.3)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(1.2)
    assert not marker.exists()
    assert not (here / OUTPUT_ROOT).exists(), "a cancelled command keeps no folder: nothing names its output"


@posix
async def test_a_command_that_reads_input_gets_none_instead_of_waiting_for_it(here: Path):
    assert (await run_command("cat; echo done", 5)).strip() == "done"


@posix
async def test_a_long_output_is_previewed_and_the_whole_of_it_is_kept_and_named(here: Path):
    result = await run_command("yes line | head -n 20000", 30)

    assert OMISSION.strip() in result
    note = result.rsplit("\n", 1)[1]
    assert note.startswith("... [output truncated; full streams kept at ") and note.endswith("]")
    first = note.split("kept at ")[1].split(" and ")[0]
    assert first.startswith(f"{OUTPUT_ROOT}/shell-") and first.endswith("/stdout.txt")
    assert (here / first).read_text().count("line\n") == 20_000


@posix
async def test_a_kept_stream_is_readable_through_the_file_tools(here: Path):
    from zett_agent.tools.files import read_file

    program = "import sys; print('START'); print('x' * 3000000); print('END'); print('ERROR', file=sys.stderr)"
    result = await bash({"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"})

    assert result.startswith("START") and "END\n" in result and "[stderr]\nERROR" in result
    stdout_path = next((here / OUTPUT_ROOT).rglob("stdout.txt")).relative_to(here).as_posix()
    stderr_path = next((here / OUTPUT_ROOT).rglob("stderr.txt")).relative_to(here).as_posix()
    assert stdout_path in result and stderr_path in result
    assert (await read_file({"file_path": stdout_path, "offset": 3})).startswith("3\tEND")


async def test_a_folder_that_vanished_between_two_calls_is_made_again(here: Path, monkeypatch):
    """Another process sharing the directory can remove an empty root while this one is starting a command."""
    from zett_agent.tools import shell as shell_module

    real_mkdtemp = shell_module.tempfile.mkdtemp
    attempts = []

    def lose_the_race_once(prefix, dir):
        attempts.append(dir)
        if len(attempts) == 1:
            raise FileNotFoundError(2, "No such file or directory", str(dir))
        return real_mkdtemp(prefix=prefix, dir=dir)

    monkeypatch.setattr(shell_module.tempfile, "mkdtemp", lose_the_race_once)

    assert (await run_command("echo hi", 10)).strip() == "hi"
    assert len(attempts) == 2 and not (here / OUTPUT_ROOT).exists()


def test_discarding_a_folder_that_holds_something_else_raises_nothing(here: Path):
    """The cleanup runs while an error is on its way out, so it may fail and must not say so."""
    from zett_agent.tools.shell import _discard

    folder = here / "shell-kept"
    folder.mkdir()
    (folder / "someone-elses").write_text("not ours")
    stdout_file, stderr_file = folder / "stdout.txt", folder / "stderr.txt"
    stdout_file.write_text("a")
    stderr_file.write_text("b")

    _discard(folder, stdout_file, stderr_file)

    assert not stdout_file.exists() and not stderr_file.exists()
    assert (folder / "someone-elses").read_text() == "not ours", "a folder that is not ours stays"


async def test_a_command_that_cannot_start_leaves_no_folder_behind(here: Path, monkeypatch):
    """No shell, no folder: the launch fails before anything could be kept, and the cleanup must not mask it."""
    from zett_agent.tools import shell as shell_module

    async def refuse(*args, **kwargs):
        raise OSError("no shell here")

    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_shell", refuse)

    with pytest.raises(OSError, match="no shell here"):
        await run_command("echo hi", 10)
    assert not (here / OUTPUT_ROOT).exists()


@posix
async def test_a_command_whose_group_is_already_gone_still_ends_in_a_timeout_error(here: Path, monkeypatch):
    """The kill can lose the race with an exiting command; the tool reports the timeout either way."""
    from zett_agent.tools import shell as shell_module

    def gone(pid, signal_number):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(shell_module.os, "killpg", gone, raising=False)

    with pytest.raises(CommandFailed, match="timed out after 1 second"):
        await run_command("sleep 1.1", 1)


@posix
async def test_a_blank_command_is_refused(here: Path):
    with pytest.raises(ValueError, match="blank"):
        await run_command("  ", 5)
    with pytest.raises(ValueError, match="blank"):
        await bash({"command": "   "})


# -- the tool as the model sees it -------------------------------------------------------------------------------


def test_the_tool_is_typed_the_way_the_model_is_told():
    assert bash.name == "bash"
    assert bash.parameters["required"] == ["command"]
    assert bash.parameters["properties"]["timeout_seconds"]["default"] == DEFAULT_TIMEOUT_SECONDS
    assert bash.parameters["properties"]["timeout_seconds"]["maximum"] == MAX_TIMEOUT_SECONDS
    assert bash.parameters["properties"]["command"]["minLength"] == 1
    assert bash.parameters["properties"]["work_dir"]["default"] is None, (
        "a call without one runs in the working directory"
    )
    assert DEFAULT_TIMEOUT_SECONDS == 60 and MAX_TIMEOUT_SECONDS == 600
    assert "timeout_seconds=60" in bash.snippet and 'work_dir="clients/vscode"' in bash.snippet
    assert any("Set work_dir instead of cd" in guideline for guideline in bash.guidelines)


@posix
async def test_the_tool_takes_its_arguments_by_name_and_validates_them(here: Path):
    assert (await bash({"command": "echo hi"})).strip() == "hi"
    with pytest.raises(ValidationError):
        await bash({"command": "echo hi", "timeout_seconds": 601})
    with pytest.raises(ValidationError):
        await bash({"command": ""})


def test_the_shell_tool_guidance_steers_the_model_away_from_shell_edits():
    guidance = " ".join(bash.guidelines)
    assert "Prefer read_file, write_file and edit for reading and changing a file" in guidance
    assert "A non-zero exit code or a timeout is reported as an error carrying the output" in guidance
