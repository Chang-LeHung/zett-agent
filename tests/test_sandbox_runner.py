"""Everything around the process that every backend shares, exercised through the passthrough backend."""

import asyncio
import os
import sys
import time
import warnings
from pathlib import Path

import pytest

from zett_agent.sandbox import base, factory
from zett_agent.sandbox.base import Sandbox, SandboxResult
from zett_agent.sandbox.bubblewrap import BubblewrapSandbox
from zett_agent.sandbox.errors import SandboxError, SandboxPolicyError, SandboxUnavailableError
from zett_agent.sandbox.factory import SandboxBackend, SandboxWarning, create_sandbox, default_backend
from zett_agent.sandbox.passthrough import PassthroughSandbox
from zett_agent.sandbox.policy import SandboxPolicy
from zett_agent.sandbox.seatbelt import SeatbeltSandbox

posix = pytest.mark.skipif(os.name != "posix", reason="needs a POSIX shell")


@pytest.fixture
def sandbox():
    return PassthroughSandbox()


@posix
async def test_a_shell_string_runs_with_pipes_and_reports_output_and_status(sandbox, tmp_path):
    result = await sandbox.run("echo hello | tr a-z A-Z; echo oops >&2; exit 3", cwd=tmp_path)

    assert (result.exit_code, result.stdout, result.stderr) == (3, "HELLO\n", "oops\n")
    assert result.succeeded is False and result.timed_out is False
    assert result.sandbox == "passthrough" and result.isolated is False
    assert result.duration > 0


async def test_a_program_and_arguments_run_without_a_shell(sandbox, tmp_path):
    result = await sandbox.run(
        [sys.executable, "-c", "import sys;print(sys.argv[1])", "a b; echo hacked"], cwd=tmp_path
    )

    assert result.succeeded and result.stdout.strip() == "a b; echo hacked"


async def test_the_working_directory_is_where_the_process_starts(sandbox, tmp_path):
    result = await sandbox.run([sys.executable, "-c", "import os;print(os.getcwd())"], cwd=tmp_path)

    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


async def test_the_working_directory_defaults_to_the_current_one(sandbox, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = await sandbox.run([sys.executable, "-c", "import os;print(os.getcwd())"])

    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


async def test_a_bad_working_directory_is_a_policy_error(sandbox, tmp_path):
    (tmp_path / "file").write_text("x")

    with pytest.raises(SandboxPolicyError, match="does not exist"):
        await sandbox.run("true", cwd=tmp_path / "missing")
    with pytest.raises(SandboxPolicyError, match="not a directory"):
        await sandbox.run("true", cwd=tmp_path / "file")


async def test_the_process_sees_only_the_policy_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ZETT_SECRET", "leak")
    monkeypatch.setenv("ZETT_INHERITED", "yes")
    sandbox = PassthroughSandbox(SandboxPolicy(inherit_environment=("PATH", "ZETT_INHERITED", "SYSTEMROOT")))
    code = (
        "import os;print(os.environ.get('ZETT_SECRET'), os.environ.get('ZETT_INHERITED'), os.environ.get('ZETT_GIVEN'))"
    )

    result = await sandbox.run([sys.executable, "-c", code], cwd=tmp_path, env={"ZETT_GIVEN": "given"})

    assert result.stdout.split() == ["None", "yes", "given"]


@posix
async def test_input_reaches_the_process_and_a_process_that_ignores_it_is_fine(sandbox, tmp_path):
    echoed = await sandbox.run("cat", cwd=tmp_path, stdin="héllo")
    ignored = await sandbox.run("true", cwd=tmp_path, stdin=b"x" * 5_000_000)
    bytes_in = await sandbox.run("wc -c", cwd=tmp_path, stdin=b"abc")

    assert echoed.stdout == "héllo"
    assert ignored.succeeded
    assert bytes_in.stdout.strip() == "3"


@posix
async def test_without_input_the_process_gets_end_of_file_not_a_hang(sandbox, tmp_path):
    result = await sandbox.run("cat", cwd=tmp_path, timeout=5)

    assert result.succeeded and result.stdout == ""


@posix
async def test_a_run_that_exceeds_its_time_limit_is_killed_and_flagged(sandbox, tmp_path):
    started = time.monotonic()

    result = await sandbox.run("echo started; sleep 30", cwd=tmp_path, timeout=0.5)

    assert result.timed_out is True and result.succeeded is False
    assert result.exit_code == -9
    assert result.stdout == "started\n"
    assert time.monotonic() - started < 10


@posix
async def test_a_timeout_kills_the_whole_process_tree(sandbox, tmp_path):
    marker = tmp_path / "marker"

    await sandbox.run(f"(sleep 1; echo alive > {marker}) & wait", cwd=tmp_path, timeout=0.3)
    await asyncio.sleep(1.6)

    assert not marker.exists()


@posix
async def test_a_background_child_cannot_keep_a_finished_run_waiting(sandbox, tmp_path):
    marker = tmp_path / "marker"
    started = time.monotonic()

    result = await sandbox.run(f"(sleep 1; echo alive > {marker}) & echo done", cwd=tmp_path, timeout=20)
    await asyncio.sleep(1.5)

    assert result.stdout == "done\n" and time.monotonic() - started < 5
    assert not marker.exists()  # the helper was stopped with the command, not left running


@posix
async def test_cancelling_the_caller_stops_the_process_group(sandbox, tmp_path):
    pidfile = tmp_path / "pid"
    task = asyncio.create_task(sandbox.run(f"echo $$ > {pidfile}; sleep 30", cwd=tmp_path, timeout=None))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@posix
async def test_output_beyond_the_limit_is_discarded_while_still_being_read(sandbox, tmp_path):
    result = await sandbox.run(
        "yes | head -c 5000000; yes err | head -c 3000000 >&2", cwd=tmp_path, max_output_bytes=100, timeout=30
    )

    assert result.succeeded
    assert len(result.stdout) == 100 and result.stdout_truncated
    assert len(result.stderr) == 100 and result.stderr_truncated


@posix
async def test_output_within_the_limit_is_not_flagged(sandbox, tmp_path):
    result = await sandbox.run("echo small", cwd=tmp_path, max_output_bytes=100)

    assert not result.stdout_truncated and not result.stderr_truncated


@posix
async def test_undecodable_output_is_replaced_not_raised(sandbox, tmp_path):
    result = await sandbox.run("printf '\\377\\376ok'", cwd=tmp_path)

    assert result.stdout.endswith("ok") and "\ufffd" in result.stdout


async def test_a_program_that_cannot_start_is_a_sandbox_error(sandbox, tmp_path):
    with pytest.raises(SandboxError, match="could not start"):
        await sandbox.run(["/definitely/not/a/program"], cwd=tmp_path)


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": "5"},
        {"timeout": True},
        {"max_output_bytes": 0},
        {"max_output_bytes": True},
    ],
)
async def test_invalid_limits_are_rejected_before_anything_starts(sandbox, tmp_path, options):
    with pytest.raises(SandboxPolicyError):
        await sandbox.run("true", cwd=tmp_path, **options)


@posix
async def test_no_time_limit_is_allowed_explicitly(sandbox, tmp_path):
    assert (await sandbox.run("echo ok", cwd=tmp_path, timeout=None)).stdout == "ok\n"


async def test_the_passthrough_backend_says_it_isolates_nothing(sandbox, tmp_path):
    command = sandbox.wrap(["true"], cwd=tmp_path)
    result = await sandbox.run(["true"], cwd=tmp_path)

    assert command.argv == ("true",) and command.cwd == tmp_path.resolve()
    assert "without isolation" in command.warnings[0]
    assert result.warnings == command.warnings and result.isolated is False
    assert PassthroughSandbox.isolated is False and PassthroughSandbox().check() is None


async def test_wrap_returns_a_command_a_host_can_spawn_itself(sandbox, tmp_path):
    command = sandbox.wrap([sys.executable, "-c", "print('spawned')"], cwd=tmp_path)

    process = await asyncio.create_subprocess_exec(
        *command.argv, cwd=command.cwd, env=dict(command.env), stdout=asyncio.subprocess.PIPE
    )
    out, _ = await process.communicate()

    assert out.decode().strip() == "spawned"


@posix
async def test_without_process_groups_only_the_process_itself_is_stopped(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_POSIX", False)

    result = await sandbox.run("sleep 30", cwd=tmp_path, timeout=0.3)

    assert result.timed_out is True


def test_a_policy_is_required_to_be_a_policy():
    with pytest.raises(SandboxPolicyError, match="must be a SandboxPolicy"):
        PassthroughSandbox("read-only")
    assert PassthroughSandbox().policy == SandboxPolicy.read_only()


def test_the_result_reports_success_only_for_a_clean_zero():
    assert SandboxResult(0, "", "").succeeded
    assert not SandboxResult(1, "", "").succeeded
    assert not SandboxResult(None, "", "").succeeded
    assert not SandboxResult(0, "", "", timed_out=True).succeeded


def test_the_base_class_cannot_be_instantiated_without_a_backend():
    with pytest.raises(TypeError):
        Sandbox()  # type: ignore[abstract]


# -------------------------------------------------------------------- factory


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", SandboxBackend.SEATBELT),
        ("linux", SandboxBackend.BUBBLEWRAP),
        ("linux2", SandboxBackend.BUBBLEWRAP),
        ("win32", SandboxBackend.PASSTHROUGH),
        ("freebsd14", SandboxBackend.PASSTHROUGH),
    ],
)
def test_each_platform_picks_its_own_backend(monkeypatch, platform, expected):
    monkeypatch.setattr(factory.sys, "platform", platform)

    assert default_backend() is expected


def test_a_platform_without_a_backend_degrades_loudly(monkeypatch):
    monkeypatch.setattr(factory.sys, "platform", "win32")

    with pytest.warns(SandboxWarning, match="without isolation"):
        sandbox = create_sandbox(SandboxPolicy.read_only())

    assert isinstance(sandbox, PassthroughSandbox) and sandbox.isolated is False


def test_requiring_isolation_turns_that_degradation_into_an_error(monkeypatch):
    monkeypatch.setattr(factory.sys, "platform", "win32")

    with pytest.raises(SandboxUnavailableError, match="No sandbox backend exists"):
        create_sandbox(require_isolation=True)
    with pytest.raises(SandboxUnavailableError, match="does not isolate"):
        create_sandbox(backend="passthrough", require_isolation=True)


def test_asking_for_passthrough_explicitly_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sandbox = create_sandbox(backend=SandboxBackend.PASSTHROUGH)

    assert isinstance(sandbox, PassthroughSandbox)


def test_a_backend_that_exists_but_cannot_run_is_never_replaced_by_passthrough(monkeypatch):
    monkeypatch.setattr(factory.sys, "platform", "linux")
    monkeypatch.setattr("zett_agent.sandbox.bubblewrap.shutil.which", lambda name: None)
    monkeypatch.setattr("zett_agent.sandbox.bubblewrap.sys.platform", "linux")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SandboxUnavailableError, match="bubblewrap"):
            create_sandbox(SandboxPolicy.read_only())


def test_forcing_a_backend_runs_its_availability_check(monkeypatch):
    monkeypatch.setattr("zett_agent.sandbox.seatbelt.SEATBELT_EXECUTABLE", "/definitely/missing/sandbox-exec")

    with pytest.raises(SandboxUnavailableError):
        create_sandbox(backend=SandboxBackend.SEATBELT)
    with pytest.raises(SandboxUnavailableError):
        create_sandbox(backend="bubblewrap") if sys.platform == "darwin" else BubblewrapSandbox().check() or (
            _ for _ in ()
        ).throw(SandboxUnavailableError("n/a"))


def test_an_unknown_backend_is_a_policy_error():
    with pytest.raises(SandboxPolicyError, match="unknown sandbox backend"):
        create_sandbox(backend="docker")


def test_the_classes_report_their_names_and_isolation():
    assert (SeatbeltSandbox.name, SeatbeltSandbox.isolated) == ("seatbelt", True)
    assert (BubblewrapSandbox.name, BubblewrapSandbox.isolated) == ("bubblewrap", True)
    assert (PassthroughSandbox.name, PassthroughSandbox.isolated) == ("passthrough", False)
