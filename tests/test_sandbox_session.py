"""The simple session API: configuration, the workspace lifecycle, and cleanup on every exit path."""

import asyncio
import gc
import os
import signal
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest

from zett_agent.sandbox import session as session_module
from zett_agent.sandbox.errors import (
    SandboxClosedError,
    SandboxCommandError,
    SandboxError,
    SandboxPolicyError,
    SandboxUnavailableError,
)
from zett_agent.sandbox.policy import NetworkMode
from zett_agent.sandbox.session import SandboxSession, parse_size

posix_only = pytest.mark.skipif(os.name != "posix", reason="needs a POSIX shell")


def session(*args, **kwargs) -> SandboxSession:
    """A session on the passthrough backend, so these cases test the session, not the platform."""
    kwargs.setdefault("backend", "passthrough")
    return SandboxSession(*args, **kwargs)


# ------------------------------------------------------------------ sizes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("512MB", 512 * 1024**2),
        ("512 mb", 512 * 1024**2),
        ("512MiB", 512 * 1024**2),
        ("512m", 512 * 1024**2),
        ("1.5G", int(1.5 * 1024**3)),
        ("2GB", 2 * 1024**3),
        ("64K", 64 * 1024),
        ("1T", 1024**4),
        ("65536", 65536),
        ("100B", 100),
        (123, 123),
    ],
)
def test_sizes_are_read_the_way_people_write_them(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "lots", "5 parsecs", "-5MB", "MB", None, 1.5, True, [1]])
def test_a_value_that_is_not_a_size_is_rejected_with_a_hint(bad):
    with pytest.raises(SandboxPolicyError, match="memory"):
        parse_size(bad, "memory")


# ------------------------------------------------------ configuration errors


@pytest.mark.parametrize(
    "options",
    [
        {"memory": "lots"},
        {"memory": "1MB"},
        {"cpus": 0},
        {"cpus": -1},
        {"cpus": "half"},
        {"cpus": True},
        {"processes": 2},
        {"open_files": 4},
        {"max_file_size": "big"},
        {"network": "yes"},
        {"backend": "docker"},
        {"memory_strategy": "maybe"},
    ],
)
def test_a_bad_option_fails_when_the_session_is_built_not_at_the_first_command(options):
    with pytest.raises((SandboxPolicyError, ValueError)):
        SandboxSession(**options)


def test_cpus_are_cores_and_become_a_percentage():
    sb = SandboxSession(cpus=0.5, memory="64MB", cpu_seconds=10, processes=16, open_files=64, max_file_size="1MB")

    limits = sb._limits
    assert (limits.cpu_percent, limits.memory_bytes, limits.cpu_seconds) == (50, 64 * 1024**2, 10)
    assert (limits.processes, limits.open_files, limits.file_size_bytes) == (16, 64, 1024**2)
    assert SandboxSession(cpus=2)._limits.cpu_percent == 200
    assert SandboxSession(cpus=0.001)._limits.cpu_percent == 1


def test_the_network_is_off_unless_asked_for():
    assert SandboxSession()._network is NetworkMode.NONE
    assert SandboxSession(network=True)._network is NetworkMode.FULL
    assert SandboxSession(network=NetworkMode.FULL)._network is NetworkMode.FULL


# -------------------------------------------------------------- lifecycle


def test_nothing_is_created_until_the_session_is_opened():
    sb = session()

    assert sb._root is None and not sb.is_open
    with pytest.raises(SandboxClosedError, match="not open"):
        sb.run("true")
    with pytest.raises(SandboxClosedError, match="not open"):
        _ = sb.workspace


@posix_only
def test_a_temporary_workspace_exists_inside_the_block_and_is_gone_after():
    with session() as sb:
        root = sb._root
        assert sb.is_open and sb.workspace.is_dir() and sb.workspace.parent == root
        (sb.workspace / "kept.txt").write_text("data")
        assert sb.run("cat kept.txt").stdout == "data"

    assert not root.exists() and not sb.is_open


@posix_only
def test_a_workspace_you_supply_is_used_and_never_deleted(tmp_path):
    (tmp_path / "mine.txt").write_text("mine")

    with session(tmp_path) as sb:
        assert sb.workspace == tmp_path.resolve()
        sb.run("echo made > made.txt", check=True)
        root = sb._root

    assert sorted(path.name for path in tmp_path.iterdir()) == ["made.txt", "mine.txt"]
    assert not root.exists()  # the scratch space was the session's, so it went


def test_a_workspace_that_is_missing_or_a_file_is_a_policy_error(tmp_path):
    (tmp_path / "file").write_text("x")
    leftovers = lambda: set(Path(session_module.tempfile.gettempdir()).glob("zett-sandbox-*"))  # noqa: E731
    before = leftovers()

    with pytest.raises(SandboxPolicyError, match="does not exist"):
        session(tmp_path / "missing").open()
    with pytest.raises(SandboxPolicyError, match="not a directory"):
        session(tmp_path / "file").open()

    assert leftovers() == before  # a session that failed to open removed what it made


def test_a_session_that_cannot_find_a_sandbox_leaves_nothing_behind(monkeypatch):
    def unavailable(*args, **kwargs):
        raise SandboxUnavailableError("no sandbox here")

    monkeypatch.setattr(session_module, "create_sandbox", unavailable)
    before = set(Path(session_module.tempfile.gettempdir()).glob("zett-sandbox-*"))

    with pytest.raises(SandboxUnavailableError):
        SandboxSession().open()

    assert set(Path(session_module.tempfile.gettempdir()).glob("zett-sandbox-*")) == before


def test_a_session_cannot_be_opened_twice_or_reopened():
    sb = session()
    sb.open()
    with pytest.raises(SandboxError, match="it is open"):
        sb.open()
    sb.close()
    with pytest.raises(SandboxError, match="it is closed"):
        sb.open()
    with pytest.raises(SandboxClosedError, match="closed"):
        sb.run("true")


def test_closing_twice_or_before_opening_is_harmless():
    sb = session()
    sb.close()
    sb.close()
    other = session().open()
    other.close()
    other.close()


@posix_only
def test_cleanup_happens_when_the_block_raises():
    with pytest.raises(ZeroDivisionError):
        with session() as sb:
            root = sb._root
            sb.run("echo x > f")
            raise ZeroDivisionError

    assert not root.exists()


@posix_only
def test_a_program_that_leaves_unenterable_directories_cannot_block_cleanup():
    with session() as sb:
        root = sb._root
        sb.run("mkdir -p a/b && echo x > a/b/f && chmod 000 a/b a", check=True)

    assert not root.exists()


@posix_only
def test_symlinks_in_the_workspace_are_removed_without_following_them(tmp_path):
    target = tmp_path / "precious"
    target.mkdir()
    (target / "file.txt").write_text("keep")

    with session() as sb:
        (sb.workspace / "link").symlink_to(target)

    assert (target / "file.txt").read_text() == "keep"


def test_a_session_that_is_never_closed_is_still_cleaned_up_when_it_is_dropped():
    sb = session().open()
    root = sb._root
    assert root.exists() and sb._finalizer.alive

    del sb
    gc.collect()

    assert not root.exists()


def test_closing_detaches_the_safety_net(monkeypatch):
    sb = session().open()
    finalizer = sb._finalizer

    sb.close()

    assert not finalizer.alive


def test_a_directory_that_cannot_be_removed_is_reported(monkeypatch):
    sb = session().open()
    monkeypatch.setattr(session_module, "_destroy", lambda path: False)

    with pytest.warns(ResourceWarning, match="could not remove its directory"):
        sb.close()
    session_module.shutil.rmtree(sb._root, ignore_errors=True)


def test_removing_a_directory_that_is_already_gone_succeeds(tmp_path):
    assert session_module._destroy(str(tmp_path / "gone")) is True


def test_a_directory_that_stays_undeletable_reports_false(tmp_path, monkeypatch):
    target = tmp_path / "stuck"
    target.mkdir()

    def refuse(path):
        raise PermissionError("no")

    monkeypatch.setattr(session_module.shutil, "rmtree", refuse)
    monkeypatch.setattr(session_module.time, "sleep", lambda seconds: None)

    assert session_module._destroy(str(target)) is False


# ----------------------------------------------------------------- running


@posix_only
def test_run_behaves_like_subprocess_run():
    with session() as sb:
        ok = sb.run("echo out; echo err >&2")
        failed = sb.run("exit 3")
        program = sb.run([sys.executable, "-c", "import sys;print(sys.argv[1])", "a b; echo hacked"])
        piped = sb.run("tr a-z A-Z", input="hello")

    assert (ok.stdout, ok.stderr, ok.succeeded) == ("out\n", "err\n", True)
    assert failed.exit_code == 3 and not failed.succeeded
    assert program.stdout.strip() == "a b; echo hacked"
    assert piped.stdout == "HELLO"


@posix_only
def test_commands_start_in_the_workspace_and_relative_cwd_is_relative_to_it():
    with session() as sb:
        workspace = sb.workspace
        (workspace / "sub").mkdir()
        here = sb.run("pwd").stdout.strip()
        inside = sb.run("pwd", cwd="sub").stdout.strip()
        absolute = sb.run("pwd", cwd="/").stdout.strip()

    assert Path(here).resolve() == workspace and Path(inside).resolve() == workspace / "sub"
    assert absolute == "/"


@posix_only
def test_check_raises_with_the_result_attached():
    with session() as sb:
        with pytest.raises(SandboxCommandError) as raised:
            sb.run("echo about to fail >&2; exit 4", check=True)
        slow = pytest.raises(SandboxCommandError)
        with slow as timed:
            sb.run("sleep 30", check=True, timeout=0.3)

    error = raised.value
    assert error.result.exit_code == 4 and "exited with status 4" in str(error) and "about to fail" in str(error)
    assert "timed out" in str(timed.value)


@posix_only
def test_check_names_the_limit_a_command_hit():
    with session(cpu_seconds=1) as sb:
        with pytest.raises(SandboxCommandError, match="hit its cpu limit"):
            sb.run([sys.executable, "-c", "while True: pass"], check=True, timeout=60)


@posix_only
def test_a_command_given_as_a_list_is_described_without_a_shell():
    with session() as sb:
        with pytest.raises(SandboxCommandError, match=r"'false'"):
            sb.run(["false"], check=True)


@posix_only
def test_the_session_timeout_is_the_default_and_a_call_can_override_it():
    with session(timeout=0.3) as sb:
        default = sb.run("sleep 30")
        overridden = sb.run("sleep 0.5; echo done", timeout=30)

    assert default.timed_out and overridden.stdout == "done\n"


@posix_only
def test_environment_is_layered_and_host_secrets_stay_out(monkeypatch):
    monkeypatch.setenv("ZETT_SESSION_SECRET", "hunter2")
    code = "import os;print(os.environ.get('ZETT_SESSION_SECRET'), os.environ.get('A'), os.environ.get('B'))"

    with session(env={"A": "session", "B": "session"}) as sb:
        result = sb.run([sys.executable, "-c", code], env={"B": "call"})

    assert result.stdout.split() == ["None", "session", "call"]


@posix_only
def test_temporary_files_go_to_a_scratch_directory_that_is_deleted():
    code = "import tempfile;print(tempfile.gettempdir())"

    with session() as sb:
        reported = sb.run([sys.executable, "-c", code]).stdout.strip()
        scratch = sb._scratch

    assert Path(reported).resolve() == scratch.resolve()
    assert not scratch.exists()


@posix_only
def test_limits_in_the_session_reach_the_command():
    with session(max_file_size="100KB") as sb:
        result = sb.run([sys.executable, "-c", "open('big','wb').write(b'x'*5_000_000)"])
        size = (sb.workspace / "big").stat().st_size

    assert not result.succeeded and size <= 100 * 1024
    assert result.resources.file_size_bytes == "rlimit"


def test_the_underlying_sandbox_and_capabilities_are_reachable():
    with session(memory="64MB", cpu_seconds=5) as sb:
        report = sb.capabilities()

        assert sb.sandbox.name == "passthrough"
        assert report.cpu_seconds == "rlimit"
        assert "open" in repr(sb) and "workspace" in repr(sb)
    assert "closed" in repr(sb)


# ------------------------------------------------------------ concurrency


@posix_only
def test_several_threads_can_run_commands_in_one_session():
    results: list[str] = []
    with session() as sb:

        def work(index: int) -> None:
            results.append(sb.run(f"echo {index}").stdout.strip())

        threads = [threading.Thread(target=work, args=(index,)) for index in range(6)]
        [thread.start() for thread in threads]
        [thread.join(20) for thread in threads]

    assert sorted(results) == [str(index) for index in range(6)]


@posix_only
async def test_the_blocking_api_works_from_code_that_is_inside_an_event_loop():
    with session() as sb:
        result = await asyncio.to_thread(sb.run, "echo from a thread")

    assert result.stdout == "from a thread\n"


@posix_only
def test_closing_while_a_command_runs_kills_it_and_tells_the_caller():
    sb = session(timeout=None).open()
    pidfile = sb.workspace / "pid"
    outcome: dict[str, BaseException] = {}

    def work() -> None:
        try:
            sb.run(f"echo $$ > {pidfile}; sleep 60")
        except BaseException as error:  # noqa: BLE001 - the test inspects what arrived
            outcome["error"] = error

    thread = threading.Thread(target=work)
    thread.start()
    for _ in range(200):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.02)
    pid = int(pidfile.read_text())
    started = time.monotonic()

    sb.close()
    thread.join(10)

    assert time.monotonic() - started < 8
    assert isinstance(outcome["error"], SandboxClosedError)
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@posix_only
def test_ctrl_c_while_waiting_stops_the_command():
    with session(timeout=None) as sb:
        pidfile = sb.workspace / "pid"
        threading.Timer(0.6, os.kill, (os.getpid(), signal.SIGINT)).start()

        with pytest.raises(KeyboardInterrupt):
            sb.run(f"echo $$ > {pidfile}; sleep 60")

        pid = int(pidfile.read_text())
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)  # the command did not outlive the interrupt


# ------------------------------------------------------------------- async


@posix_only
async def test_the_async_form_has_the_same_lifecycle():
    async with session(memory="64MB") as sb:
        root = sb._root
        await asyncio.to_thread((sb.workspace / "x.py").write_text, "print('async ok')")
        result = await sb.arun([sys.executable, "x.py"], check=True)
        with pytest.raises(SandboxCommandError):
            await sb.arun("exit 2", check=True)

    assert result.stdout.strip() == "async ok"
    assert not root.exists()


@posix_only
async def test_closing_an_async_session_cancels_runs_in_flight():
    sb = await session(timeout=None).__aenter__()
    task = asyncio.create_task(sb.arun("sleep 60"))
    await asyncio.sleep(0.5)

    await sb.__aexit__(None, None, None)

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not sb._root.exists()


async def test_the_async_form_refuses_a_closed_session():
    sb = session()

    with pytest.raises(SandboxClosedError):
        await sb.arun("true")


def test_creating_a_session_issues_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with session() as sb:
            assert sb.is_open
