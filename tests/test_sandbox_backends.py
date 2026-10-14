"""One contract, run against every isolating backend this machine can actually run.

The same cases run for Seatbelt on macOS and for bubblewrap on Linux, so a
difference in what the two enforce shows up as a failing case rather than a
surprise. A backend the machine cannot run is skipped, unless
``ZETT_REQUIRE_SANDBOX=1`` says the machine is supposed to have one: CI sets it,
so a missing bubblewrap fails the job instead of silently skipping every case.
"""

import asyncio
import os
import signal
import socket
import sys
import uuid
from pathlib import Path

import pytest

from zett_agent.sandbox.errors import SandboxUnavailableError
from zett_agent.sandbox.factory import SandboxBackend, create_sandbox, default_backend
from zett_agent.sandbox.policy import NetworkMode, ResourceLimits, SandboxPolicy
from zett_agent.sandbox.resources import MemoryStrategy, watchdog_available

REQUIRED = os.environ.get("ZETT_REQUIRE_SANDBOX") == "1"
posix_only = pytest.mark.skipif(os.name != "posix", reason="needs a POSIX shell")


def usable(backend: SandboxBackend) -> str | None:
    """Return why a backend cannot run here, or None when it can."""
    try:
        create_sandbox(backend=backend)
    except SandboxUnavailableError as error:
        return str(error)
    return None


BACKENDS = [SandboxBackend.SEATBELT, SandboxBackend.BUBBLEWRAP]
UNUSABLE = {backend: usable(backend) for backend in BACKENDS}


def test_the_platforms_own_backend_is_present_when_ci_requires_one():
    own = default_backend()
    if not REQUIRED or own is SandboxBackend.PASSTHROUGH:
        pytest.skip("no isolating backend is required on this machine")
    assert UNUSABLE[own] is None, f"ZETT_REQUIRE_SANDBOX=1 but {own.value} is unavailable: {UNUSABLE[own]}"


@pytest.fixture(params=BACKENDS, ids=[backend.value for backend in BACKENDS])
def backend(request) -> SandboxBackend:
    reason = UNUSABLE[request.param]
    if reason is not None:
        if REQUIRED and request.param is default_backend():
            pytest.fail(reason)
        pytest.skip(reason)
    return request.param


@pytest.fixture
def work(tmp_path) -> Path:
    directory = tmp_path / "work"
    directory.mkdir()
    return directory.resolve()


@pytest.fixture
def outside():
    """A path outside every writable path and outside the temporary directories, removed afterwards."""
    path = Path("/var/tmp") / f"zett-sandbox-test-{uuid.uuid4().hex}"
    yield path
    if path.is_dir():
        path.rmdir()
    elif path.exists():
        path.unlink()


def make(backend, work, *, limits: ResourceLimits | None = None, strategy: MemoryStrategy | None = None, **policy):
    """Build a sandbox for one case, with the limits and the memory mechanism it asks for."""
    sandbox = create_sandbox(
        SandboxPolicy.workspace_write(work, limits=limits or ResourceLimits(), **policy), backend=backend
    )
    if strategy is not None:
        sandbox.memory_strategy = strategy
    return sandbox


def ended_by(result, number: int) -> bool:
    """Whether a signal ended the command: ``-N`` when run directly, ``128 + N`` under bubblewrap's init."""
    return result.exit_code in (-number, 128 + number)


def test_every_backend_reports_how_it_enforces_each_limit(backend, work, capsys):
    """Print what this machine does with each limit; the sandbox job in CI reads it back."""
    limits = ResourceLimits(
        memory_bytes=256 * 1024**2, cpu_seconds=30, cpu_percent=50, processes=64, file_size_bytes=10**6
    )

    report = make(backend, work, limits=limits).capabilities()

    with capsys.disabled():
        print(f"\n{sys.platform} {backend.value}: {report.mechanisms}")
    assert report.cpu_seconds == "rlimit" and report.core_dumps == "rlimit"
    assert report.file_size_bytes == "rlimit"
    assert report.memory in ("cgroup", "rlimit_as", "watchdog", "none")
    assert report.processes in ("cgroup", "rlimit")
    assert report.cpu_percent in ("cgroup", "none")
    if report.memory == "none":
        assert report.warnings  # nothing enforced it, and that is said out loud


@posix_only
async def test_a_file_size_limit_holds_inside_the_sandbox(backend, work):
    sandbox = make(backend, work, limits=ResourceLimits(file_size_bytes=200_000, open_files=64))

    result = await sandbox.run(
        [sys.executable, "-c", "f=open('big.bin','wb')\n[f.write(b'x'*10_000) for _ in range(50)]"], cwd=work
    )

    assert not result.succeeded
    assert (work / "big.bin").stat().st_size <= 200_000
    assert result.resources.file_size_bytes == "rlimit"


@posix_only
async def test_a_processor_time_limit_holds_inside_the_sandbox(backend, work):
    result = await make(backend, work, limits=ResourceLimits(cpu_seconds=1)).run(
        [sys.executable, "-c", "x=0\nwhile True: x+=1"], cwd=work, timeout=60
    )

    assert result.timed_out is False and ended_by(result, signal.SIGXCPU)
    assert result.resources.limits_hit == ("cpu",)


@posix_only
async def test_a_process_limit_holds_inside_the_sandbox(backend, work):
    program = (
        "import os, time\n"
        "n = 0\n"
        "try:\n"
        "    for _ in range(300):\n"
        "        if os.fork() == 0:\n"
        "            time.sleep(20)\n"
        "            os._exit(0)\n"
        "        n += 1\n"
        "except OSError:\n"
        "    pass\n"
        "print('forked', n)\n"
    )
    result = await make(backend, work, limits=ResourceLimits(processes=32)).run(
        [sys.executable, "-c", program], cwd=work, timeout=60
    )

    # The command started (the limit is headroom above what the user already runs, so it
    # is not refused outright) and was stopped well before 300. The bounds are loose
    # because other processes of the same user count against the same allowance.
    assert result.succeeded, result.stderr
    forked = int(result.stdout.split()[-1])
    assert 1 <= forked <= 40


@posix_only
async def test_an_open_file_limit_holds_inside_the_sandbox(backend, work):
    result = await make(backend, work, limits=ResourceLimits(open_files=16)).run(
        [sys.executable, "-c", "f=[open('/dev/null') for _ in range(200)]\nprint('opened 200')"], cwd=work
    )

    assert not result.succeeded and "opened 200" not in result.stdout


@posix_only
async def test_core_dumps_are_off_inside_the_sandbox(backend, work):
    result = await make(backend, work).run(
        [sys.executable, "-c", "import resource;print(resource.getrlimit(resource.RLIMIT_CORE))"], cwd=work
    )

    assert result.stdout.strip() == "(0, 0)"


@posix_only
async def test_a_memory_limit_is_enforced_by_whatever_this_machine_has(backend, work):
    sandbox = make(
        backend,
        work,
        limits=ResourceLimits(memory_bytes=128 * 1024**2),
    )
    program = "x=bytearray(600*1024*1024)\nprint('holding')\nimport time; time.sleep(20)"
    result = await sandbox.run([sys.executable, "-c", program], cwd=work, timeout=60)
    mechanism = result.resources.memory

    if mechanism == "watchdog":
        assert result.exit_code == -signal.SIGKILL and result.resources.limits_hit == ("memory",)
        assert result.resources.peak_memory_bytes > 128 * 1024**2
    elif mechanism == "rlimit_as":
        assert not result.succeeded and "holding" not in result.stdout
    elif mechanism == "cgroup":
        assert not result.succeeded and result.resources.limits_hit == ("memory",)
    else:  # pragma: no cover - only on a platform with no memory mechanism at all
        pytest.fail(f"no memory limit was enforced: {result.resources.warnings}")


@posix_only
async def test_a_watchdog_stops_a_runaway_inside_the_sandbox(backend, work):
    """The watchdog is the fallback, so it is asked for explicitly wherever it can run."""
    if not watchdog_available():  # pragma: no cover - POSIX only
        pytest.skip("no ps to sample with")
    sandbox = make(
        backend,
        work,
        limits=ResourceLimits(memory_bytes=128 * 1024**2),
    )
    sandbox.memory_strategy = MemoryStrategy.WATCHDOG
    program = "x=bytearray(600*1024*1024)\nimport time; time.sleep(20)"

    result = await sandbox.run([sys.executable, "-c", program], cwd=work, timeout=60)

    assert result.resources.memory == "watchdog" and result.resources.limits_hit == ("memory",)
    assert result.duration < 15


@posix_only
async def test_limits_travel_with_a_command_a_host_spawns_itself(backend, work):
    command = make(backend, work, limits=ResourceLimits(file_size_bytes=1000)).wrap(
        [sys.executable, "-c", "import resource;print(resource.getrlimit(resource.RLIMIT_FSIZE))"], cwd=work
    )

    process = await asyncio.create_subprocess_exec(
        *command.argv,
        cwd=command.cwd,
        env=dict(command.env),
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
        preexec_fn=command.preexec,
    )
    out, _ = await process.communicate()

    assert out.decode().strip() == "(1000, 1000)"


async def test_the_backend_reports_that_it_isolates(backend, work):
    result = await make(backend, work).run("echo hello", cwd=work)

    assert result.succeeded and result.stdout == "hello\n"
    assert result.isolated is True and result.sandbox == backend.value


@posix_only
async def test_writes_succeed_inside_a_writable_path_and_land_on_the_host(backend, work):
    result = await make(backend, work).run("echo data > note.txt && cat note.txt", cwd=work)

    assert result.succeeded and result.stdout == "data\n"
    assert (work / "note.txt").read_text() == "data\n"


@posix_only
async def test_writes_outside_every_writable_path_are_refused(backend, work, outside):
    result = await make(backend, work).run(f"echo escaped > {outside}", cwd=work)

    assert not result.succeeded
    assert not outside.exists()


@posix_only
async def test_a_symlink_inside_the_workspace_cannot_carry_a_write_out(backend, work, outside):
    outside.mkdir()
    (work / "escape").symlink_to(outside)

    result = await make(backend, work).run("echo escaped > escape/file.txt", cwd=work)

    assert not result.succeeded
    assert not (outside / "file.txt").exists()


@posix_only
async def test_temporary_files_always_work(backend, work):
    result = await make(backend, work).run(
        'echo scratch > "$TMPDIR/zett-scratch" && cat "$TMPDIR/zett-scratch"', cwd=work
    )

    assert result.succeeded and result.stdout == "scratch\n"


@posix_only
async def test_protected_files_stay_unwritable_inside_a_writable_path(backend, work):
    (work / ".git" / "hooks").mkdir(parents=True)
    (work / ".git" / "config").write_text("[core]\n")
    (work / ".bashrc").write_text("# mine\n")
    sandbox = make(backend, work)

    hook = await sandbox.run("echo evil > .git/hooks/pre-commit", cwd=work)
    config = await sandbox.run("echo evil >> .git/config", cwd=work)
    shell = await sandbox.run("echo evil >> .bashrc", cwd=work)
    head = await sandbox.run("echo ref > .git/HEAD", cwd=work)

    assert not hook.succeeded and not (work / ".git" / "hooks" / "pre-commit").exists()
    assert not config.succeeded and (work / ".git" / "config").read_text() == "[core]\n"
    assert not shell.succeeded and (work / ".bashrc").read_text() == "# mine\n"
    assert head.succeeded  # the rest of .git stays writable


@posix_only
async def test_protection_can_be_switched_off(backend, work):
    (work / ".bashrc").write_text("# mine\n")

    result = await make(backend, work, protect_defaults=False).run("echo extra >> .bashrc", cwd=work)

    assert result.succeeded and (work / ".bashrc").read_text() == "# mine\nextra\n"


@posix_only
async def test_a_denied_write_path_beats_a_writable_parent(backend, work):
    (work / "frozen").mkdir()
    (work / "frozen" / "keep.txt").write_text("original")
    policy = SandboxPolicy(writable_paths=(work,), denied_write_paths=(work / "frozen",))
    sandbox = create_sandbox(policy, backend=backend)

    refused = await sandbox.run("echo changed > frozen/keep.txt", cwd=work)
    allowed = await sandbox.run("echo ok > free.txt", cwd=work)

    assert not refused.succeeded and (work / "frozen" / "keep.txt").read_text() == "original"
    assert allowed.succeeded


@posix_only
async def test_a_denied_read_path_cannot_be_read(backend, work):
    (work / "private").mkdir()
    (work / "private" / "secret.txt").write_text("the-secret-value")
    (work / "public.txt").write_text("fine")
    policy = SandboxPolicy(writable_paths=(work,), denied_read_paths=(work / "private",))
    sandbox = create_sandbox(policy, backend=backend)

    secret = await sandbox.run("cat private/secret.txt", cwd=work)
    public = await sandbox.run("cat public.txt", cwd=work)

    assert not secret.succeeded and "the-secret-value" not in secret.stdout + secret.stderr
    assert public.stdout == "fine"


@posix_only
async def test_a_denied_read_file_cannot_be_read(backend, work):
    (work / "token").write_text("the-token-value")
    policy = SandboxPolicy(writable_paths=(work,), denied_read_paths=(work / "token",))

    result = await create_sandbox(policy, backend=backend).run("cat token", cwd=work)

    assert "the-token-value" not in result.stdout + result.stderr


@posix_only
async def test_the_rest_of_the_machine_stays_readable(backend, work):
    result = await make(backend, work).run(
        f"cat {Path(sys.executable).resolve().parent}/../pyvenv.cfg 2>/dev/null; ls / >/dev/null", cwd=work
    )

    assert result.succeeded


@pytest.fixture
def listener():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    yield server.getsockname()[1]
    server.close()


def connect_program(port: int) -> list[str]:
    code = f"import socket; socket.create_connection(('127.0.0.1', {port}), 3); print('connected')"
    return [sys.executable, "-c", code]


async def test_the_network_is_closed_by_default(backend, work, listener):
    result = await make(backend, work).run(connect_program(listener), cwd=work, timeout=20)

    assert not result.succeeded and "connected" not in result.stdout


async def test_the_network_opens_when_the_policy_says_so(backend, work, listener):
    sandbox = make(backend, work, network=NetworkMode.FULL)

    result = await sandbox.run(connect_program(listener), cwd=work, timeout=20)

    assert result.succeeded and result.stdout.strip() == "connected"


async def test_host_secrets_in_the_environment_do_not_reach_the_process(backend, work, monkeypatch):
    monkeypatch.setenv("ZETT_HOST_SECRET", "hunter2")
    code = "import os; print(os.environ.get('ZETT_HOST_SECRET'), os.environ.get('ZETT_GIVEN'))"

    result = await make(backend, work).run([sys.executable, "-c", code], cwd=work, env={"ZETT_GIVEN": "given"})

    assert result.succeeded and result.stdout.split() == ["None", "given"]


@posix_only
async def test_the_whole_process_tree_stops_at_the_time_limit(backend, work):
    marker = work / "marker"

    result = await make(backend, work).run(f"(sleep 1; echo alive > {marker}) & wait", cwd=work, timeout=0.5)
    await asyncio.sleep(1.8)

    assert result.timed_out and not marker.exists()


@posix_only
async def test_a_program_the_agent_wrote_runs_inside_the_boundary(backend, work, outside):
    script = work / "agent_program.py"
    script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path('result.txt').write_text('computed')\n"
        f"try:\n    pathlib.Path({str(outside)!r}).write_text('escaped')\nexcept OSError:\n    print('blocked')\n"
    )

    result = await make(backend, work).run([sys.executable, "agent_program.py"], cwd=work)

    assert result.succeeded and result.stdout.strip() == "blocked"
    assert (work / "result.txt").read_text() == "computed"
    assert not outside.exists()


@posix_only
async def test_input_and_exit_codes_pass_through(backend, work):
    result = await make(backend, work).run("read line; echo got:$line; exit 7", cwd=work, stdin="hello\n")

    assert result.exit_code == 7 and result.stdout == "got:hello\n"


async def test_wrap_gives_a_host_a_command_it_can_spawn_itself(backend, work):
    command = make(backend, work).wrap([sys.executable, "-c", "print('spawned')"], cwd=work)

    process = await asyncio.create_subprocess_exec(
        *command.argv, cwd=command.cwd, env=dict(command.env), stdout=asyncio.subprocess.PIPE, start_new_session=True
    )
    out, _ = await process.communicate()

    assert out.decode().strip() == "spawned"


# --------------------------------------------------- the session API, for real


@posix_only
def test_a_session_confines_writes_and_destroys_its_workspace(backend, outside):
    from zett_agent.sandbox.session import SandboxSession

    with SandboxSession(backend=backend.value, memory="512MB", cpu_seconds=30) as sb:
        root = sb._root
        inside = sb.run("echo data > note.txt && cat note.txt", check=True)
        escaped = sb.run(f"echo escaped > {outside}")
        assert inside.stdout == "data\n" and (sb.workspace / "note.txt").read_text() == "data\n"
        assert not escaped.succeeded and not outside.exists()
        assert sb.sandbox.name == backend.value

    assert not root.exists()


@posix_only
def test_a_session_keeps_the_network_off_and_secrets_out(backend, listener, monkeypatch):
    from zett_agent.sandbox.session import SandboxSession

    monkeypatch.setenv("ZETT_HOST_SECRET", "hunter2")
    code = "import os;print(os.environ.get('ZETT_HOST_SECRET'))"
    with SandboxSession(backend=backend.value) as sb:
        offline = sb.run(connect_program(listener), timeout=20)
        secret = sb.run([sys.executable, "-c", code])
    with SandboxSession(backend=backend.value, network=True) as sb:
        online = sb.run(connect_program(listener), timeout=20)

    assert not offline.succeeded and "connected" not in offline.stdout
    assert secret.stdout.strip() == "None"
    assert online.succeeded and online.stdout.strip() == "connected"


@posix_only
def test_a_session_enforces_its_limits(backend):
    from zett_agent.sandbox.errors import SandboxCommandError
    from zett_agent.sandbox.session import SandboxSession

    with SandboxSession(backend=backend.value, cpu_seconds=1, max_file_size="100KB", timeout=60) as sb:
        spin = sb.run([sys.executable, "-c", "while True: pass"])
        grow = sb.run([sys.executable, "-c", "open('big','wb').write(b'x'*5_000_000)"])
        size = (sb.workspace / "big").stat().st_size
        with pytest.raises(SandboxCommandError, match="cpu limit"):
            sb.run([sys.executable, "-c", "while True: pass"], check=True)

    assert ended_by(spin, signal.SIGXCPU) and spin.resources.limits_hit == ("cpu",)
    assert not grow.succeeded and size <= 100 * 1024


@posix_only
def test_closing_a_session_stops_a_command_that_is_still_running(backend):
    import threading
    import time

    from zett_agent.sandbox.errors import SandboxClosedError
    from zett_agent.sandbox.session import SandboxSession

    sb = SandboxSession(backend=backend.value, timeout=None).open()
    root = sb._root
    caught: list[BaseException] = []

    def work() -> None:
        try:
            sb.run("sleep 60")
        except BaseException as error:  # noqa: BLE001 - inspected below
            caught.append(error)

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(1.0)
    started = time.monotonic()
    sb.close()
    thread.join(15)

    assert time.monotonic() - started < 10 and not root.exists()
    assert len(caught) == 1 and isinstance(caught[0], SandboxClosedError)
