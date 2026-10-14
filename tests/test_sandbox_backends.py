"""One contract, run against every isolating backend this machine can actually run.

The same cases run for Seatbelt on macOS and for bubblewrap on Linux, so a
difference in what the two enforce shows up as a failing case rather than a
surprise. A backend the machine cannot run is skipped, unless
``ZETT_REQUIRE_SANDBOX=1`` says the machine is supposed to have one: CI sets it,
so a missing bubblewrap fails the job instead of silently skipping every case.
"""

import asyncio
import os
import socket
import sys
import uuid
from pathlib import Path

import pytest

from zett_agent.sandbox.errors import SandboxUnavailableError
from zett_agent.sandbox.factory import SandboxBackend, create_sandbox, default_backend
from zett_agent.sandbox.policy import NetworkMode, SandboxPolicy

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


def make(backend, work, **policy) -> "tuple":
    sandbox = create_sandbox(SandboxPolicy.workspace_write(work, **policy), backend=backend)
    return sandbox


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
