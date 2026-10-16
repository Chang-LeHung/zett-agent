"""The macOS Seatbelt profile and the Linux bubblewrap command line, rendered without running either."""

import subprocess
import sys
from pathlib import Path

import pytest

from zett_agent.sandbox import linux, macos
from zett_agent.sandbox.errors import SandboxUnavailableError
from zett_agent.sandbox.linux import LinuxSandbox, build_bubblewrap_arguments
from zett_agent.sandbox.macos import SEATBELT_EXECUTABLE, MacOSSandbox, build_seatbelt_profile
from zett_agent.sandbox.policy import NetworkMode, ResolvedPolicy, SandboxPolicy


def resolved(**fields) -> ResolvedPolicy:
    defaults = {
        "writable": (),
        "denied_write": (),
        "protected_write": (),
        "denied_read": (),
        "network": NetworkMode.NONE,
    }
    return ResolvedPolicy(**{**defaults, **fields})


# ------------------------------------------------------------------ Seatbelt


def test_the_profile_denies_by_default_and_never_embeds_a_path():
    nasty = Path('/work/a") (allow default) ("b')

    profile, params = build_seatbelt_profile(resolved(writable=(nasty,)))

    assert profile.startswith("(version 1)\n(deny default)\n")
    assert "allow default" not in profile and "/work" not in profile
    assert params == {"WRITE0": str(nasty)}
    assert '(allow file-write* (subpath (param "WRITE0")))' in profile


def test_rules_are_ordered_so_that_denials_override_permissions():
    profile, params = build_seatbelt_profile(
        resolved(
            writable=(Path("/w"),),
            denied_write=(Path("/w/no"),),
            protected_write=(Path("/w/.git/hooks"),),
            denied_read=(Path("/h/.ssh"),),
        ),
        temp_directories=(Path("/t"),),
    )

    assert params == {
        "WRITE0": "/w",
        "WRITE1": "/t",
        "NOWRITE0": "/w/no",
        "NOWRITE1": "/w/.git/hooks",
        "NOREAD0": "/h/.ssh",
    }
    assert profile.index("(allow file-read*)") < profile.index('(deny file-read* (subpath (param "NOREAD0")))')
    assert profile.index('(allow file-write* (subpath (param "WRITE1")))') < profile.index(
        '(deny file-write* (subpath (param "NOWRITE0")))'
    )


def test_the_network_is_closed_unless_the_policy_opens_it():
    closed, _ = build_seatbelt_profile(resolved())
    opened, _ = build_seatbelt_profile(resolved(network=NetworkMode.FULL))

    assert "network" not in closed
    assert "(allow network*)" in opened and "com.apple.dnssd.service" in opened


def test_macos_wrap_passes_every_path_as_a_parameter(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(macos, "_user_temp_directory", lambda: tmp_path / "tmpdir")
    sandbox = MacOSSandbox(SandboxPolicy.workspace_write(work, protect_defaults=False))

    command = sandbox._wrap(("echo", "hi"), work, {"PATH": "/bin"})

    assert command.argv[:2] == (SEATBELT_EXECUTABLE, "-p")
    assert command.argv[-2:] == ("echo", "hi")
    parameters = [command.argv[i + 1] for i, value in enumerate(command.argv) if value == "-D"]
    assert f"WRITE0={work.resolve()}" in parameters
    assert command.cwd == work and dict(command.env) == {"PATH": "/bin"}


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS reports the per-user temporary directory")
def test_macos_reports_the_user_temp_directory():
    directory = macos._user_temp_directory()

    assert directory is not None and directory.is_absolute() and directory.is_dir()


def test_the_user_temp_directory_is_absent_when_libc_cannot_report_one(monkeypatch):
    class Libc:
        def __init__(self, result):
            self.result = result

        def confstr(self, name, buffer, size):
            return self.result

    monkeypatch.setattr(macos.ctypes, "CDLL", lambda name: Libc(0))
    assert macos._user_temp_directory() is None

    def broken(name):
        raise OSError(name)

    monkeypatch.setattr(macos.ctypes, "CDLL", broken)
    assert macos._user_temp_directory() is None


def completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, "", stderr)


def test_macos_check_reports_a_missing_launcher(monkeypatch):
    monkeypatch.setattr(macos.os, "access", lambda path, mode: False)

    with pytest.raises(SandboxUnavailableError, match="needs macOS"):
        MacOSSandbox().check()


def test_macos_check_explains_a_nested_sandbox_and_remembers_success(monkeypatch):
    calls = []
    monkeypatch.setattr(macos.os, "access", lambda path, mode: True)

    def run(arguments, **kwargs):
        calls.append(arguments)
        return completed(1, "sandbox_apply: Operation not permitted")

    monkeypatch.setattr(macos.subprocess, "run", run)
    with pytest.raises(SandboxUnavailableError, match="already inside a sandbox"):
        MacOSSandbox().check()

    monkeypatch.setattr(macos.subprocess, "run", lambda *a, **k: calls.append(a) or completed(0))
    sandbox = MacOSSandbox()
    sandbox.check()
    sandbox.check()
    assert len(calls) == 2  # the failing probe, then one successful probe that is not repeated


def test_macos_check_reports_a_launcher_that_cannot_start(monkeypatch):
    monkeypatch.setattr(macos.os, "access", lambda path, mode: True)

    def run(*args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(macos.subprocess, "run", run)

    with pytest.raises(SandboxUnavailableError, match="could not be started"):
        MacOSSandbox().check()


# ---------------------------------------------------------------- bubblewrap


def test_the_full_bubblewrap_command_line_for_a_typical_policy(tmp_path):
    work, secret = tmp_path / "work", tmp_path / "secrets"
    (work / ".git" / "hooks").mkdir(parents=True)
    secret.mkdir()
    (work / "token").write_text("x")
    policy = resolved(
        writable=(work,),
        protected_write=(work / ".git" / "hooks", work / ".bashrc"),
        denied_read=(secret, work / "token"),
    )

    arguments, warnings = build_bubblewrap_arguments(
        policy, ("python3", "build.py"), work, {"PATH": "/bin", "HOME": "/h"}, bwrap="/usr/bin/bwrap"
    )

    assert arguments == [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user-try",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--unshare-net",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", str(work), str(work),
        "--ro-bind", str(work / ".git" / "hooks"), str(work / ".git" / "hooks"),
        "--tmpfs", str(secret), "--remount-ro", str(secret),
        "--ro-bind", "/dev/null", str(work / "token"),
        "--clearenv",
        "--setenv", "PATH", "/bin",
        "--setenv", "HOME", "/h",
        "--setenv", "TMPDIR", "/tmp",
        "--chdir", str(work),
        "--",
        "python3", "build.py",
    ]  # fmt: skip
    assert warnings == []  # the missing .bashrc is a default protection, so it is not worth a warning


def test_the_network_namespace_is_kept_only_when_the_policy_opens_the_network(tmp_path):
    closed, _ = build_bubblewrap_arguments(resolved(), ("true",), tmp_path, {})
    opened, _ = build_bubblewrap_arguments(resolved(network=NetworkMode.FULL), ("true",), tmp_path, {})

    assert "--unshare-net" in closed and "--unshare-net" not in opened
    assert "--share-net" not in opened


def test_unwritable_paths_follow_symlinks_and_skip_what_is_outside_a_writable_path(tmp_path):
    work, other = tmp_path / "work", tmp_path / "other"
    work.mkdir()
    other.mkdir()
    (work / "real").write_text("x")
    (work / "link").symlink_to(work / "real")

    arguments, _ = build_bubblewrap_arguments(
        resolved(writable=(work,), protected_write=(work / "link", work / "real", other)), ("true",), tmp_path, {}
    )

    assert arguments.count("--ro-bind") == 2  # the root, and the one real file once
    assert ["--ro-bind", str(work / "real"), str(work / "real")] == arguments[
        arguments.index(str(work / "real")) - 1 :
    ][:3]
    assert str(other) not in arguments


def test_explicit_denials_that_cannot_be_mounted_are_reported(tmp_path):
    work = tmp_path / "work"
    work.mkdir()

    _, warnings = build_bubblewrap_arguments(
        resolved(
            writable=(work,),
            denied_write=(work / "gone",),
            protected_write=(work / "gone",),
            denied_read=(tmp_path / "nope",),
        ),
        ("true",),
        tmp_path,
        {},
    )

    assert any("denied_write_paths entry does not exist" in warning for warning in warnings)
    assert any("denied_read_paths entry does not exist" in warning for warning in warnings)


def test_a_denied_read_symlink_is_masked_as_a_file_not_a_directory(tmp_path):
    target = tmp_path / "dir"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)

    arguments, _ = build_bubblewrap_arguments(resolved(denied_read=(link,)), ("true",), tmp_path, {})

    assert ["--ro-bind", "/dev/null", str(link)] == arguments[arguments.index("/dev/null") - 1 :][:3]


def test_the_sandbox_temp_directory_always_wins_over_a_host_tmpdir(tmp_path):
    arguments, _ = build_bubblewrap_arguments(resolved(), ("true",), tmp_path, {"TMPDIR": "/var/tmp/host"})

    assert "/var/tmp/host" not in arguments
    index = arguments.index("TMPDIR")
    assert arguments[index + 1] == "/tmp"


def test_linux_wrap_builds_a_launchable_command(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    sandbox = LinuxSandbox(SandboxPolicy.workspace_write(work, protect_defaults=False))
    sandbox._executable = "/usr/bin/bwrap"

    command = sandbox._wrap(("echo", "hi"), work, {"PATH": "/bin"})

    assert command.argv[0] == "/usr/bin/bwrap" and command.argv[-3:] == ("--", "echo", "hi")
    assert command.cwd == work and command.env["PATH"] == "/bin" and command.env["TMPDIR"] == "/tmp"


def test_linux_check_needs_linux(monkeypatch):
    monkeypatch.setattr(linux.sys, "platform", "darwin")

    with pytest.raises(SandboxUnavailableError, match="needs Linux"):
        LinuxSandbox().check()


def test_linux_check_reports_a_missing_binary(monkeypatch):
    monkeypatch.setattr(linux.sys, "platform", "linux")
    monkeypatch.setattr(linux.shutil, "which", lambda name: None)

    with pytest.raises(SandboxUnavailableError, match="apt install bubblewrap"):
        LinuxSandbox().check()


def test_linux_check_names_the_usual_causes_when_the_probe_fails(monkeypatch):
    monkeypatch.setattr(linux.sys, "platform", "linux")
    monkeypatch.setattr(linux.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(linux.os, "access", lambda path, mode: True)
    monkeypatch.setattr(
        linux.subprocess, "run", lambda *a, **k: completed(1, "bwrap: setting up uid map: Permission denied")
    )

    with pytest.raises(SandboxUnavailableError) as raised:
        LinuxSandbox().check()

    message = str(raised.value)
    assert "uid map" in message and "apparmor_restrict_unprivileged_userns" in message
    assert "Docker" in message and "WSL1" in message


def test_linux_check_reports_a_binary_that_cannot_start(monkeypatch):
    monkeypatch.setattr(linux.sys, "platform", "linux")
    monkeypatch.setattr(linux.os, "access", lambda path, mode: True)

    def run(*args, **kwargs):
        raise OSError("bad interpreter")

    monkeypatch.setattr(linux.subprocess, "run", run)

    with pytest.raises(SandboxUnavailableError, match="could not be started"):
        LinuxSandbox(bwrap_path="/opt/bwrap").check()


def test_linux_check_accepts_a_working_binary_and_probes_once(monkeypatch):
    probes = []
    monkeypatch.setattr(linux.sys, "platform", "linux")
    monkeypatch.setattr(linux.os, "access", lambda path, mode: True)
    monkeypatch.setattr(linux.subprocess, "run", lambda arguments, **k: probes.append(arguments) or completed(0))
    sandbox = LinuxSandbox(bwrap_path="/opt/bwrap")

    sandbox.check()
    sandbox.check()

    assert len(probes) == 1 and probes[0][0] == "/opt/bwrap" and probes[0][-1] == "true"
    assert sandbox._executable == "/opt/bwrap"
