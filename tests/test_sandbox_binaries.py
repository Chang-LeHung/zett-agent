"""The programs the sandbox launches, and the bundled shell for a Linux system without one."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from zett_agent.sandbox import binaries
from zett_agent.sandbox.binaries import BUNDLE_DIRECTORY, bundled_busybox, shell_argv
from zett_agent.sandbox.errors import SandboxUnavailableError
from zett_agent.sandbox.policy import command_arguments
from zett_agent.sandbox.resources import _linux_process_table, _linux_task_count

#: ``e_machine`` values at offset 18 of an ELF header.
ELF_X86_64 = 62
ELF_AARCH64 = 183


def elf_machine(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(20)
    assert header[:4] == b"\x7fELF", f"{path} is not an ELF file"
    return struct.unpack("<H", header[18:20])[0]


@pytest.mark.parametrize(("machine", "directory"), [("x86_64", "linux-x86_64"), ("aarch64", "linux-aarch64")])
def test_a_bundled_busybox_is_shipped_for_each_linux_architecture(machine, directory):
    path = bundled_busybox(machine, platform_name="linux")
    assert path == BUNDLE_DIRECTORY / directory / "busybox"
    assert path is not None and path.is_file() and os.access(path, os.X_OK)


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("x86_64", ELF_X86_64), ("AMD64", ELF_X86_64), ("aarch64", ELF_AARCH64), ("arm64", ELF_AARCH64)],
)
def test_each_bundled_binary_matches_the_architecture_its_directory_names(machine, expected):
    path = bundled_busybox(machine, platform_name="linux")
    assert path is not None
    assert elf_machine(path) == expected


@pytest.mark.parametrize("machine", ["armv7l", "ppc64le", "riscv64", "i686"])
def test_an_architecture_without_a_bundle_gets_none(machine):
    assert bundled_busybox(machine, platform_name="linux") is None


@pytest.mark.parametrize("platform_name", ["darwin", "win32", "freebsd13"])
def test_only_linux_consults_the_bundle(platform_name):
    assert bundled_busybox("x86_64", platform_name=platform_name) is None


def test_this_machine_resolves_the_shell_on_its_path():
    system = shutil.which("sh")
    assert system is not None
    assert shell_argv() == (system,)


def test_an_explicit_shell_is_returned_as_given():
    assert shell_argv("/bin/bash") == ("/bin/bash",)
    assert shell_argv(("/usr/bin/env", "bash")) == ("/usr/bin/env", "bash")


def test_the_bundled_shell_is_used_when_the_system_has_none(monkeypatch, tmp_path):
    busybox = tmp_path / "busybox"
    busybox.write_text("#!/bin/sh\n")
    monkeypatch.setattr(binaries, "bundled_busybox", lambda *args, **kwargs: busybox)
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    assert shell_argv() == (str(busybox), "sh")


def test_without_a_shell_anywhere_the_error_says_what_to_install(monkeypatch):
    monkeypatch.setattr(binaries, "bundled_busybox", lambda *args, **kwargs: None)
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    with pytest.raises(SandboxUnavailableError, match="No shell is available"):
        shell_argv()


def test_a_command_string_uses_the_resolved_shell_and_a_sequence_does_not():
    assert command_arguments("echo hi | cat") == (*shell_argv(), "-c", "echo hi | cat")
    assert command_arguments("ls", shell="/bin/busybox-sh") == ("/bin/busybox-sh", "-c", "ls")
    assert command_arguments(["python3", "a b.py"]) == ("python3", "a b.py")


def test_the_resolved_shell_can_actually_run_a_command():
    completed = subprocess.run([*shell_argv(), "-c", "printf hi"], capture_output=True, timeout=30, check=False)
    assert completed.returncode == 0 and completed.stdout == b"hi"


# ------------------------------------------------------------------------------ /proc


def fake_proc(tmp_path: Path, processes: dict[int, tuple[int, int, int, int]]) -> Path:
    """Build a /proc tree: pid -> (uid, threads, parent, resident pages)."""
    root = tmp_path / "proc"
    root.mkdir()
    for pid, (uid, threads, parent, pages) in processes.items():
        directory = root / str(pid)
        directory.mkdir()
        (directory / "status").write_text(f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nThreads:\t{threads}\n")
        fields = [
            "S",
            str(parent),
            str(pid),
            "0",
            "-1",
            "4194560",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "20",
            "0",
            "1",
        ]
        fields += ["0"] * 4  # itrealvalue, starttime, vsize
        fields += [str(pages)]
        (directory / "stat").write_text(f"{pid} (python) {' '.join(fields)}\n")
    return root


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="/proc parsing is exercised where the code runs")
def test_linux_counts_only_this_users_threads(tmp_path):
    uid = os.getuid()
    root = fake_proc(
        tmp_path,
        {
            10: (uid, 3, 1, 100),
            11: (uid, 1, 10, 50),
            12: (uid + 1, 7, 1, 10),  # another user
        },
    )
    assert _linux_task_count(root) == 4


def test_linux_reads_the_process_tree_from_proc(tmp_path):
    root = fake_proc(tmp_path, {10: (0, 1, 1, 2), 11: (0, 1, 10, 3)})
    table = _linux_process_table(root)
    assert table is not None
    rows, children = table
    page_size = os.sysconf("SC_PAGE_SIZE")
    assert rows[10] == (1, 10, 2 * page_size // 1024)
    assert rows[11][:2] == (10, 11)
    assert sorted(children[1]) == [10] and children[10] == [11]


def test_a_missing_proc_tree_is_reported_as_unknown(tmp_path):
    assert _linux_task_count(tmp_path / "absent") is None
    assert _linux_process_table(tmp_path / "absent") is None
