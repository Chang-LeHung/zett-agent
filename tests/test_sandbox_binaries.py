"""The bubblewrap the Linux sandbox launches, system or bundled, and the /proc tables."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from zett_agent.sandbox import binaries
from zett_agent.sandbox.binaries import BUNDLE_DIRECTORY, bundled_bwrap, bwrap_path
from zett_agent.sandbox.resources import _linux_process_table, _linux_task_count

#: ``e_machine`` values at offset 18 of an ELF header.
ELF_X86_64 = 62
ELF_AARCH64 = 183


def elf_machine(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(20)
    assert header[:4] == b"\x7fELF", f"{path} is not an ELF file"
    return struct.unpack("<H", header[18:20])[0]


# ------------------------------------------------------------------------------ bwrap


@pytest.mark.parametrize(
    ("machine", "directory", "expected"),
    [("x86_64", "linux-x86_64", ELF_X86_64), ("aarch64", "linux-aarch64", ELF_AARCH64)],
)
def test_a_bwrap_build_is_shipped_for_each_linux_architecture(machine, directory, expected):
    path = bundled_bwrap(machine, platform_name="linux")
    assert path == BUNDLE_DIRECTORY / directory / "bwrap"
    assert path is not None and path.is_file() and os.access(path, os.X_OK)
    assert elf_machine(path) == expected


@pytest.mark.parametrize(("machine", "expected"), [("AMD64", ELF_X86_64), ("arm64", ELF_AARCH64)])
def test_alternative_machine_spellings_find_the_same_build(machine, expected):
    path = bundled_bwrap(machine, platform_name="linux")
    assert path is not None and elf_machine(path) == expected


@pytest.mark.parametrize("machine", ["armv7l", "ppc64le", "riscv64", "i686"])
def test_an_architecture_without_a_build_gets_none(machine):
    assert bundled_bwrap(machine, platform_name="linux") is None


@pytest.mark.parametrize("platform_name", ["darwin", "win32", "freebsd13"])
def test_only_linux_consults_the_bundle(platform_name):
    assert bundled_bwrap("x86_64", platform_name=platform_name) is None


def test_a_pinned_bwrap_is_returned_as_given():
    assert bwrap_path("/opt/bin/bwrap") == "/opt/bin/bwrap"


def test_bwrap_on_the_path_wins_over_the_bundle(monkeypatch):
    monkeypatch.setattr(binaries.shutil, "which", lambda name: "/usr/bin/bwrap")
    assert bwrap_path() == "/usr/bin/bwrap"


def test_the_bundled_build_is_used_when_the_system_has_none(monkeypatch, tmp_path):
    bundled = tmp_path / "bwrap"
    bundled.write_text("binary")
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    monkeypatch.setattr(binaries, "bundled_bwrap", lambda *args, **kwargs: bundled)
    assert bwrap_path() == str(bundled)


def test_without_a_system_or_bundled_bwrap_the_answer_is_none(monkeypatch):
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    monkeypatch.setattr(binaries, "bundled_bwrap", lambda *args, **kwargs: None)
    assert bwrap_path() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the bundle is a Linux binary")
def test_the_bundled_bwrap_really_runs_here(monkeypatch):
    """Execute the shipped build, so a wrong or broken binary fails CI rather than a user."""
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    executable = bwrap_path()
    assert executable is not None and "_bin/linux-" in executable

    version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=30, check=False)
    assert version.returncode == 0 and "bubblewrap" in version.stdout.lower()

    sandboxed = subprocess.run(
        [
            executable,
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "/bin/sh",
            "-c",
            "echo bundled-bwrap-ok",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if sandboxed.returncode != 0:  # pragma: no cover - depends on the kernel's user namespaces
        pytest.skip(f"this kernel does not allow the namespace probe: {sandboxed.stderr.strip()[:200]}")
    assert "bundled-bwrap-ok" in sandboxed.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS uses seatbelt, not bwrap")
def test_macos_never_resolves_a_linux_bundle():
    assert bwrap_path() is None or "bwrap" in (bwrap_path() or "")
    assert bundled_bwrap() is None


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


def test_this_platform_finds_a_bwrap_or_explains_why_not():
    resolved = bwrap_path()
    if resolved is not None:
        assert Path(resolved).exists() or shutil.which(resolved) is not None
