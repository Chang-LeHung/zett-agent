"""Resource limits: validation, planning for a machine, and what the kernel and the watchdog enforce.

The end-to-end cases run through :class:`~zett_agent.sandbox.passthrough.PassthroughSandbox`,
which has no boundary of its own, so a limit that holds there is a limit the
kernel is enforcing rather than the sandbox's mounts.
"""

import asyncio
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from zett_agent.sandbox import resources as resources_module
from zett_agent.sandbox.base import SandboxResult
from zett_agent.sandbox.errors import SandboxPolicyError
from zett_agent.sandbox.factory import create_sandbox
from zett_agent.sandbox.passthrough import PassthroughSandbox
from zett_agent.sandbox.policy import MINIMUM_MEMORY_BYTES, ResourceLimits, SandboxPolicy
from zett_agent.sandbox.resources import (
    CPU_PERIOD_MICROSECONDS,
    CgroupHandle,
    CgroupV2,
    MemoryStrategy,
    ResourcePlan,
    ResourceReport,
    apply_rlimits,
    cgroup2_mount,
    descendant_pids,
    own_cgroup_path,
    plan_resources,
    plan_rlimits,
    process_tree_rss_bytes,
    rlimit_support,
    user_task_count,
)
from zett_agent.sandbox.seatbelt import SeatbeltSandbox

posix_only = pytest.mark.skipif(os.name != "posix", reason="needs a POSIX kernel")
darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux")


def sandbox_with(limits: ResourceLimits, **kwargs) -> PassthroughSandbox:
    return PassthroughSandbox(SandboxPolicy(limits=limits), **kwargs)


# ------------------------------------------------------------ the declaration


@pytest.mark.parametrize(
    "limits",
    [
        {"cpu_seconds": 0},
        {"cpu_seconds": 1.5},
        {"cpu_seconds": True},
        {"cpu_percent": 0},
        {"processes": 7},
        {"file_size_bytes": 0},
        {"open_files": 15},
        {"memory_bytes": MINIMUM_MEMORY_BYTES - 1},
        {"core_dumps": "no"},
        {"memory_sample_seconds": 0},
        {"memory_sample_seconds": 120},
    ],
)
def test_limits_outside_their_range_are_rejected(limits):
    with pytest.raises(SandboxPolicyError):
        ResourceLimits(**limits)


def test_empty_limits_request_nothing_and_disable_core_dumps():
    limits = ResourceLimits()

    assert limits.requested() == ()
    assert limits.core_dumps is False


def test_requested_lists_what_was_asked_for():
    assert ResourceLimits(cpu_percent=50, processes=32).requested() == ("cpu_percent", "processes")


def test_limits_are_immutable_and_live_on_the_policy():
    limits = ResourceLimits(memory_bytes=64 * 1024**2)

    with pytest.raises(AttributeError):
        limits.memory_bytes = 1
    assert SandboxPolicy(limits=limits).limits is limits
    assert SandboxPolicy.workspace_write(Path.cwd(), limits=limits).limits is limits
    with pytest.raises(SandboxPolicyError, match="limits must be a ResourceLimits"):
        SandboxPolicy(limits={"memory_bytes": 1})


# --------------------------------------------------------------- the kernel


@posix_only
def test_the_kernel_reports_which_limits_it_accepts():
    supported = rlimit_support()

    assert {"cpu_seconds", "processes", "file_size_bytes", "open_files", "core_dumps"} <= supported
    # macOS has no address-space limit, which is why memory needs another mechanism there.
    assert ("memory_bytes" in supported) is sys.platform.startswith("linux")


@posix_only
def test_planning_sets_soft_and_hard_equal_so_a_process_cannot_raise_its_own_limit():
    steps, warnings = plan_rlimits(ResourceLimits(file_size_bytes=4096, open_files=64))

    by_name = {step.name: step for step in steps}
    assert by_name["file_size_bytes"].soft == by_name["file_size_bytes"].hard == 4096
    assert by_name["open_files"].soft == by_name["open_files"].hard == 64
    assert warnings == ()


@posix_only
def test_the_processor_time_hard_limit_is_a_second_above_the_soft_one():
    """With the two equal, Linux sends SIGKILL and skips SIGXCPU, so the run could not say it hit the limit."""
    steps, _ = plan_rlimits(ResourceLimits(cpu_seconds=30))

    cpu = next(step for step in steps if step.name == "cpu_seconds")
    assert (cpu.soft, cpu.hard) == (30, 31)


def test_the_grace_second_never_passes_the_machines_hard_limit(monkeypatch):
    monkeypatch.setattr(resources_module._resource, "getrlimit", lambda rid: (30, 30))

    steps, _ = plan_rlimits(ResourceLimits(cpu_seconds=30), supported=frozenset({"cpu_seconds"}))

    cpu = next(step for step in steps if step.name == "cpu_seconds")
    assert (cpu.soft, cpu.hard) == (30, 30)


def test_the_process_limit_is_headroom_above_what_the_user_already_runs():
    """RLIMIT_NPROC is compared with every process the user owns, so the request is added to that count."""
    supported = frozenset({"processes", "core_dumps"})
    steps, warnings = plan_rlimits(ResourceLimits(processes=32), supported=supported, user_tasks=500)

    processes = next(step for step in steps if step.name == "processes")
    assert processes.soft == processes.hard == 532
    assert warnings == ()


def test_an_unreadable_process_count_leaves_the_limit_unenforced_and_says_so(monkeypatch):
    monkeypatch.setattr(resources_module, "user_task_count", lambda: None)

    steps, warnings = plan_rlimits(ResourceLimits(processes=32), supported=frozenset({"processes"}))

    assert not any(step.name == "processes" for step in steps)
    assert any("current process count could not be read" in warning for warning in warnings)


@posix_only
def test_the_user_task_count_is_the_number_of_processes_this_user_owns():
    count = user_task_count()

    assert count is not None and count >= 1


def test_the_user_task_count_is_unknown_when_ps_cannot_run(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no ps")

    monkeypatch.setattr(resources_module.subprocess, "run", broken)
    assert user_task_count() is None

    monkeypatch.setattr(
        resources_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "bad")
    )
    assert user_task_count() is None


def test_threads_are_counted_on_linux_and_processes_elsewhere(monkeypatch):
    seen: list[list[str]] = []

    def record(command, **kwargs):
        seen.append(list(command))
        return subprocess.CompletedProcess(command, 0, "1\n2\n3\nx\n", "")

    monkeypatch.setattr(resources_module.subprocess, "run", record)
    monkeypatch.setattr(resources_module.sys, "platform", "linux")
    assert user_task_count() == 3
    monkeypatch.setattr(resources_module.sys, "platform", "darwin")
    assert user_task_count() == 3

    assert "-L" in seen[0] and "lwp=" in seen[0]
    assert "-L" not in seen[1] and "pid=" in seen[1]


def test_core_dumps_are_disabled_unless_asked_for():
    off, _ = plan_rlimits(ResourceLimits(), supported=frozenset({"core_dumps"}))
    on, _ = plan_rlimits(ResourceLimits(core_dumps=True), supported=frozenset({"core_dumps"}))

    assert [step.name for step in off] == ["core_dumps"] and off[0].soft == 0
    assert [step.name for step in on] == []


def test_a_limit_this_kernel_does_not_support_is_reported_not_dropped():
    steps, warnings = plan_rlimits(ResourceLimits(memory_bytes=64 * 1024**2), supported=frozenset())

    assert steps == ()
    assert any(warning.startswith("memory_bytes cannot be limited") for warning in warnings)


def test_a_request_above_the_hard_limit_is_lowered_and_reported(monkeypatch):
    monkeypatch.setattr(resources_module._resource, "getrlimit", lambda rid: (64, 128))

    steps, warnings = plan_rlimits(ResourceLimits(open_files=1000), supported=frozenset({"open_files"}))

    assert [(step.soft, step.hard) for step in steps if step.name == "open_files"] == [(128, 128)]
    assert any("above the hard limit 128" in warning for warning in warnings)


@posix_only
def test_applying_the_steps_sets_what_the_kernel_sees():
    """The steps run in the child, so they are applied in a forked child and read back there."""
    (step,) = plan_rlimits(ResourceLimits(cpu_seconds=7), supported=frozenset({"cpu_seconds"}))[0]
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child only reports
        os.close(read)
        apply_rlimits((step,))
        os.write(write, str(resource.getrlimit(resource.RLIMIT_CPU)).encode())
        os._exit(0)
    os.close(write)
    reported = os.read(read, 100).decode()
    os.close(read)
    os.waitpid(pid, 0)

    assert reported == "(7, 8)"  # the hard limit is a second above the soft one


@posix_only
def test_applying_a_step_the_kernel_refuses_does_not_stop_the_command():
    step = resources_module.RlimitStep(resource=-1, name="bogus", soft=1, hard=1)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child only reports
        os.close(read)
        apply_rlimits((step,))
        os.write(write, b"survived")
        os._exit(0)
    os.close(write)
    assert os.read(read, 100) == b"survived"
    os.close(read)
    os.waitpid(pid, 0)


# ------------------------------------------------------------------ planning


def test_a_limit_nothing_can_enforce_is_named_in_the_warnings(monkeypatch):
    monkeypatch.setattr(resources_module, "rlimit_support", lambda: frozenset())
    monkeypatch.setattr(resources_module, "watchdog_available", lambda: False)

    plan = plan_resources(ResourceLimits(memory_bytes=64 * 1024**2, cpu_percent=50, processes=32))

    assert plan.report.memory == "none" and plan.report.cpu_percent == "none"
    assert any("no cgroup, no address-space limit" in warning for warning in plan.warnings)
    assert any("cpu_percent needs cgroup v2" in warning for warning in plan.warnings)


def test_the_memory_strategy_decides_what_happens_to_memory(monkeypatch):
    monkeypatch.setattr(resources_module, "rlimit_support", lambda: frozenset({"memory_bytes", "core_dumps"}))
    limits = ResourceLimits(memory_bytes=64 * 1024**2)

    kernel = plan_resources(limits, strategy=MemoryStrategy.KERNEL)
    watchdog = plan_resources(limits, strategy=MemoryStrategy.WATCHDOG)
    none = plan_resources(limits, strategy=MemoryStrategy.NONE)

    assert kernel.report.memory == "rlimit_as" and not kernel.watchdog
    assert [step.name for step in kernel.rlimits] == ["memory_bytes", "core_dumps"]
    assert watchdog.report.memory == "watchdog" and watchdog.watchdog
    # Core dumps are a limit of their own, so they are still asked of the kernel.
    assert [step.name for step in watchdog.rlimits] == ["core_dumps"]
    assert any("overshot" in warning for warning in watchdog.warnings)
    assert none.report.memory == "none" and any("strategy is 'none'" in w for w in none.warnings)


def test_kernel_strategy_refuses_to_pretend_when_the_kernel_cannot(monkeypatch):
    monkeypatch.setattr(resources_module, "rlimit_support", lambda: frozenset())

    plan = plan_resources(ResourceLimits(memory_bytes=64 * 1024**2), strategy=MemoryStrategy.KERNEL)

    assert plan.report.memory == "none" and not plan.watchdog
    assert any("forbids the watchdog" in warning for warning in plan.warnings)


def test_an_unknown_strategy_is_a_policy_error():
    with pytest.raises(SandboxPolicyError, match="unknown memory strategy"):
        plan_resources(ResourceLimits(), strategy="maybe")


def test_sandbox_rejects_a_strategy_this_platform_cannot_run(monkeypatch):
    monkeypatch.setattr("zett_agent.sandbox.base.watchdog_available", lambda: False)

    with pytest.raises(SandboxPolicyError, match="needs a POSIX system"):
        PassthroughSandbox(memory_strategy=MemoryStrategy.WATCHDOG)
    with pytest.raises(SandboxPolicyError, match="unknown memory strategy"):
        PassthroughSandbox(memory_strategy="maybe")


def test_capabilities_reports_what_this_machine_would_do_without_changing_anything():
    policy = SandboxPolicy(
        limits=ResourceLimits(memory_bytes=64 * 1024**2, cpu_seconds=30, processes=32, file_size_bytes=10**6)
    )

    report = PassthroughSandbox(policy).capabilities()

    assert report.cpu_seconds == "rlimit" and report.file_size_bytes == "rlimit"
    assert report.mechanisms["processes"] in ("rlimit", "cgroup")
    assert report.peak_memory_bytes is None and report.limits_hit == ()
    if sys.platform.startswith("linux"):
        assert report.memory in ("cgroup", "rlimit_as")
    else:
        assert report.memory == "watchdog" and any("overshot" in w for w in report.warnings)


def test_a_report_names_the_mechanism_of_every_limit():
    report = ResourceReport(memory="cgroup", cpu_percent="cgroup", peak_memory_bytes=100, limits_hit=("memory",))

    assert report.mechanisms["memory"] == "cgroup" and report.mechanisms["open_files"] == "none"
    assert report.mechanisms["processes"] == "none" and report.peak_memory_bytes == 100


# ------------------------------------------------------------------- cgroups


def fake_cgroup_tree(tmp_path: Path, own: str = "/own", controllers: tuple[str, ...] = ("memory", "cpu", "pids")):
    """Build a directory that looks like a cgroup v2 hierarchy, and the two /proc files that describe it."""
    mount = tmp_path / "cgroup"
    (mount / own.lstrip("/")).mkdir(parents=True)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"30 1 0:26 {mount} {mount} rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup2 rw\n")
    proc_cgroup = tmp_path / "cgroup.file"
    proc_cgroup.write_text(f"0::{own}\n")
    return mount, mountinfo, proc_cgroup


def make_group(handle_path: Path, controllers: tuple[str, ...] = ("memory", "cpu", "pids")) -> CgroupHandle:
    handle_path.mkdir(parents=True, exist_ok=True)
    files = {
        "memory": ("memory.max", "memory.swap.max", "memory.oom.group", "memory.peak", "memory.events"),
        "cpu": ("cpu.max", "cpu.stat"),
        "pids": ("pids.max", "pids.peak"),
    }
    for controller in controllers:
        for name in files[controller]:
            (handle_path / name).write_text(
                "max 100000" if name == "cpu.max" else ("max\n" if name.endswith(".max") else "0")
            )
    (handle_path / "cgroup.procs").write_text("")
    return CgroupHandle(path=handle_path)


@pytest.fixture
def linux_platform(monkeypatch):
    """Make the platform checks in the module behave as they do on Linux."""
    monkeypatch.setattr(resources_module, "sys", SimpleNamespace(platform="linux"))
    return None


def test_the_cgroup_mount_and_the_own_group_are_read_from_proc(tmp_path, linux_platform):
    mount, mountinfo, proc_cgroup = fake_cgroup_tree(tmp_path, own="/user.slice/user-1000.slice/session-1.scope")

    found = cgroup2_mount(mountinfo)

    assert found == mount
    assert own_cgroup_path(found, proc_cgroup) == mount / "user.slice/user-1000.slice/session-1.scope"
    assert CgroupV2.discover(mount=mount, cgroup_path=proc_cgroup) == CgroupV2(
        directory=mount / "user.slice/user-1000.slice/session-1.scope"
    )


def test_a_machine_without_cgroup_v2_reports_nothing(tmp_path, linux_platform):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("30 1 0:26 /sys/fs/cgroup rw - cgroup cgroup rw\n")

    assert cgroup2_mount(mountinfo) is None
    assert cgroup2_mount(tmp_path / "missing") is None
    assert own_cgroup_path(Path("/nowhere"), tmp_path / "missing") is None


def test_delegation_is_probed_by_creating_and_removing_a_group(tmp_path):
    mount, mountinfo, proc_cgroup = fake_cgroup_tree(tmp_path)
    hierarchy = CgroupV2.discover(mount=mount, cgroup_path=proc_cgroup)

    assert hierarchy is not None and hierarchy.delegated() is True
    assert list((mount / "own").iterdir()) == []  # the probe was cleaned up
    assert hierarchy.delegated() is True  # and only probed once


def test_a_hierarchy_that_refuses_a_child_group_says_so(tmp_path, monkeypatch):
    mount, mountinfo, proc_cgroup = fake_cgroup_tree(tmp_path)
    hierarchy = CgroupV2.discover(mount=mount, cgroup_path=proc_cgroup)
    monkeypatch.setattr(resources_module.Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError()))

    assert hierarchy.delegated() is False and hierarchy.create("zett") is None


def test_the_cgroup_hierarchy_is_only_looked_for_on_linux(tmp_path):
    mount, mountinfo, _ = fake_cgroup_tree(tmp_path)

    found = cgroup2_mount(mountinfo)

    if sys.platform.startswith("linux"):
        assert found == mount
    else:
        assert found is None  # there is no cgroup v2 to look at on this platform


def test_one_group_is_created_per_run(tmp_path):
    mount, mountinfo, proc_cgroup = fake_cgroup_tree(tmp_path)
    hierarchy = CgroupV2.discover(mount=mount, cgroup_path=proc_cgroup)

    handle = hierarchy.create("zett-sandbox")

    assert handle is not None and handle.path.parent == mount / "own"
    assert handle.path.name.startswith("zett-sandbox-")
    assert handle.release() is True and not handle.path.exists()
    assert handle.release() is True  # releasing twice is not an error


def test_the_limits_are_written_to_the_control_files_of_the_group(tmp_path):
    handle = make_group(tmp_path / "group")

    assert handle.set_memory(512 * 1024**2) is True
    assert (handle.path / "memory.max").read_text() == str(512 * 1024**2)
    assert (handle.path / "memory.swap.max").read_text() == "0"
    assert (handle.path / "memory.oom.group").read_text() == "1"
    assert handle.set_cpu_percent(150) is True
    assert (handle.path / "cpu.max").read_text() == f"{150 * 1000} {CPU_PERIOD_MICROSECONDS}"
    assert handle.set_processes(64) is True
    assert (handle.path / "pids.max").read_text() == "64"
    assert handle.applied == {"memory": "cgroup", "cpu_percent": "cgroup", "processes": "cgroup"}


def test_a_controller_that_did_not_reach_the_group_is_reported_as_unavailable(tmp_path):
    handle = make_group(tmp_path / "group", controllers=())

    assert handle.set_memory(512 * 1024**2) is False
    assert handle.set_cpu_percent(50) is False
    assert handle.set_processes(64) is False
    assert handle.applied == {}
    assert not (handle.path / "memory.max").exists()  # nothing was invented


def test_the_child_joins_the_group_by_writing_its_own_pid(tmp_path):
    handle = make_group(tmp_path / "group")

    handle.join()

    assert (handle.path / "cgroup.procs").read_text() == "0"


def test_joining_a_group_that_is_gone_does_not_stop_the_child(tmp_path):
    handle = CgroupHandle(path=tmp_path / "missing")

    handle.join()  # nothing raised: the child has to reach exec


def test_usage_is_read_back_from_the_group(tmp_path):
    handle = make_group(tmp_path / "group")
    (handle.path / "memory.peak").write_text("268435456\n")
    (handle.path / "memory.events").write_text("low 0\nhigh 2\nmax 3\noom 1\noom_kill 1\n")
    (handle.path / "cpu.stat").write_text("usage_usec 123456\nuser_usec 100000\n")
    (handle.path / "pids.peak").write_text("7\n")

    usage = handle.usage()

    assert usage.peak_memory_bytes == 268435456
    assert usage.memory_events_max == 3 and usage.oom_kills == 1
    assert usage.cpu_usage_microseconds == 123456 and usage.peak_processes == 7


def test_a_group_that_reports_nothing_yields_nothing(tmp_path):
    handle = make_group(tmp_path / "group", controllers=())

    usage = handle.usage()

    assert usage.peak_memory_bytes is None and usage.oom_kills == 0 and usage.cpu_usage_microseconds is None


def test_a_group_is_not_created_when_only_asking_what_would_happen(tmp_path, monkeypatch):
    mount, mountinfo, proc_cgroup = fake_cgroup_tree(tmp_path)
    monkeypatch.setattr(resources_module, "cgroup2_mount", lambda *a, **k: mount)
    monkeypatch.setattr(resources_module, "own_cgroup_path", lambda *a, **k: mount / "own")
    limits = ResourceLimits(memory_bytes=256 * 1024**2, cpu_percent=50, processes=64)

    plan = plan_resources(limits, create_cgroup=False)

    assert plan.report.memory == "cgroup" and plan.report.cpu_percent == "cgroup"
    assert plan.report.processes == "cgroup"
    assert plan.cgroup is None and list((mount / "own").iterdir()) == []


def test_a_cgroup_replaces_the_rlimit_for_the_limits_it_takes_over(tmp_path, monkeypatch):
    handle = make_group(tmp_path / "group")
    monkeypatch.setattr(resources_module.CgroupV2, "create", lambda self, name: handle)
    monkeypatch.setattr(resources_module.CgroupV2, "delegated", lambda self: True)
    monkeypatch.setattr(resources_module, "cgroup2_mount", lambda *a, **k: tmp_path)
    monkeypatch.setattr(resources_module, "own_cgroup_path", lambda *a, **k: tmp_path / "own")
    limits = ResourceLimits(memory_bytes=256 * 1024**2, processes=64, cpu_seconds=30)

    plan = plan_resources(limits)

    assert plan.report.memory == "cgroup" and plan.report.processes == "cgroup"
    assert plan.report.cpu_seconds == "rlimit"  # a cgroup does not limit total processor time
    assert sorted(step.name for step in plan.rlimits) == ["core_dumps", "cpu_seconds"]
    assert plan.cgroup is handle and plan.watchdog is False


def test_a_group_is_measured_while_it_still_exists(tmp_path, monkeypatch):
    """The kernel's own numbers are read before the group is released, or they are gone with it."""
    handle = make_group(tmp_path / "group")
    (handle.path / "memory.peak").write_text("123456789\n")
    (handle.path / "memory.events").write_text("max 2\noom_kill 2\n")
    released: list[str] = []
    original = handle.release

    def release() -> bool:
        released.append(str(handle.path))
        return original()

    monkeypatch.setattr(handle, "release", release)
    monkeypatch.setattr(
        PassthroughSandbox,
        "_plan",
        lambda self, *, create_cgroup=True: ResourcePlan(report=ResourceReport(memory="cgroup"), cgroup=handle),
    )

    result = asyncio.run(PassthroughSandbox().run("true"))

    assert result.resources.peak_memory_bytes == 123456789
    assert result.resources.limits_hit == ("memory",)
    assert released == [str(handle.path)]  # released, but only after it was read


def test_a_group_without_the_memory_controller_falls_back(tmp_path, monkeypatch):
    handle = make_group(tmp_path / "group", controllers=("pids",))
    monkeypatch.setattr(resources_module.CgroupV2, "create", lambda self, name: handle)
    monkeypatch.setattr(resources_module.CgroupV2, "delegated", lambda self: True)
    monkeypatch.setattr(resources_module, "cgroup2_mount", lambda *a, **k: tmp_path)
    monkeypatch.setattr(resources_module, "own_cgroup_path", lambda *a, **k: tmp_path / "own")

    plan = plan_resources(
        ResourceLimits(memory_bytes=256 * 1024**2, cpu_percent=50, processes=64), strategy=MemoryStrategy.KERNEL
    )

    assert plan.report.memory in ("rlimit_as", "none") and plan.report.cpu_percent == "none"
    assert plan.report.processes == "cgroup"
    assert any("cpu_percent needs cgroup v2" in warning for warning in plan.warnings)


# ------------------------------------------------------------------- sampling


def ps_table(*rows: tuple[int, int, int, int]):
    """Build a fake ``ps -axo pid=,ppid=,pgid=,rss=`` that returns the given rows."""
    text = "".join(f"{pid} {ppid} {pgid} {rss}\n" for pid, ppid, pgid, rss in rows)
    return lambda command, **kwargs: subprocess.CompletedProcess(list(command), 0, text, "")


def test_the_memory_of_a_tree_is_summed_from_ps():
    measured = process_tree_rss_bytes(os.getpid())

    assert measured is not None and measured > 0
    assert measured % 1024 == 0  # ps reports kibibytes


def test_the_tree_is_the_descendants_and_the_process_group_together(monkeypatch):
    monkeypatch.setattr(
        resources_module.subprocess,
        "run",
        ps_table(
            (100, 1, 100, 10),  # the launcher, leader of its group
            (101, 100, 101, 20),  # a child in a session of its own (what bubblewrap does)
            (102, 101, 101, 40),  # and its child
            (103, 1, 100, 80),  # orphaned to init but still in the group
            (104, 1, 104, 5000),  # unrelated
            (105, 104, 104, 5000),  # unrelated
        ),
    )

    assert process_tree_rss_bytes(100) == (10 + 20 + 40 + 80) * 1024


def test_a_process_that_exited_leaves_only_what_is_still_in_its_group(monkeypatch):
    monkeypatch.setattr(resources_module.subprocess, "run", ps_table((200, 1, 100, 64), (300, 1, 300, 999)))

    assert process_tree_rss_bytes(100) == 64 * 1024
    assert process_tree_rss_bytes(999) == 0


def test_descendants_include_children_in_other_sessions_but_not_the_caller_or_strangers(monkeypatch):
    monkeypatch.setattr(
        resources_module.subprocess,
        "run",
        ps_table((100, 1, 100, 1), (101, 100, 101, 1), (102, 101, 101, 1), (103, 1, 100, 1), (104, 1, 104, 1)),
    )

    assert descendant_pids(100) == {101, 102, 103}


def test_no_descendants_are_reported_when_ps_cannot_be_read(monkeypatch):
    monkeypatch.setattr(
        resources_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "bad")
    )

    assert descendant_pids(100) == frozenset()


def test_a_cycle_in_the_table_cannot_make_the_walk_loop(monkeypatch):
    monkeypatch.setattr(resources_module.subprocess, "run", ps_table((100, 101, 100, 8), (101, 100, 101, 8)))

    assert process_tree_rss_bytes(100) == 16 * 1024


@posix_only
def test_a_grandchild_in_its_own_session_is_still_counted():
    """The case a process-group measurement misses: the sandbox starts the command in a new session."""
    program = (
        "import os, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; x=bytearray(150*1024*1024); time.sleep(30)'],"
        " start_new_session=True)\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", program], start_new_session=True)
    try:
        measured = 0
        for _ in range(100):
            time.sleep(0.1)
            measured = process_tree_rss_bytes(child.pid) or 0
            if measured > 120 * 1024**2:
                break
        assert measured > 120 * 1024**2
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def test_nothing_can_be_measured_when_ps_cannot_run(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no ps")

    monkeypatch.setattr(resources_module.subprocess, "run", broken)
    assert process_tree_rss_bytes(os.getpid()) is None

    monkeypatch.setattr(
        resources_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "bad")
    )
    assert process_tree_rss_bytes(os.getpid()) is None


def test_unparsable_rows_are_ignored(monkeypatch):
    def noisy(command, **kwargs):
        text = f"   \n{os.getpid()} 1 {os.getpid()} 1024\nnonsense\n1 2 3 x\n1 2 3\n"
        return subprocess.CompletedProcess([], 0, text, "")

    monkeypatch.setattr(resources_module.subprocess, "run", noisy)

    assert process_tree_rss_bytes(os.getpid()) == 1024 * 1024


# --------------------------------------------------------- limits in a run


@posix_only
async def test_a_file_size_limit_stops_a_file_from_growing(tmp_path):
    sandbox = sandbox_with(ResourceLimits(file_size_bytes=100_000, open_files=64))

    result = await sandbox.run(
        [sys.executable, "-c", "f=open('big.bin','wb')\n[f.write(b'x'*10_000) for _ in range(50)]"],
        cwd=tmp_path,
    )

    assert not result.succeeded
    assert result.resources.file_size_bytes == "rlimit"
    assert (tmp_path / "big.bin").stat().st_size <= 100_000


@posix_only
async def test_a_processor_time_limit_kills_a_busy_loop(tmp_path):
    sandbox = sandbox_with(ResourceLimits(cpu_seconds=1))

    result = await sandbox.run([sys.executable, "-c", "x=0\nwhile True: x+=1"], cwd=tmp_path, timeout=60)

    assert result.timed_out is False  # the limit ended it, not the wall clock
    assert result.exit_code == -signal.SIGXCPU  # SIGXCPU, not SIGKILL: the hard limit is a second higher
    assert result.resources.limits_hit == ("cpu",)
    assert result.resources.cpu_seconds == "rlimit"


@posix_only
async def test_processor_time_does_not_count_sleeping(tmp_path):
    sandbox = sandbox_with(ResourceLimits(cpu_seconds=30))

    result = await sandbox.run(
        [sys.executable, "-c", "import time\ntime.sleep(0.1)\nprint('slept')"], cwd=tmp_path, timeout=60
    )

    assert result.succeeded and result.stdout.strip() == "slept"
    assert result.resources.limits_hit == ()


@posix_only
async def test_a_process_limit_stops_a_fork_bomb_after_its_headroom(tmp_path):
    sandbox = sandbox_with(ResourceLimits(processes=32))
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

    result = await sandbox.run([sys.executable, "-c", program], cwd=tmp_path, timeout=60)

    forked = int(result.stdout.split()[-1])
    # It could add some processes and was stopped well short of the 300 it tried. The
    # allowance is "what the user ran when the limit was planned, plus 32", so it grows
    # when other processes of this user exit in the meantime (a CI runner does this
    # constantly: one run got 41). An exact bound would be a flaky assertion.
    assert 1 <= forked < 200
    assert result.resources.processes == "rlimit"


@posix_only
async def test_an_open_file_limit_stops_a_program_from_holding_more(tmp_path):
    sandbox = sandbox_with(ResourceLimits(open_files=16))

    result = await sandbox.run(
        [sys.executable, "-c", "f=[open('/dev/null') for _ in range(200)]\nprint('opened 200')"], cwd=tmp_path
    )

    assert not result.succeeded and "opened 200" not in result.stdout
    assert result.resources.open_files == "rlimit"


@posix_only
async def test_core_dumps_are_off_by_default(tmp_path):
    sandbox = sandbox_with(ResourceLimits())

    result = await sandbox.run(
        [sys.executable, "-c", "import resource;print(resource.getrlimit(resource.RLIMIT_CORE))"], cwd=tmp_path
    )

    assert result.stdout.strip() == "(0, 0)" and result.resources.core_dumps == "rlimit"


@posix_only
async def test_the_watchdog_stops_a_program_that_holds_too_much_memory(tmp_path):
    sandbox = sandbox_with(ResourceLimits(memory_bytes=128 * 1024**2), memory_strategy=MemoryStrategy.WATCHDOG)
    program = "x=bytearray(600*1024*1024)\nprint('holding', len(x))\nimport time; time.sleep(20)\n"

    result = await sandbox.run([sys.executable, "-c", program], cwd=tmp_path, timeout=60)

    assert result.exit_code == -signal.SIGKILL
    assert result.resources.limits_hit == ("memory",)
    assert result.resources.memory == "watchdog"
    assert result.resources.peak_memory_bytes > 128 * 1024**2
    assert result.duration < 10  # stopped as soon as it went over, not at the timeout


@posix_only
async def test_a_program_within_its_memory_limit_is_left_alone(tmp_path):
    sandbox = sandbox_with(ResourceLimits(memory_bytes=512 * 1024**2), memory_strategy=MemoryStrategy.WATCHDOG)
    program = "x=bytearray(32*1024*1024)\nprint('ok', len(x))\nimport time; time.sleep(0.1)"

    result = await sandbox.run([sys.executable, "-c", program], cwd=tmp_path, timeout=60)

    assert result.succeeded and result.stdout.startswith("ok")
    assert result.resources.limits_hit == ()
    assert result.resources.peak_memory_bytes is not None


@posix_only
async def test_a_watchdog_that_cannot_measure_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr("zett_agent.sandbox.base.process_tree_rss_bytes", lambda pid: None)
    sandbox = sandbox_with(ResourceLimits(memory_bytes=64 * 1024**2), memory_strategy=MemoryStrategy.WATCHDOG)

    result = await sandbox.run([sys.executable, "-c", "print('ran')"], cwd=tmp_path, timeout=30)

    assert result.succeeded and result.resources.limits_hit == ()
    assert any("could not read the command's memory" in warning for warning in result.resources.warnings)


@posix_only
async def test_a_kernel_memory_limit_refuses_the_allocation(tmp_path):
    sandbox = sandbox_with(ResourceLimits(memory_bytes=64 * 1024**2), memory_strategy=MemoryStrategy.KERNEL)

    result = await sandbox.run(
        [sys.executable, "-c", "x=bytearray(600*1024*1024)\nprint('allocated')"], cwd=tmp_path, timeout=60
    )

    if result.resources.memory == "rlimit_as":
        assert not result.succeeded and "MemoryError" in result.stderr
    else:
        # macOS has no address-space limit, so the strategy asked for a kernel
        # limit that does not exist and nothing enforced this.
        assert result.resources.memory == "none" and result.succeeded
        assert any("forbids the watchdog" in warning for warning in result.resources.warnings)


async def test_a_run_without_limits_reports_nothing_enforced(tmp_path):
    sandbox = SandboxPolicy(limits=ResourceLimits())
    result = await sandbox_with(ResourceLimits()).run([sys.executable, "-c", "print('plain')"], cwd=tmp_path)

    assert result.succeeded and result.resources.limits_hit == () and result.resources.peak_memory_bytes is None
    assert result.resources.memory == "none" and result.resources.core_dumps == "rlimit"
    assert sandbox.limits.requested() == ()


def test_a_result_carries_its_report():
    report = ResourceReport(memory="watchdog", peak_memory_bytes=1, limits_hit=("memory",))
    result = SandboxResult(0, "", "", resources=report)

    assert result.resources is report and result.resources.mechanisms["memory"] == "watchdog"


# ------------------------------------------------------ limits through wrap()


def test_wrap_hands_the_kernel_limits_to_the_caller(tmp_path):
    sandbox = sandbox_with(ResourceLimits(file_size_bytes=1000, cpu_seconds=5))

    command = sandbox.wrap("echo hi", cwd=tmp_path)

    assert command.limits.file_size_bytes == 1000
    assert command.preexec is not None
    # Only run() can sample memory, so wrap() says which limits it did not apply.
    assert not any("only run() enforces" in warning for warning in command.warnings)


def test_wrap_says_when_a_limit_needs_the_parent(tmp_path):
    sandbox = sandbox_with(ResourceLimits(memory_bytes=64 * 1024**2), memory_strategy=MemoryStrategy.WATCHDOG)

    command = sandbox.wrap("echo hi", cwd=tmp_path)

    assert any("only run() enforces" in warning and "memory" in warning for warning in command.warnings)
    assert any("overshot" in warning for warning in command.warnings)


@posix_only
async def test_a_host_can_apply_the_limits_itself(tmp_path):
    sandbox = sandbox_with(ResourceLimits(file_size_bytes=1000))
    command = sandbox.wrap(
        [sys.executable, "-c", "import resource;print(resource.getrlimit(resource.RLIMIT_FSIZE))"], cwd=tmp_path
    )

    process = await asyncio.create_subprocess_exec(
        *command.argv,
        cwd=command.cwd,
        env=dict(command.env),
        stdout=asyncio.subprocess.PIPE,
        preexec_fn=command.preexec,
    )
    out, _ = await process.communicate()

    assert out.decode().strip() == "(1000, 1000)"


def test_capabilities_and_a_run_agree_on_the_mechanism(tmp_path):
    sandbox = sandbox_with(ResourceLimits(memory_bytes=64 * 1024**2, file_size_bytes=10**6))

    planned = sandbox.capabilities()

    assert planned.memory == sandbox._plan().report.memory
    assert planned.file_size_bytes == "rlimit"


def test_the_sandbox_reports_the_same_mechanisms_through_the_factory(tmp_path):
    policy = SandboxPolicy(limits=ResourceLimits(cpu_seconds=10))

    sandbox = create_sandbox(policy, backend="passthrough")

    assert isinstance(sandbox, PassthroughSandbox)
    assert sandbox.capabilities().cpu_seconds == "rlimit"


@darwin_only
def test_macos_has_no_kernel_memory_limit_and_says_so():
    report = SeatbeltSandbox(SandboxPolicy(limits=ResourceLimits(memory_bytes=64 * 1024**2))).capabilities()

    assert report.memory == "watchdog"
    assert any("overshot" in warning for warning in report.warnings)
