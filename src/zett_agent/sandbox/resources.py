"""How a :class:`~zett_agent.sandbox.policy.ResourceLimits` is enforced on a machine.

Three mechanisms exist, and a machine uses the strongest one it has for each
limit:

``cgroup``
    A Linux control group gives exact, tree-wide accounting: the kernel charges
    every process the sandbox starts to one group, kills it when it goes over
    ``memory.max``, denies a fork past ``pids.max``, and gives it a share of a
    processor through ``cpu.max``. It needs a cgroup v2 group that this user may
    create children in, with the controllers enabled for them. The kernel enables
    controllers for children only in a group that holds no processes itself, so a
    group that a login shell or a service runs in does not qualify; it takes
    delegation, such as systemd's ``Delegate=yes``, and a layout where the process
    sits in a leaf. A GitHub Actions runner, checked in this project's CI, has
    neither: its group belongs to root and has no controllers enabled for children.
    Expect the ``rlimit`` and ``watchdog`` mechanisms on most machines.

``rlimit``
    A POSIX per-process limit, inherited by everything the process starts. The
    kernel enforces it exactly, but each process gets its own budget, and an
    address-space limit counts memory that is mapped and never touched, which is
    why a virtual machine inside the sandbox can fail under it. Linux accepts an
    address-space limit; macOS rejects it, so macOS has no kernel memory limit.

``watchdog``
    The parent samples the memory of the command's process tree and group and stops
    it when it goes over.
    It works anywhere a ``ps`` command does and it can be overshot by whatever the
    program allocates between two samples, so it is a guard rail, not a boundary.
    It is what macOS uses for memory, and what Linux falls back to when neither a
    cgroup nor an address-space limit is available.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, cast

from .._compat import StrEnum
from .errors import SandboxPolicyError
from .policy import ResourceLimits

try:  # POSIX only; Windows has no rlimits.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on Windows
    _resource = None  # type: ignore[assignment]

__all__ = [
    "CGROUP_ROOT",
    "CPU_PERIOD_MICROSECONDS",
    "CgroupHandle",
    "CgroupUsage",
    "CgroupV2",
    "MemoryStrategy",
    "ResourcePlan",
    "ResourceReport",
    "RlimitStep",
    "apply_rlimits",
    "cgroup2_mount",
    "descendant_pids",
    "own_cgroup_path",
    "plan_resources",
    "plan_rlimits",
    "process_tree_rss_bytes",
    "rlimit_support",
    "user_task_count",
    "watchdog_available",
]

#: Where a cgroup v2 hierarchy is mounted when the machine uses the unified layout.
CGROUP_ROOT = Path("/sys/fs/cgroup")
#: Where Linux exposes one directory per process; a test points this at a fixture tree.
_PROC_ROOT = Path("/proc")
#: The period a ``cpu.max`` quota is spread over, in microseconds. 100 ms is the
#: kernel default and the unit ``cpu_percent`` is scaled to.
CPU_PERIOD_MICROSECONDS = 100_000

#: The POSIX limit behind each field, and the value used to probe whether the
#: kernel accepts it. A kernel that does not support a resource answers EINVAL.
_RLIMIT_FIELDS: dict[str, tuple[str, int]] = {
    "cpu_seconds": ("RLIMIT_CPU", 1),
    "processes": ("RLIMIT_NPROC", 1),
    "file_size_bytes": ("RLIMIT_FSIZE", 1),
    "open_files": ("RLIMIT_NOFILE", 4),
    # macOS rejects this one, which is why memory needs another mechanism there.
    "memory_bytes": ("RLIMIT_AS", 64 * 1024 * 1024),
    "core_dumps": ("RLIMIT_CORE", 0),
}


class MemoryStrategy(StrEnum):
    """Which mechanism to use for ``memory_bytes``.

    ``KERNEL`` and ``WATCHDOG`` exist because neither mechanism is right for every
    program: an address-space limit is exact but refuses memory a program only
    reserves, and a watchdog is generous but can be overshot.
    """

    #: The strongest mechanism this machine has: cgroup, then address space, then watchdog.
    AUTO = "auto"
    #: Only a kernel limit (cgroup or address space). A program that reserves
    #: memory it never uses, such as a virtual machine or the JVM, may be refused.
    KERNEL = "kernel"
    #: Only the sampling watchdog. Start-up is never refused, and the limit can be
    #: overshot by whatever is allocated between two samples.
    WATCHDOG = "watchdog"
    #: Do not enforce ``memory_bytes`` at all, and say so in the warnings.
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RlimitStep:
    """One ``setrlimit`` call: soft and hard are both set so a process cannot raise them back."""

    resource: int
    name: str
    soft: int
    hard: int


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """How each requested limit is enforced, and what a run measured.

    Attributes:
        memory: ``cgroup``, ``rlimit_as``, ``watchdog`` or ``none``.
        cpu_seconds: ``rlimit`` or ``none``.
        cpu_percent: ``cgroup`` or ``none``.
        processes: ``cgroup``, ``rlimit`` or ``none``.
        file_size_bytes: ``rlimit`` or ``none``.
        open_files: ``rlimit`` or ``none``.
        core_dumps: ``rlimit`` or ``none``.
        peak_memory_bytes: Highest memory seen, when the mechanism measures it.
        limits_hit: Limits the process reached: ``memory``, ``cpu`` or ``file_size``.
            A limit the kernel reports by killing the process is read from the
            signal that ended it, so a program that catches or ignores that signal
            (a Python process ignores ``SIGXFSZ``, for instance) instead sees the
            operation fail and this stays empty.
        warnings: Limits that were asked for and cannot be enforced on this machine,
            and any that were lowered to the machine's own ceiling.
    """

    memory: str = "none"
    cpu_seconds: str = "none"
    cpu_percent: str = "none"
    processes: str = "none"
    file_size_bytes: str = "none"
    open_files: str = "none"
    core_dumps: str = "none"
    peak_memory_bytes: int | None = None
    limits_hit: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    #: Report field for each limit a caller can set on
    #: :class:`~zett_agent.sandbox.policy.ResourceLimits`.
    FIELDS: ClassVar[tuple[str, ...]] = (
        "memory",
        "cpu_seconds",
        "cpu_percent",
        "processes",
        "file_size_bytes",
        "open_files",
        "core_dumps",
    )

    @property
    def mechanisms(self) -> dict[str, str]:
        """How each limit stands, by report field name, including ``none`` and ``watchdog``."""
        return {name: getattr(self, name) for name in self.FIELDS}


# -------------------------------------------------------------------- rlimits


#: Run in a fresh interpreter: for each limit, lower it the way a sandbox would and report
#: the names the kernel accepted. A separate process means the probe can never lower
#: this process's own limits, and no fork happens in a program that has threads.
_PROBE_PROGRAM = """
import resource, sys
for name, attribute, value in {checks!r}:
    rid = getattr(resource, attribute, None)
    if rid is None:
        continue
    try:
        _, hard = resource.getrlimit(rid)
        soft = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(rid, (soft, hard))
    except (ValueError, OSError):
        continue
    print(name)
"""


@lru_cache(maxsize=1)
def rlimit_support() -> frozenset[str]:
    """Return the limit names this kernel actually accepts.

    Probed once, in a short-lived child interpreter, because a kernel answers
    EINVAL for a resource it does not implement and the only way to know is to
    try. macOS rejects the address-space limit, which is what makes memory
    different there. If the child cannot run, nothing is reported as supported,
    so nothing is claimed that was not checked.
    """
    if _resource is None or os.name != "posix":
        return frozenset()
    checks = [(name, attribute, value) for name, (attribute, value) in _RLIMIT_FIELDS.items()]
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_PROGRAM.format(checks=checks)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    return frozenset(line.strip() for line in result.stdout.splitlines() if line.strip() in _RLIMIT_FIELDS)


def user_task_count() -> int | None:
    """Return how many processes (and, on Linux, threads) this user owns right now.

    ``RLIMIT_NPROC`` is compared with this number, not with the size of one process
    tree: a fork fails once the user owns that many in total. Linux counts threads
    too (it is "more precisely" a limit on threads), and macOS counts processes
    only. Linux reads ``/proc`` and macOS asks ``ps``, which is the system-provided
    path on a platform that has no ``/proc``.
    None means it could not be counted, and the caller then does not pretend to
    limit processes.
    """
    if os.name != "posix":
        return None
    if sys.platform.startswith("linux"):
        counted = _linux_task_count()
        if counted is not None:
            return counted
    return _ps_task_count()


def _linux_task_count(root: Path | None = None) -> int | None:
    """Count this uid's tasks from ``/proc``, as ``ps -L -U <uid>`` would.

    Each process's ``status`` names its real uid and how many threads it has; the
    sum is the number ``RLIMIT_NPROC`` applies to. None means ``/proc`` is not
    readable, and the caller falls back to ``ps``.
    """
    base = _PROC_ROOT if root is None else root
    uid = str(os.getuid())
    try:
        entries = list(base.iterdir())
    except OSError:
        return None
    total = 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # the process exited while it was being read
        fields = {line.split(":", 1)[0]: line.split(":", 1)[1] for line in status.splitlines() if ":" in line}
        if fields.get("Uid", "").split()[:1] != [uid]:
            continue
        threads = fields.get("Threads", "").strip()
        total += int(threads) if threads.isdigit() else 1
    return total if total else None


def _ps_task_count() -> int | None:
    """Count this uid's processes (threads on Linux) by asking ``ps``."""
    command = ["ps", "-U", str(os.getuid()), "-o", "pid="]
    if sys.platform.startswith("linux"):
        command = ["ps", "-L", "-U", str(os.getuid()), "-o", "lwp="]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.split() if line.isdigit())


def plan_rlimits(
    limits: ResourceLimits, *, supported: frozenset[str] | None = None, user_tasks: int | None = None
) -> tuple[tuple[RlimitStep, ...], tuple[str, ...]]:
    """Turn the POSIX limits into ``setrlimit`` calls, and say what cannot be applied.

    A requested value above the machine's hard limit is lowered to it with a
    warning rather than failing: the hard limit is already a ceiling the process
    cannot pass, so the request cannot be honoured as written and the caller is
    told. ``core_dumps`` is off unless asked for, so it appears in almost every
    plan. ``cpu_percent`` has no POSIX limit and is left to the cgroup.

    Two limits are not applied as written:

    * ``cpu_seconds`` sets the hard limit one second above the soft one. With the two
      equal, Linux skips ``SIGXCPU`` and kills the process with ``SIGKILL``, so the run
      could not say it hit the limit; with a second between them the process is ended
      by ``SIGXCPU`` at the limit, and the hard limit still stops one that handles it.
    * ``processes`` is the number of processes the command may *add*. The kernel
      compares ``RLIMIT_NPROC`` with every process the user owns, so the limit is that
      count plus the request. Taking the number literally would stop every command on
      a machine where the user already runs more than that. Other processes of the same
      user that start during the run use up the same allowance, which is why a cgroup,
      where one is available, is preferred.

    Args:
        user_tasks: The user's current process count; counted here when None.
    """
    kernel = rlimit_support() if supported is None else supported
    steps: list[RlimitStep] = []
    warnings: list[str] = []
    fields = [name for name in limits.requested() if name in _RLIMIT_FIELDS]
    if not limits.core_dumps:
        fields.append("core_dumps")
    for name in fields:
        attribute, _ = _RLIMIT_FIELDS[name]
        resource_id = getattr(_resource, attribute, None) if _resource is not None else None
        if _resource is None or resource_id is None or name not in kernel:
            warnings.append(f"{name} cannot be limited on {sys.platform}: this kernel does not support {attribute}")
            continue
        requested = 0 if name == "core_dumps" else getattr(limits, name)
        if name == "processes":
            current = user_task_count() if user_tasks is None else user_tasks
            if current is None:
                warnings.append("processes cannot be limited: the user's current process count could not be read")
                continue
            requested += current
        soft, hard = _resource.getrlimit(resource_id)
        ceiling = requested if hard == _resource.RLIM_INFINITY else min(requested, hard)
        if ceiling != requested:
            warnings.append(
                f"{name}: requested {requested} is above the hard limit {hard} and was lowered to {ceiling}"
            )
        grace = 1 if name == "cpu_seconds" else 0
        upper = ceiling + grace if hard == _resource.RLIM_INFINITY else min(ceiling + grace, hard)
        steps.append(RlimitStep(resource=resource_id, name=name, soft=ceiling, hard=upper))
    return tuple(steps), tuple(warnings)


def apply_rlimits(steps: tuple[RlimitStep, ...]) -> None:
    """Apply the steps. Runs in the child before ``exec``, so it never raises.

    The plan was made by the parent from this kernel's own answers, so a failure
    here means the limit is gone in a way the parent cannot report through an
    exception. Nothing is raised into the child: the command has to start, and
    :meth:`~zett_agent.sandbox.base.Sandbox.capabilities` already told the caller
    what this machine can do.
    """
    if _resource is None:  # pragma: no cover - there is no plan without the module
        return
    for step in steps:
        try:
            _resource.setrlimit(step.resource, (step.soft, step.hard))
        except (ValueError, OSError):  # pragma: no cover - the parent probed this
            continue


# -------------------------------------------------------------------- cgroups


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except OSError:
        return []


def cgroup2_mount(mountinfo_path: Path | None = None) -> Path | None:
    """Return the cgroup v2 mount point, or None when the machine has no unified hierarchy.

    Linux only. The mount is found in ``/proc/self/mountinfo`` as a ``cgroup2``
    entry; a machine on the old split hierarchy reports nothing here.
    """
    if not sys.platform.startswith("linux"):
        return None
    path = mountinfo_path if mountinfo_path is not None else Path("/proc/self/mountinfo")
    for line in _read_lines(path):
        fields = line.split()
        if len(fields) > 4 and fields[-3] == "cgroup2":
            return Path(fields[4])
    return None


def own_cgroup_path(mount: Path, cgroup_path: Path | None = None) -> Path | None:
    """Return the cgroup this process belongs to, below ``mount``.

    ``/proc/self/cgroup`` holds one ``hierarchy:controllers:path`` line per
    hierarchy; the unified one has empty controllers (``0::/user.slice/...``).
    """
    path = cgroup_path if cgroup_path is not None else Path("/proc/self/cgroup")
    for line in _read_lines(path):
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1] == "":
            return mount / parts[2].lstrip("/")
    return None


@dataclass
class CgroupUsage:
    """What a cgroup reports after a run."""

    peak_memory_bytes: int | None = None
    memory_events_max: int = 0
    oom_kills: int = 0
    cpu_usage_microseconds: int | None = None
    peak_processes: int | None = None


@dataclass
class CgroupHandle:
    """One cgroup created for one run: it is created by the parent, joined by the child."""

    path: Path
    #: Limits the kernel accepted, keyed the way :class:`ResourceReport` names them.
    applied: dict[str, str] = field(default_factory=dict)
    #: True when this handle only answers what would happen, for
    #: :meth:`~zett_agent.sandbox.base.Sandbox.capabilities`, and changes nothing.
    dry_run: bool = False

    def _write(self, name: str, value: str) -> bool:
        """Write one control file; False when that file is not there.

        A cgroup has a control file only when its controller reached this group,
        so a missing file is how the kernel says the limit cannot be set here. The
        check comes first because writing would create an ordinary file in a group
        that has none, which is not what a real hierarchy does.
        """
        if self.dry_run:
            # Nothing was created, so the answer is what a real handle would find:
            # the hierarchy was probed, which is what makes this group exist at all.
            return True
        target = self.path / name
        if not target.exists():
            return False
        try:
            target.write_text(value)
        except OSError:
            return False
        return True

    def set_memory(self, memory_bytes: int) -> bool:
        """Cap the group's memory and stop it from swapping its way around the cap."""
        if not self._write("memory.max", str(memory_bytes)):
            return False
        self._write("memory.swap.max", "0")
        # Kill the whole group at once rather than one task at a time, so a parent
        # cannot keep running after its children were taken.
        self._write("memory.oom.group", "1")
        self.applied["memory"] = "cgroup"
        return True

    def set_cpu_percent(self, cpu_percent: int) -> bool:
        """Give the group ``cpu_percent`` of one processor, as a quota per period."""
        quota = cpu_percent * (CPU_PERIOD_MICROSECONDS // 100)
        if not self._write("cpu.max", f"{quota} {CPU_PERIOD_MICROSECONDS}"):
            return False
        self.applied["cpu_percent"] = "cgroup"
        return True

    def set_processes(self, processes: int) -> bool:
        """Cap how many processes and threads the group may hold."""
        if not self._write("pids.max", str(processes)):
            return False
        self.applied["processes"] = "cgroup"
        return True

    def join(self) -> None:
        """Move the calling process into this cgroup. Runs in the child before ``exec``.

        Writing ``0`` to ``cgroup.procs`` moves the writer, so the command and
        everything it starts are charged to this group from the first instruction
        on. Nothing is raised: the child has to reach ``exec``.
        """
        try:
            (self.path / "cgroup.procs").write_text("0")
        except OSError:  # pragma: no cover - the parent created and probed this
            pass

    def usage(self) -> CgroupUsage:
        """Read what the kernel recorded for the group."""
        usage = CgroupUsage()
        peak = _read_lines(self.path / "memory.peak") or _read_lines(self.path / "memory.current")
        if peak:
            try:
                usage.peak_memory_bytes = int(peak[0].strip())
            except ValueError:
                pass
        for line in _read_lines(self.path / "memory.events") or _read_lines(self.path / "memory.events.local"):
            key, _, value = line.partition(" ")
            if key == "max":
                usage.memory_events_max = int(value or 0)
            elif key == "oom_kill":
                usage.oom_kills = int(value or 0)
        for line in _read_lines(self.path / "cpu.stat"):
            key, _, value = line.partition(" ")
            if key == "usage_usec":
                try:
                    usage.cpu_usage_microseconds = int(value)
                except ValueError:
                    pass
        peak_processes = _read_lines(self.path / "pids.peak") or _read_lines(self.path / "pids.current")
        if peak_processes:
            try:
                usage.peak_processes = int(peak_processes[0].strip())
            except ValueError:
                pass
        return usage

    def release(self) -> bool:
        """Remove the cgroup directory, retrying briefly while its processes exit."""
        for attempt in range(5):
            try:
                self.path.rmdir()
                return True
            except FileNotFoundError:
                return True
            except OSError:
                if attempt == 4:
                    return False
                time.sleep(0.02 * (attempt + 1))
        return False  # pragma: no cover - the loop always returns


@dataclass
class CgroupV2:
    """The cgroup v2 hierarchy, and whether it can host a sandbox's own group."""

    directory: Path
    _delegated: bool | None = None

    @classmethod
    def discover(cls, *, mount: Path | None = None, cgroup_path: Path | None = None) -> CgroupV2 | None:
        """Return the hierarchy this process could create a group in, or None.

        None means the machine has no cgroup v2, or this process is not in one.
        Whether the controllers are delegated is a separate question, asked by
        :meth:`delegated`, because answering it creates and removes a directory.
        """
        mount_point = mount if mount is not None else cgroup2_mount()
        if mount_point is None or not mount_point.is_dir():
            return None
        own = own_cgroup_path(mount_point, cgroup_path)
        return None if own is None else cls(directory=own)

    def delegated(self) -> bool:
        """Whether a child cgroup can be created here, probed once and remembered."""
        if self._delegated is None:
            probe = self.directory / f"zett-probe-{os.getpid()}"
            try:
                probe.mkdir()
            except (OSError, FileExistsError):
                self._delegated = False
            else:
                try:
                    probe.rmdir()
                except OSError:  # pragma: no cover - a probe we made should be removable
                    pass
                self._delegated = True
        return self._delegated

    def create(self, name: str) -> CgroupHandle | None:
        """Create one group for one run, or None when this hierarchy cannot hold it."""
        if not self.delegated():
            return None
        path = self.directory / f"{name}-{os.getpid()}"
        try:
            if path.exists():
                path = self.directory / f"{name}-{os.getpid()}-{id(path)}"
            path.mkdir()
        except OSError:
            return None
        return CgroupHandle(path=path)


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """What will be done about resources for one run.

    Attributes:
        report: The mechanism behind each limit, before the run measured anything.
        rlimits: The ``setrlimit`` calls to make in the child.
        cgroup: The group the child joins, when one was created.
        watchdog: Whether the parent samples the process group's memory.
        warnings: Limits that were asked for and cannot be enforced here.
    """

    report: ResourceReport
    rlimits: tuple[RlimitStep, ...] = ()
    cgroup: CgroupHandle | None = None
    watchdog: bool = False
    warnings: tuple[str, ...] = ()
    #: The watchdog's sampling interval, from the limits it was planned for.
    sample_seconds: float = 0.25

    def preexec(self) -> Callable[[], None] | None:
        """Return a callable for ``subprocess``' ``preexec_fn``, or None when nothing to do."""
        cgroup, steps = self.cgroup, self.rlimits
        if cgroup is None and not steps:
            return None

        def apply() -> None:
            if cgroup is not None:
                cgroup.join()
            apply_rlimits(steps)

        return apply

    def cleanup(self) -> None:
        """Release what planning created."""
        if self.cgroup is not None:
            self.cgroup.release()

    def measured(
        self, *, peak_memory_bytes: int | None, limits_hit: tuple[str, ...], warnings: tuple[str, ...] = ()
    ) -> ResourceReport:
        """Return the report with what the run observed added to it."""
        return replace(
            self.report,
            peak_memory_bytes=peak_memory_bytes,
            limits_hit=limits_hit,
            warnings=tuple(dict.fromkeys((*self.report.warnings, *warnings))),
        )


def _cgroup_for(
    limits: ResourceLimits, name: str, *, create: bool, directory: Path | None, cgroup_path: Path | None
) -> CgroupHandle | None:
    hierarchy = CgroupV2.discover(mount=directory, cgroup_path=cgroup_path)
    if hierarchy is None or not hierarchy.delegated():
        return None
    if not create:
        # Report only: nothing is created, so nothing can be joined or measured, but
        # the mechanisms a real run would use are the same.
        return CgroupHandle(path=hierarchy.directory / f"{name}-{os.getpid()}", dry_run=True)
    return hierarchy.create(name)


def plan_resources(
    limits: ResourceLimits,
    *,
    strategy: MemoryStrategy | str = MemoryStrategy.AUTO,
    name: str = "zett-sandbox",
    create_cgroup: bool = True,
    cgroup_directory: Path | None = None,
    cgroup_path: Path | None = None,
) -> ResourcePlan:
    """Decide how to enforce ``limits`` on this machine.

    The order of preference is the strongest mechanism each limit has: a cgroup
    for memory, processor share and process count; a POSIX limit for processor
    time, file size, open files and the address space; and the parent's sampling
    as the last resort for memory. Anything that cannot be enforced is named in
    the warnings, in the words of the thing that is missing.

    Args:
        limits: What the caller asked for.
        strategy: How to treat ``memory_bytes``; see :class:`MemoryStrategy`.
        name: Prefix of the cgroup to create.
        create_cgroup: False asks only what would be done, for
            :meth:`~zett_agent.sandbox.base.Sandbox.capabilities`, and creates nothing.
        cgroup_directory: Override the cgroup v2 mount, for tests.
        cgroup_path: Override ``/proc/self/cgroup``, for tests.
    """
    try:
        chosen = MemoryStrategy(strategy)
    except ValueError:
        raise SandboxPolicyError(f"unknown memory strategy: {strategy!r}") from None

    kernel = rlimit_support()
    planned, warnings = plan_rlimits(limits, supported=kernel)
    steps = {step.name: step for step in planned}
    mechanisms: dict[str, str] = dict.fromkeys(
        ("memory", "cpu_seconds", "cpu_percent", "processes", "file_size_bytes", "open_files", "core_dumps"), "none"
    )
    if "core_dumps" in steps:
        mechanisms["core_dumps"] = "rlimit"

    handle = _cgroup_for(limits, name, create=create_cgroup, directory=cgroup_directory, cgroup_path=cgroup_path)
    used_rlimits: list[RlimitStep] = []
    notes = list(warnings)

    if limits.memory_bytes is not None:
        if chosen is MemoryStrategy.NONE:
            notes.append("memory_bytes was requested but the memory strategy is 'none'")
        elif handle is not None and handle.set_memory(limits.memory_bytes):
            mechanisms["memory"] = "cgroup"
        elif chosen in (MemoryStrategy.AUTO, MemoryStrategy.KERNEL) and "memory_bytes" in steps:
            used_rlimits.append(steps["memory_bytes"])
            mechanisms["memory"] = "rlimit_as"
        elif chosen is MemoryStrategy.KERNEL:
            notes.append("memory_bytes: this kernel cannot enforce it and the memory strategy forbids the watchdog")
        elif watchdog_available():
            mechanisms["memory"] = "watchdog"
            notes.append(
                "memory_bytes is enforced by sampling the process group, which can be overshot between samples; "
                "Linux enforces it exactly with cgroup v2 or, if you accept an address-space limit, with "
                "MemoryStrategy.KERNEL"
            )
        else:
            notes.append(f"memory_bytes cannot be enforced on {sys.platform}: no cgroup, no address-space limit")

    if limits.cpu_percent is not None:
        if handle is not None and handle.set_cpu_percent(limits.cpu_percent):
            mechanisms["cpu_percent"] = "cgroup"
        else:
            notes.append(
                "cpu_percent needs cgroup v2 with the cpu controller delegated to this process, and no other "
                "mechanism limits a share of a processor"
            )

    if limits.processes is not None:
        if handle is not None and handle.set_processes(limits.processes):
            mechanisms["processes"] = "cgroup"
        elif "processes" in steps:
            used_rlimits.append(steps["processes"])
            mechanisms["processes"] = "rlimit"
        else:
            notes.append("processes cannot be limited on this machine")

    for field_name in ("cpu_seconds", "file_size_bytes", "open_files", "core_dumps"):
        if field_name in steps:
            used_rlimits.append(steps[field_name])
            mechanisms[field_name] = "rlimit"

    if handle is not None:
        mechanisms.update(handle.applied)
    # A limit the cgroup or the watchdog took over is settled, so the note about the
    # kernel limit it would otherwise have needed is noise. What is kept is the note
    # that says how the fallback behaves.
    settled = {"memory_bytes": "memory", "processes": "processes"}
    notes = [
        note
        for note in notes
        if not any(
            note.startswith(f"{field} cannot be limited") and mechanisms[report_field] != "none"
            for field, report_field in settled.items()
        )
    ]
    # A limit the cgroup took over is not also asked of the kernel twice.
    used_rlimits = (
        [step for step in used_rlimits if step.name not in handle.applied] if handle is not None else used_rlimits
    )
    return ResourcePlan(
        # ``mechanisms`` holds exactly the per-limit fields of ``ResourceReport``, one string each.
        report=ResourceReport(**cast("dict[str, Any]", mechanisms), warnings=tuple(dict.fromkeys(notes))),
        rlimits=tuple(used_rlimits),
        cgroup=handle if create_cgroup else None,
        watchdog=mechanisms["memory"] == "watchdog",
        warnings=tuple(dict.fromkeys(notes)),
        sample_seconds=limits.memory_sample_seconds,
    )


def _process_table() -> tuple[dict[int, tuple[int, int, int]], dict[int, list[int]]] | None:
    """Read every process as ``pid -> (parent, group, resident KiB)`` and the parent-to-children map.

    Linux reads ``/proc``; macOS asks ``ps`` once, which serves both the memory
    watchdog and the tree kill. None means the table could not be read.
    """
    if sys.platform.startswith("linux"):
        table = _linux_process_table()
        if table is not None:
            return table
    return _ps_process_table()


def _linux_process_table(
    root: Path | None = None,
) -> tuple[dict[int, tuple[int, int, int]], dict[int, list[int]]] | None:
    """Build the process table from ``/proc`` without running a program."""
    base = _PROC_ROOT if root is None else root
    try:
        entries = list(base.iterdir())
    except OSError:
        return None
    page_size = os.sysconf("SC_PAGE_SIZE")
    rows: dict[int, tuple[int, int, int]] = {}
    children: dict[int, list[int]] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # the process exited while it was being read
        # The command name is parenthesized and may itself contain spaces or parentheses.
        _, _, tail = stat.rpartition(")")
        fields = tail.split()
        if len(fields) < 22:
            continue
        try:
            parent, group, resident_pages = int(fields[1]), int(fields[2]), int(fields[21])
        except ValueError:
            continue
        process = int(entry.name)
        rows[process] = (parent, group, resident_pages * page_size // 1024)
        children.setdefault(parent, []).append(process)
    if not rows:
        return None
    return rows, children


def _ps_process_table() -> tuple[dict[int, tuple[int, int, int]], dict[int, list[int]]] | None:
    """Build the process table from one ``ps`` call."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss="], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    rows: dict[int, tuple[int, int, int]] = {}
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            process, parent, group, rss = (int(field) for field in fields)
        except ValueError:
            continue
        rows[process] = (parent, group, rss)
        children.setdefault(parent, []).append(process)
    return rows, children


def _tree_members(pid: int, rows: dict[int, tuple[int, int, int]], children: dict[int, list[int]]) -> set[int]:
    """The processes in ``pid``'s group plus ``pid`` and everything below it."""
    members = {process for process, (_, group, _) in rows.items() if group == pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in rows:
            members.add(current)
        pending.extend(child for child in children.get(current, ()) if child not in members)
    return members


def descendant_pids(pid: int) -> frozenset[int]:
    """Return the processes below ``pid``, including ones that started a session of their own.

    A process group is not enough to stop a command: a child that calls ``setsid``
    leaves the group, and killing the group misses it. While the parent is alive the
    parent links are still there, so the tree finds it. Once the parent has exited
    the child is adopted by ``init`` and nothing links it to the command any more,
    which is why this is used when a command is being stopped, not afterwards.
    Empty when ``ps`` cannot be read.
    """
    table = _process_table()
    if table is None:
        return frozenset()
    rows, children = table
    return frozenset(_tree_members(pid, rows, children) - {pid, os.getpid()})


def process_tree_rss_bytes(pid: int) -> int | None:
    """Return the resident memory of ``pid``, its descendants and its process group.

    The group alone misses a command that starts its own session, which is exactly
    what bubblewrap does (``--new-session``), so its processes sit in a group of
    their own. The descendants alone miss a process that was orphaned and
    reparented to ``init``, while it is still in the group. A process that does
    both, detaching and orphaning itself, is in neither and is not counted.

    The host runs this, outside the sandbox, so it sees everything the sandbox
    started. None means ``ps`` could not be read, which the watchdog reports rather
    than treating as "no memory used".
    """
    table = _process_table()
    if table is None:
        return None
    rows, children = table
    # ps reports resident size in kibibytes.
    return sum(rows[process][2] for process in _tree_members(pid, rows, children)) * 1024


def watchdog_available() -> bool:
    """Whether the parent can sample a process group here."""
    return os.name == "posix"
