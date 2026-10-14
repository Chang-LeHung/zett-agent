"""What a sandboxed process may touch, independent of how a platform enforces it."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .._compat import StrEnum
from .errors import SandboxPolicyError

__all__ = [
    "DEFAULT_INHERITED_ENVIRONMENT",
    "MINIMUM_MEMORY_BYTES",
    "PROTECTED_DIRECTORY_NAMES",
    "PROTECTED_FILE_NAMES",
    "SENSITIVE_HOME_PATHS",
    "SENSITIVE_SOCKET_PATHS",
    "NetworkMode",
    "ResolvedPolicy",
    "ResourceLimits",
    "SandboxPolicy",
    "build_environment",
    "command_arguments",
    "resolve_policy",
]


class NetworkMode(StrEnum):
    """How much of the network a sandboxed process can reach."""

    #: No network at all. Nothing can connect out, and nothing can connect in.
    NONE = "none"
    #: Unrestricted network. Filtering by host needs a proxy and is not offered.
    FULL = "full"


#: Environment variables a sandboxed process receives from the host when the
#: policy does not say otherwise. Everything else, which is where API keys and
#: tokens usually live, is withheld; a caller adds what a command needs through
#: the ``env`` argument of a run. The Windows names are matched without regard
#: to case.
DEFAULT_INHERITED_ENVIRONMENT: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "TERM",
    "TZ",
    "TMPDIR",
    "COLORTERM",
    "NO_COLOR",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "PROGRAMDATA",
)

#: Files that run code in a later, unsandboxed context (a login shell, git, an
#: editor). They stay unwritable inside a writable directory, so a process that
#: may write the workspace still cannot plant something the user runs next.
PROTECTED_FILE_NAMES: tuple[str, ...] = (
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".gitconfig",
    ".gitmodules",
    ".ripgreprc",
    ".mcp.json",
)

#: Directories with the same property. ``.git/hooks`` and ``.git/config`` are
#: protected wherever a repository is found, not by name alone.
PROTECTED_DIRECTORY_NAMES: tuple[str, ...] = (".vscode", ".idea")

#: Credential locations under the user's home directory. A policy that keeps
#: ``protect_defaults`` on hides them from reads, because a process that can read
#: them and reach the network can send them anywhere.
SENSITIVE_HOME_PATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".azure",
    ".kube",
    ".config/gcloud",
    ".config/gh",
    ".docker/config.json",
    ".docker/run/docker.sock",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "Library/Keychains",
)

#: Container-engine sockets. Connecting to one is the same as being root on the
#: host, and a read-only mount does not stop a connection to a socket file.
SENSITIVE_SOCKET_PATHS: tuple[str, ...] = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/var/run/podman/podman.sock",
    "/run/podman/podman.sock",
)

#: Directories the search for protected paths does not descend into: they are
#: large, generated, and not where a repository is hiding.
_SKIPPED_DIRECTORIES = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",
        "dist",
        "build",
        ".git",
    }
)
_PROTECTED_SEARCH_DEPTH = 3
_PROTECTED_SEARCH_ENTRIES = 10_000


def _as_path(value: str | os.PathLike[str], field: str) -> Path:
    """Expand ``~`` and require an absolute path, so a policy never depends on the working directory."""
    if not isinstance(value, (str, os.PathLike)):
        raise SandboxPolicyError(f"{field} entries must be paths, not {type(value).__name__}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SandboxPolicyError(f"{field} must be absolute (or start with ~): {str(value)!r}")
    if "\0" in str(path):
        raise SandboxPolicyError(f"{field} must not contain NUL: {str(value)!r}")
    return path


def _as_paths(values: Iterable[str | os.PathLike[str]], field: str) -> tuple[Path, ...]:
    if isinstance(values, (str, os.PathLike)):
        raise SandboxPolicyError(f"{field} must be a sequence of paths, not one path")
    return tuple(_as_path(value, field) for value in values)


#: Smallest memory cap the sandbox accepts. A launcher has to start and map the
#: filesystem inside the sandbox before the command runs, and that needs room.
MINIMUM_MEMORY_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """What a sandboxed process may use, and how much.

    Every field is optional; ``None`` means the machine's own default, which is
    usually no limit. A value is a request, not a promise: how much of it the
    kernel can enforce depends on the platform, and
    :meth:`~zett_agent.sandbox.base.Sandbox.capabilities` says what a given
    machine will actually do with each one. What a backend cannot enforce is
    reported in the run's warnings rather than silently dropped. See
    :doc:`../../learn/sandbox` for the table of what each platform enforces.

    Args:
        memory_bytes: Memory the process and everything it starts may use. The
            kernel accounts for the whole tree where cgroups are available; a
            per-process address-space limit is used where they are not, which
            also counts memory that is mapped but never touched. On macOS no
            kernel limit exists and a sampling watchdog stops a runaway process
            instead; see :class:`~zett_agent.sandbox.base.MemoryStrategy`.
        cpu_seconds: Total processor time the process tree may consume. The
            kernel kills it once the time is spent, which is different from the
            wall-clock ``timeout`` of a run: a program that sleeps uses no
            processor time at all.
        cpu_percent: Share of one processor the process tree may use, where the
            kernel can limit it (cgroups). 100 is one core, 250 is two and a
            half. Sleeps and waiting do not count against it.
        processes: Number of processes and threads the tree may create. The
            smallest accepted value leaves room for the sandbox's own launcher,
            which is itself a process in the tree; on macOS threads count too, as
            the kernel counts them. A small value stops a program from starting
            helpers at all.
        file_size_bytes: Largest file the process may create or extend. Writing
            past it fails; a file already open can be truncated by it.
        open_files: Number of file descriptors the process may hold open. The
            smallest accepted value again leaves room for the launcher, which needs
            descriptors of its own to set the sandbox up.
        core_dumps: Whether a crash may write a core file. Off by default: a core
            file of a large process is large, and it holds whatever the process
            had read.
        memory_sample_seconds: How often the watchdog measures memory where one
            is used. Smaller values catch a runaway sooner and cost a little more.

    Examples:
        Give a command half a processor, 30 seconds of processor time, 512 MiB
        and no more than 64 processes::

            limits = ResourceLimits(
                memory_bytes=512 * 1024**2, cpu_seconds=30, cpu_percent=50, processes=64
            )
            policy = SandboxPolicy(writable_paths=["/work"], limits=limits)

    .. warning::
        A resource limit protects the machine from a process that runs away. It
        is not a defense against a process that respects no limits: it can still
        use what it is allowed as fast as it likes, and the watchdog used on
        macOS can be overshot between two samples.
    """

    memory_bytes: int | None = None
    cpu_seconds: int | None = None
    cpu_percent: int | None = None
    processes: int | None = None
    file_size_bytes: int | None = None
    open_files: int | None = None
    core_dumps: bool = False
    memory_sample_seconds: float = 0.25

    def __post_init__(self) -> None:
        # The floors are what the launcher needs to start at all: it is a process in
        # the tree with descriptors of its own, and a limit below these stops it
        # before the command runs.
        for name, minimum, note in (
            ("cpu_seconds", 1, ""),
            ("cpu_percent", 1, ""),
            ("processes", 8, " (the launcher is a process in the tree)"),
            ("file_size_bytes", 1, ""),
            ("open_files", 16, " (the launcher needs descriptors of its own)"),
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise SandboxPolicyError(f"{name} must be an integer of at least {minimum}{note}, or None")
        if self.memory_bytes is not None:
            if (
                isinstance(self.memory_bytes, bool)
                or not isinstance(self.memory_bytes, int)
                or self.memory_bytes < MINIMUM_MEMORY_BYTES
            ):
                raise SandboxPolicyError(
                    f"memory_bytes must be an integer of at least {MINIMUM_MEMORY_BYTES} "
                    f"({MINIMUM_MEMORY_BYTES // 1024**2} MiB), or None"
                )
        if not isinstance(self.core_dumps, bool):
            raise SandboxPolicyError("core_dumps must be a boolean")
        if (
            isinstance(self.memory_sample_seconds, bool)
            or not isinstance(self.memory_sample_seconds, (int, float))
            or not 0.01 <= self.memory_sample_seconds <= 60
        ):
            raise SandboxPolicyError("memory_sample_seconds must be between 0.01 and 60")

    def requested(self) -> tuple[str, ...]:
        """Names of the limits that were asked for, in field order."""
        return tuple(
            field
            for field in (
                "memory_bytes",
                "cpu_seconds",
                "cpu_percent",
                "processes",
                "file_size_bytes",
                "open_files",
            )
            if getattr(self, field) is not None
        )


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """The boundary a sandboxed process runs inside.

    The shape follows what the established agent sandboxes converge on. Reads
    are open by default and a deny list hides credentials; writes are closed by
    default and only the listed directories open; the network is closed by
    default; and a few paths stay protected even inside a writable directory. A
    policy says what to enforce, never how: each backend turns it into its own
    mechanism, so the same policy works everywhere a backend exists.

    Every path must be absolute (``~`` is expanded), so a policy means the same
    thing wherever it is built. Writable paths must exist; they are resolved
    through symbolic links, and a write is judged by where it lands.

    Args:
        writable_paths: Directories (or files) the process may write. Empty
            means it can write nowhere but the temporary directory.
        denied_write_paths: Paths that stay unwritable even inside a writable
            path. A denial beats a permission.
        denied_read_paths: Paths the process cannot read. Add the secrets that
            matter to you; ``protect_defaults`` already covers common ones.
        network: Whether the process can reach the network.
        protect_defaults: Keep shell and git configuration, ``.git/hooks`` and
            editor settings unwritable inside writable paths, and hide the
            usual credential files and container sockets from reads.
        inherit_environment: Names of host environment variables the process
            receives. The default is a short, secret-free list; a command that
            needs more gets it through the ``env`` argument of a run.
        limits: What the process may use: memory, processor time, processor
            share, processes, file size and open files. Empty by default, which
            leaves the machine's own limits in place.

    Examples:
        Let a process write one directory and nothing else::

            policy = SandboxPolicy.workspace_write("/work/project")

        Allow the network as well, still hiding the credential files::

            policy = SandboxPolicy.workspace_write("/work/project", network=NetworkMode.FULL)

    .. note::
        Temporary files always work: the process gets a private ``/tmp`` on
        Linux, and the user temporary directory and ``/tmp`` on macOS.

    .. warning::
        A policy bounds what a process can do on this machine. It is not a
        defense against a process that uses the network access you allow, or
        against a kernel or sandbox vulnerability.
    """

    writable_paths: tuple[Path, ...] = ()
    denied_write_paths: tuple[Path, ...] = ()
    denied_read_paths: tuple[Path, ...] = ()
    network: NetworkMode = NetworkMode.NONE
    protect_defaults: bool = True
    inherit_environment: tuple[str, ...] = DEFAULT_INHERITED_ENVIRONMENT
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "writable_paths", _as_paths(self.writable_paths, "writable_paths"))
        object.__setattr__(self, "denied_write_paths", _as_paths(self.denied_write_paths, "denied_write_paths"))
        object.__setattr__(self, "denied_read_paths", _as_paths(self.denied_read_paths, "denied_read_paths"))
        try:
            object.__setattr__(self, "network", NetworkMode(self.network))
        except ValueError:
            raise SandboxPolicyError(f"network must be a NetworkMode, not {self.network!r}") from None
        if not isinstance(self.protect_defaults, bool):
            raise SandboxPolicyError("protect_defaults must be a boolean")
        names = self.inherit_environment
        if isinstance(names, str) or not all(isinstance(name, str) and name and "=" not in name for name in names):
            raise SandboxPolicyError("inherit_environment must be a sequence of variable names")
        object.__setattr__(self, "inherit_environment", tuple(names))
        if not isinstance(self.limits, ResourceLimits):
            raise SandboxPolicyError("limits must be a ResourceLimits")

    @classmethod
    def read_only(
        cls, *, network: NetworkMode = NetworkMode.NONE, limits: ResourceLimits | None = None
    ) -> SandboxPolicy:
        """Return a policy that can read the machine and write nothing but temporary files."""
        return cls(network=network, limits=limits if limits is not None else ResourceLimits())

    @classmethod
    def workspace_write(
        cls,
        *roots: str | os.PathLike[str],
        network: NetworkMode = NetworkMode.NONE,
        protect_defaults: bool = True,
        limits: ResourceLimits | None = None,
    ) -> SandboxPolicy:
        """Return a policy that can write ``roots``, read the machine, and use the network only if asked."""
        return cls(
            writable_paths=roots,
            network=network,
            protect_defaults=protect_defaults,
            limits=limits if limits is not None else ResourceLimits(),
        )


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    """A policy turned into concrete absolute paths that a backend can enforce.

    Attributes:
        writable: Existing directories or files the process may write, with
            symbolic links resolved.
        denied_write: Paths the caller marked unwritable.
        protected_write: Default-protected paths inside the writable ones. A
            path may not exist yet; a backend that cannot deny a missing path
            skips it.
        denied_read: Paths hidden from reads, the caller's and the defaults
            that exist on this machine.
        network: The network mode.
        warnings: Things the policy asked for that could not be set up.
    """

    writable: tuple[Path, ...]
    denied_write: tuple[Path, ...]
    protected_write: tuple[Path, ...]
    denied_read: tuple[Path, ...]
    network: NetworkMode
    warnings: tuple[str, ...] = ()

    @property
    def unwritable(self) -> tuple[Path, ...]:
        """Every path to keep unwritable: the caller's denials, then the protected ones."""
        return (*self.denied_write, *self.protected_write)


def _unique(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _protected_in(root: Path) -> tuple[list[Path], bool]:
    """Find paths under one writable root that must stay unwritable.

    The root itself always contributes its well-known names, present or not, so a
    backend that can deny a missing path stops them being created. A shallow
    search then finds the same names, and git repositories, below the root.
    """
    found: list[Path] = []
    for name in (*PROTECTED_FILE_NAMES, *PROTECTED_DIRECTORY_NAMES):
        found.append(root / name)
    found.extend((root / ".git" / "hooks", root / ".git" / "config"))
    visited = 0
    pending: deque[tuple[Path, int]] = deque([(root, 0)])
    while pending:
        directory, depth = pending.popleft()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            visited += 1
            if visited > _PROTECTED_SEARCH_ENTRIES:
                return found, True
            name = entry.name
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            path = directory / name
            if name == ".git":
                if is_directory:
                    found.extend((path / "hooks", path / "config"))
                continue
            if name in PROTECTED_FILE_NAMES or (is_directory and name in PROTECTED_DIRECTORY_NAMES):
                found.append(path)
            if is_directory and name not in _SKIPPED_DIRECTORIES and depth + 1 < _PROTECTED_SEARCH_DEPTH:
                pending.append((path, depth + 1))
    return found, False


def _drop_conflicts(
    denied_read: list[Path], writable: Sequence[Path], policy: SandboxPolicy, warnings: list[str]
) -> list[Path]:
    """Keep a denied-read path from hiding a writable one.

    A path the process may write is a path it must be able to read. A denial
    that equals or contains a writable path is a contradiction. The caller's own
    entry is rejected, because it is a mistake to fix; a default one is dropped
    with a warning, because the caller never wrote it.
    """
    explicit = {path.resolve(strict=False) for path in policy.denied_read_paths}
    kept: list[Path] = []
    for denied in denied_read:
        inside = [path for path in writable if path == denied or path.is_relative_to(denied)]
        if not inside:
            kept.append(denied)
        elif denied in explicit:
            raise SandboxPolicyError(f"denied_read_paths entry {denied} would hide the writable path {inside[0]}")
        else:
            warnings.append(f"default read protection of {denied} skipped: it contains the writable path {inside[0]}")
    return kept


def resolve_policy(policy: SandboxPolicy, *, home: Path | None = None) -> ResolvedPolicy:
    """Turn a policy into the concrete paths a backend enforces.

    Raises:
        SandboxPolicyError: If a writable path does not exist, or is the
            filesystem root (which would defeat the sandbox).
    """
    warnings: list[str] = []
    writable: list[Path] = []
    for path in policy.writable_paths:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise SandboxPolicyError(f"writable path does not exist or cannot be resolved: {path}") from error
        if resolved == Path(resolved.anchor):
            raise SandboxPolicyError(
                "refusing to make the filesystem root writable; use PassthroughSandbox to run without isolation"
            )
        writable.append(resolved)

    denied_write = _unique(path.resolve(strict=False) for path in policy.denied_write_paths)
    denied_read = [path.resolve(strict=False) for path in policy.denied_read_paths]

    protected: list[Path] = []
    if policy.protect_defaults:
        base = home if home is not None else Path.home()
        candidates = [base / relative for relative in SENSITIVE_HOME_PATHS]
        candidates.extend(Path(socket) for socket in SENSITIVE_SOCKET_PATHS)
        denied_read.extend(path.resolve(strict=False) for path in candidates if os.path.lexists(path))
        for root in _unique(writable):
            paths, truncated = _protected_in(root)
            protected.extend(paths)
            if truncated:
                warnings.append(
                    f"{root}: too many entries to search for protected files; only part of the tree is covered"
                )
    denied_read = _drop_conflicts(denied_read, writable, policy, warnings)
    return ResolvedPolicy(
        writable=_unique(writable),
        denied_write=denied_write,
        protected_write=_unique(protected),
        denied_read=_unique(denied_read),
        network=policy.network,
        warnings=tuple(warnings),
    )


def build_environment(
    names: Iterable[str],
    extra: Mapping[str, str] | None = None,
    *,
    source: Mapping[str, str] | None = None,
    case_insensitive: bool | None = None,
) -> dict[str, str]:
    """Build the environment a sandboxed process starts with.

    The result holds the variables in ``names`` that the host has, then
    ``extra``, which wins. Nothing else leaks through, so a token that sits in
    the host environment stays there. Windows matches names without regard to
    case, as the platform does.

    Raises:
        SandboxPolicyError: If ``extra`` is not a mapping of strings, or has a
            name the operating system cannot hold.
    """
    host = os.environ if source is None else source
    insensitive = os.name == "nt" if case_insensitive is None else case_insensitive
    fold = str.upper if insensitive else (lambda name: name)
    wanted = {fold(name) for name in names}
    environment = {key: value for key, value in host.items() if fold(key) in wanted}
    if extra:
        if not isinstance(extra, Mapping):
            raise SandboxPolicyError("env must be a mapping of names to values")
        for key, value in extra.items():
            if not isinstance(key, str) or not key or "=" in key or "\0" in key:
                raise SandboxPolicyError(f"invalid environment variable name: {key!r}")
            if not isinstance(value, str) or "\0" in value:
                raise SandboxPolicyError(f"environment variable {key!r} must be a string without NUL")
            for existing in [name for name in environment if fold(name) == fold(key)]:
                del environment[existing]
            environment[key] = value
    return environment


def command_arguments(command: str | Sequence[str], *, shell: str = "/bin/sh") -> tuple[str, ...]:
    """Return the argument vector for a command given as a shell string or as arguments.

    A string runs through ``shell -c``, so pipes and redirections work. A
    sequence runs directly, which is the safer form for a program and its
    arguments because nothing is parsed.

    Raises:
        SandboxPolicyError: If the command is empty or contains NUL.
    """
    if isinstance(command, str):
        if not command.strip():
            raise SandboxPolicyError("command cannot be blank")
        arguments: tuple[str, ...] = (shell, "-c", command)
    else:
        arguments = tuple(command)
        if not arguments or not all(isinstance(item, str) for item in arguments) or not arguments[0]:
            raise SandboxPolicyError("command must be a string or a non-empty sequence of strings")
    if any("\0" in item for item in arguments):
        raise SandboxPolicyError("command must not contain NUL")
    return arguments
