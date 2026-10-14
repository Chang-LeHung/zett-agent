Run commands in a sandbox
=========================

A tool that runs a shell command, or a program the model just wrote, runs with
everything your process can do. A sandbox gives that command a smaller world:
it can read the machine, write only where you say, and use the network only if
you allow it. The kernel enforces the boundary on the command and on every
process it starts.

The sandbox and an approval prompt are two different controls. The sandbox
defines what a command *can* do; asking a person first decides whether it *runs*
at all. Use them together, for example with
:class:`~zett_agent.extensions.shell_approval.ShellApprovalExtension`.

Use a session
----------------------------------------

:class:`~zett_agent.sandbox.session.SandboxSession` is the way to use the
sandbox. Say where commands may write and how much they may use, then use it like
ordinary Python. Leaving the ``with`` block removes everything the session
created.

.. code-block:: python

   from zett_agent.sandbox.session import SandboxSession

   with SandboxSession("/work/project", memory="512MB", cpus=1, cpu_seconds=60) as sb:
       sb.run("python build.py", check=True)
       print(sb.run("ls dist").stdout)

Leave the workspace out and the session makes a private one and deletes it on
exit, which is what you want for a program someone else wrote:

.. code-block:: python

   with SandboxSession(memory="256MB", timeout=30) as sb:
       (sb.workspace / "job.py").write_text(code)
       result = sb.run(["python3", "job.py"], check=True)
   # the workspace and its files are gone here

``run`` behaves like :func:`subprocess.run`. A list runs the program directly
with nothing parsed, which is the safer form for anything built from untrusted
text; a string runs through ``/bin/sh -c``, so pipes work. ``check=True`` raises
:class:`~zett_agent.sandbox.errors.SandboxCommandError`, which carries the whole
result. From async code use ``async with`` and ``await sb.arun(...)``.

What the session does for you:

* **Writes are confined** to the workspace and private scratch space. Everything
  else on the machine is read-only.
* **The network is off** unless you pass ``network=True``.
* **Secrets stay out.** Credential files such as ``~/.ssh`` are unreadable, and
  commands get a short environment allowlist rather than a copy of yours.
* **Nothing outlives the block.** Closing, normally or through an exception,
  cancels commands that are still running, kills their whole process groups, and
  then deletes the directories the session created. A program that leaves
  directories it cannot enter does not block the cleanup. If you never close the
  session, deleting the object or exiting the interpreter cleans up too.
* **Your own directories are never touched.** A ``workspace`` or ``writable``
  path you pass is used and left exactly as the commands made it.

The options map straight onto the lower-level policy described below:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``workspace``
     - Directory commands may write. Omit it for a private one that is deleted.
   * - ``memory``, ``max_file_size``
     - A byte count or a string such as ``"512MB"`` (``K``, ``M``, ``G``, ``T`` are powers of 1024).
   * - ``cpus``
     - Cores: ``0.5`` is half of one core. Linux only.
   * - ``cpu_seconds``, ``processes``, ``open_files``
     - Processor time, processes and threads, and file descriptors.
   * - ``network``
     - ``True`` for unrestricted network; the default is none.
   * - ``writable``, ``deny_write``, ``deny_read``
     - More writable directories, and paths that stay unwritable or unreadable.
   * - ``env``, ``timeout``
     - Variables every command gets, and the default time limit in seconds.
   * - ``backend``, ``require_isolation``, ``memory_strategy``
     - Force a backend, refuse to run unisolated, and choose how memory is limited.

.. note::

   Limits apply to each command, not to the session as a whole: two commands run
   one after the other each get the full memory budget.

:download:`Download sandbox.py <../_examples/sandbox.py>` for a complete program.

.. literalinclude:: ../_examples/sandbox.py
   :language: python
   :pyobject: main

The lower-level API
----------------------------------------

A session builds the pieces below for you. Use them directly when you need
something it does not offer: your own process handling with ``wrap``, one
sandbox shared by several workspaces, or a policy built once and reused.

.. code-block:: python

   from zett_agent.sandbox.factory import create_sandbox
   from zett_agent.sandbox.policy import SandboxPolicy

   sandbox = create_sandbox(SandboxPolicy.workspace_write("/work/project"))
   result = await sandbox.run(["python3", "build.py"], cwd="/work/project")

Every backend returns the same :class:`~zett_agent.sandbox.base.SandboxResult`:
the exit status, standard output and error, whether it timed out, whether output
was cut off, and whether the backend isolated anything.

Describe the boundary
----------------------------------------

One :class:`~zett_agent.sandbox.policy.SandboxPolicy` says what to enforce, and
each backend turns it into its own mechanism. It follows the shape the
established agent sandboxes share: reads are open and a deny list hides secrets,
writes are closed and only listed paths open, the network is closed, and a few
paths stay protected even inside a writable one.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``writable_paths``
     - Directories the process may write. Empty means it can write only temporary files.
   * - ``denied_write_paths``
     - Stay unwritable even inside a writable path. A denial beats a permission.
   * - ``denied_read_paths``
     - Cannot be read. Add the secrets that matter to you.
   * - ``network``
     - ``NetworkMode.NONE`` (the default) or ``NetworkMode.FULL``.
   * - ``protect_defaults``
     - Keep shell and git configuration, ``.git/hooks`` and editor settings unwritable inside writable paths, and hide common credential files and container sockets from reads.
   * - ``inherit_environment``
     - Which host environment variables the process receives.
   * - ``limits``
     - How much the process may use: memory, processor time, files and processes.

Paths must be absolute (``~`` is expanded), so a policy means the same thing
wherever it is built. ``SandboxPolicy.read_only()`` and
``SandboxPolicy.workspace_write(*roots)`` cover the two common shapes.

The environment is a short allowlist, not a copy of yours. An API key in your
shell does not reach a command unless you pass it with
``sandbox.run(..., env={"TOKEN": value})``.

Protected files matter because a process that can write your workspace could
plant something that runs later, outside the sandbox: a ``.git/hooks/pre-commit``
script, a line in ``.bashrc``, an editor task. They stay unwritable, and the
default read protection hides ``~/.ssh``, ``~/.aws``, ``~/.gnupg`` and the like,
because a process that can read a credential and reach the network can send it
anywhere.

Limit what a process uses
----------------------------------------

A boundary says *where* a process may act. A budget says *how much* it may use,
which is what keeps one runaway command from taking the machine with it. Both live
in the same policy::

   from zett_agent.sandbox.policy import ResourceLimits, SandboxPolicy

   limits = ResourceLimits(
       memory_bytes=512 * 1024**2,   # the whole tree, not one process
       cpu_seconds=30,               # processor time, so sleeping costs nothing
       cpu_percent=50,               # half of one core
       processes=64,
       file_size_bytes=10 * 1024**2,
       open_files=256,
   )
   policy = SandboxPolicy(writable_paths=["/work/project"], limits=limits)

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``memory_bytes``
     - Memory the process and everything it starts may use.
   * - ``cpu_seconds``
     - Total processor time before the kernel kills the process. A program that sleeps uses none of it.
   * - ``cpu_percent``
     - Share of one processor, where the kernel can limit a share. 100 is one core.
   * - ``processes``
     - Processes and threads the tree may create. On macOS threads count too.
   * - ``file_size_bytes``
     - Largest file the process may create or extend.
   * - ``open_files``
     - File descriptors the process may hold open.
   * - ``core_dumps``
     - Off by default: a core file of a large process is large and holds what it read.

A limit is a request, not a promise. What the kernel can do about it depends on
the platform, and the sandbox uses the strongest mechanism the machine has:

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Limit
     - macOS
     - Linux
   * - ``memory_bytes``
     - Sampling watchdog. The kernel has no memory limit to set: macOS rejects the POSIX address-space limit, and there are no cgroups.
     - cgroup ``memory.max`` when the controllers are delegated to you; otherwise the POSIX address-space limit, per process.
   * - ``cpu_seconds``
     - POSIX limit (``RLIMIT_CPU``), inherited by the tree.
     - Same.
   * - ``cpu_percent``
     - Not available; nothing limits a share of a processor.
     - cgroup ``cpu.max``, a quota per 100 ms period.
   * - ``processes``
     - POSIX limit (``RLIMIT_NPROC``), counted per user, threads included.
     - cgroup ``pids.max`` when available, otherwise ``RLIMIT_NPROC``.
   * - ``file_size_bytes``
     - POSIX limit (``RLIMIT_FSIZE``).
     - Same.
   * - ``open_files``
     - POSIX limit (``RLIMIT_NOFILE``).
     - Same.
   * - ``core_dumps``
     - POSIX limit (``RLIMIT_CORE``).
     - Same.

Windows enforces none of these, so a policy that asks for them gets warnings
saying so. On every platform a limit that cannot be enforced is named in
``result.resources.warnings`` rather than quietly skipped, and nothing is ever
reported as enforced when it was not.

Ask before you run
----------------------------------------

:meth:`~zett_agent.sandbox.base.Sandbox.capabilities` returns how each limit
*would* be enforced here, without creating or changing anything. A host can refuse
a policy this machine cannot enforce as written instead of finding out from a run:

.. code-block:: python

   report = sb.capabilities()  # or sandbox.capabilities() on the lower-level API
   if report.memory not in ("cgroup", "rlimit_as"):
       raise RuntimeError(report.warnings)

   result = await sandbox.run(["python3", "build.py"], cwd="/work/project")
   print(result.resources.mechanisms, result.resources.peak_memory_bytes, result.resources.limits_hit)

``result.resources`` also reports what the run measured: the highest memory seen,
and which limits the process reached (``memory``, ``cpu`` or ``file_size``).

.. _memory-strategy:

Choosing how memory is limited
----------------------------------------

No single mechanism suits every program, so ``memory_bytes`` takes a strategy.
Pass it when you build the sandbox:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Strategy
     - Behaviour
   * - ``MemoryStrategy.AUTO``
     - The default: cgroup, then the address-space limit, then the watchdog.
   * - ``MemoryStrategy.KERNEL``
     - Only a kernel limit, and a warning where none exists. Exact, but an address-space limit counts memory a program *maps* rather than uses, so a virtual machine or the JVM inside the sandbox can be refused memory it never touches.
   * - ``MemoryStrategy.WATCHDOG``
     - Only the watchdog. Start-up is never refused and the limit can be overshot by whatever is allocated between two samples (0.25 s by default, one ``ps`` per sample).
   * - ``MemoryStrategy.NONE``
     - Enforce nothing, and say so.

.. code-block:: python

   from zett_agent.sandbox.resources import MemoryStrategy
   from zett_agent.sandbox.session import SandboxSession

   strongest = create_sandbox(policy)  # whatever this machine does best
   exact = create_sandbox(policy, memory_strategy=MemoryStrategy.KERNEL)
   generous = SandboxSession(memory="512MB", memory_strategy="watchdog")

.. warning::
   A resource limit protects the machine from a process that runs away, which is
   the usual failure of generated code. It is not a defense against a process that
   is trying to escape it: the watchdog can be overshot between samples, and a
   process can use everything it is allowed as fast as it likes.

What is not here
----------------------------------------

* **No total disk quota.** Writes are confined to the policy's paths and a single
  file can be capped, but nothing bounds the total size of a workspace.
* **No bandwidth limit**, and no limit on *which* hosts the network may reach: see
  the note above about proxies.
* **No per-limit grace or priority**: a limit is a hard line, and reaching it
  ends the process.

What each platform provides
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Platform
     - Backend
     - Notes
   * - macOS
     - :class:`~zett_agent.sandbox.seatbelt.SeatbeltSandbox`
     - Built in; nothing to install. Uses ``sandbox-exec``, which Apple has deprecated but ships.
   * - Linux
     - :class:`~zett_agent.sandbox.bubblewrap.BubblewrapSandbox`
     - Install ``bubblewrap``. Needs unprivileged user namespaces.
   * - Windows
     - :class:`~zett_agent.sandbox.passthrough.PassthroughSandbox`
     - No isolation and no resource limits: Windows has neither rlimits nor cgroups. The interface is the same, ``isolated`` says ``False``, and every limit asked for is reported as unenforced.

:func:`~zett_agent.sandbox.factory.create_sandbox` picks the platform's own.

.. important::

   A backend that exists but cannot run raises
   :class:`~zett_agent.sandbox.errors.SandboxUnavailableError` with what to
   install or change. The sandbox layer never answers that by running the command
   unsandboxed. The one case that degrades is a platform with no backend at all:
   ``create_sandbox`` returns a ``PassthroughSandbox`` and emits a
   ``SandboxWarning``. Pass ``require_isolation=True`` to make that an error too.

Spawn it yourself
----------------------------------------

``run`` is built on ``wrap``, which rewrites a command to run inside the sandbox
and returns it without running anything. A host that manages its own processes,
for streaming output or a pty, spawns the result itself:

.. code-block:: python

   command = sandbox.wrap(["make", "test"], cwd="/work/project")
   process = await asyncio.create_subprocess_exec(
       *command.argv, cwd=command.cwd, env=dict(command.env), start_new_session=True
   )

Start it in its own process group so the whole tree can be stopped.

What ``run`` adds around the process
----------------------------------------

Every backend shares this, so code behaves the same except in what it can confine:

* A time limit (60 seconds by default, ``None`` for none). When it passes, the
  whole process group is killed.
* A cap on the output kept from each stream (1 MiB by default). The rest is read
  and discarded, so a process that prints forever can neither fill memory nor
  block on a full pipe.
* A command that finishes does not leave helpers behind. Background children are
  stopped with it, and a child that still holds the output pipe cannot keep a
  finished run waiting.
* Cancelling the awaiting task stops the process group before the cancellation
  propagates.

What it does not do
----------------------------------------

Be clear about these before relying on it:

* **No filtering by host.** The network is all or nothing. Allowing named
  domains needs a proxy, which this does not include.
* **No seccomp on Linux.** With the network open the process shares the host
  network namespace and can reach local services there. Closing it (the default)
  isolates them.
* **No total disk quota and no bandwidth limit**, as above.
* **macOS network and sockets.** ``NetworkMode.FULL`` opens everything,
  including local services. ``NONE`` closes it all.
* **A protected path that does not exist yet.** macOS protects it from
  creation. On Linux a path can only be protected once it exists.
* **Denials are silent.** A blocked operation shows up as the command's own
  "Operation not permitted". The sandbox does not report violations.
* **Nesting.** A process already inside a Seatbelt sandbox cannot start another;
  ``check`` says so.
* **Windows.** No isolation at all.
* **Not a defense against a kernel or sandbox vulnerability,** or against a
  process that misuses the access you did allow.

On Linux the working directory must be visible inside the sandbox. ``/tmp`` is a
private tmpfs there, so a directory under ``/tmp`` is visible only if it is, or
is inside, a writable path.

Next: :doc:`tools` for the shell tool this protects, or
:doc:`../extending/built-ins` for the approval extension.

Related API: :class:`~zett_agent.sandbox.session.SandboxSession`,
:class:`~zett_agent.sandbox.base.Sandbox`,
:class:`~zett_agent.sandbox.policy.SandboxPolicy`,
:class:`~zett_agent.sandbox.policy.ResourceLimits`,
:class:`~zett_agent.sandbox.resources.ResourceReport`,
:class:`~zett_agent.sandbox.resources.MemoryStrategy`,
:func:`~zett_agent.sandbox.factory.create_sandbox`.
