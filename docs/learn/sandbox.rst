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

Run a command
----------------------------------------

.. code-block:: python

   from zett_agent.sandbox.factory import create_sandbox
   from zett_agent.sandbox.policy import SandboxPolicy

   sandbox = create_sandbox(SandboxPolicy.workspace_write("/work/project"))

   result = await sandbox.run(["python3", "build.py"], cwd="/work/project")
   if not result.succeeded:
       print(result.stderr)

   listing = await sandbox.run("ls | sort", cwd="/work/project")

Every backend returns the same :class:`~zett_agent.sandbox.base.SandboxResult`:
the exit status, standard output and error, whether it timed out, whether output
was cut off, and whether the backend isolated anything. A list runs the program
directly with nothing parsed, which is the safer form; a string runs through
``/bin/sh -c``, so pipes work.

:download:`Download sandbox.py <../_examples/sandbox.py>` for a complete program.

.. literalinclude:: ../_examples/sandbox.py
   :language: python
   :pyobject: main

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
     - No isolation. The interface is the same, and ``isolated`` says ``False``.

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
* **No resource limits.** Memory, CPU time and process counts are not capped.
  The time limit and the output cap are the only limits.
* **No seccomp on Linux.** With the network open the process shares the host
  network namespace and can reach local services there. Closing it (the default)
  isolates them.
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

Related API: :class:`~zett_agent.sandbox.base.Sandbox`,
:class:`~zett_agent.sandbox.policy.SandboxPolicy`,
:func:`~zett_agent.sandbox.factory.create_sandbox`.
