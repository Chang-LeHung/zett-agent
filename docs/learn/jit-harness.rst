Build and reuse JIT capabilities
========================================

A conversation can need more than a fixed toolbox. ``JITHarnessExtension`` gives
a trusted model a repository: a Python package in a directory on disk. The model
writes tools and Markdown notes into it with the file tools it already has,
and the harness imports the package and registers what it finds. Because the
repository is ordinary files, it survives the process, and the next session
finds it again.

JIT means building the tool environment when a task needs it, not a Python
just-in-time compiler. This feature uses ordinary function calling on both
Responses and Chat Completions, and it adds only four tools to the request.

.. warning::

   Generated Python runs with your application's privileges. Imports and
   module-level statements execute when the model calls ``repo_load``, before
   any tool call. The default host imports the package into your process, and is
   not a sandbox. Enable it only when trusted code execution is appropriate, and
   review a repository before loading one that came from elsewhere. To keep a
   crash, a hang, or a runaway allocation out of your process, pass a
   :class:`~zett_agent.jit_harness.host.SubprocessToolHost`; see
   :ref:`where-the-code-runs`.

Enable a session-local repository
---------------------------------

Give the extension a directory that holds one subdirectory per session, and the
model a way to write files there:

.. code-block:: python

   from pathlib import Path

   from zett_agent.agent import AgentRunConfig
   from zett_agent.client import create_agent
   from zett_agent.extensions.memory import InMemoryMessageAccumulator
   from zett_agent.jit_harness import JITHarnessExtension

   jit = JITHarnessExtension(Path.home() / ".myapp" / "jit_harness")
   client = await create_agent(
       model,
       config=AgentRunConfig(session_id="builder"),
       tools=[add, write_file],
       extensions=[InMemoryMessageAccumulator(), jit],
   )
   reply = await client.run("Build and use a reusable tool that doubles an integer.")

Run this inside an async function. ``model`` is a configured function-calling
provider, ``add`` is your typed base tool, and ``write_file`` is any file tool
that can reach the directory. The harness never writes source for the model: it
can only run what the model, or your application, put on disk. The extension
supplies current tool guidance, so an additional ``ToolGuidelinesExtension`` is
not necessary. An explicit extension list replaces optional defaults; include
history as above or use :doc:`session persistence <sessions>`.

Each session's repository lives in ``<root>/<session id>/<repo>/``.
``jit.session_directory(session_id)`` returns the session part. A conventional
session id is the directory name; one with a path separator, a leading dot, or a
very long id is turned into a readable stem plus a digest, so two sessions never
share a directory and none can leave the root.

The model's four tools
----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Tool
     - Behavior
   * - ``repo_create(repo)``
     - Create the session's repository, a package with an empty ``__init__.py``, and return its ``path``. Asking again returns the existing one.
   * - ``repo_load()``
     - Load the package and list what it provides: the names of its tools, and each Markdown note with its kind, description and path. Call it after each edit, or first in a resumed session. Calling it again changes nothing unless a Python file changed; see :ref:`jit-load`.
   * - ``repo_unload()``
     - Hide every tool. The files, and the notes, stay on disk until the next ``repo_load``.
   * - ``profile_tools(names=None)``
     - Show how tools have done, the application's as well as the model's own: calls, success rate, mean time, mean output size, and recent errors. See :ref:`jit-profile-tool`.

All four use the runtime's serial scheduling mode and are reserved names. There
is one repository per session: a second ``repo_create`` with another name is an
error, so the model keeps everything it learns in one place. Each answers with
the repository's name, absolute path, state (``created``, ``loaded`` or
``unloaded``), and the names it provides.

Write a tool
------------

A tool is a top-level ``@tool`` function in any module of the package. A module
can define several, helper modules are fine, and a tool imported from elsewhere
is not registered:

.. literalinclude:: ../_examples/jit_harness.py
   :language: python
   :start-after: SOURCE =
   :end-before: '''

The tool contract is the same as :doc:`tools`: annotated arguments, a
description, and non-empty guidelines. JIT tools are ordinary functions, not
deferred or client-side search tools. Names are 1–64 letters, digits,
underscores, or hyphens; they cannot repeat, shadow a base tool, or reuse a
reserved name.

``harness`` is already in every module's globals, so a module does not import it.
It is deliberately not a builtin and not importable: only modules the harness
loads see it. Split a large tool into modules and import siblings relatively
(``from .steps import value``); a subdirectory needs an ``__init__.py`` to be a
subpackage, and a plain directory is a namespace package. A repository holds at
most 64 modules of 200,000 bytes each.

Loading imports the package under a throwaway name, so a repository called
``json`` cannot shadow the standard library. Nothing is cached as bytecode, so
two quick edits of the same size are never served stale. The package stays
imported while it is loaded, which means a tool may import a sibling lazily
inside its body too.

.. _where-the-code-runs:

Where the code runs
-------------------

A repository's Python runs in one of two hosts. Both register the same tools and
answer ``harness.call_tool`` the same way; they differ in where the code lives.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Host
     - What it isolates
     - What it costs
   * - :class:`~zett_agent.jit_harness.host.InProcessToolHost` (default)
     - Nothing. A tool is an ordinary call in your process.
     - A tool may return any Python value, and a tool cannot crash the run.
   * - :class:`~zett_agent.jit_harness.host.SubprocessToolHost`
     - A crash, an exit, a hang, and memory growth stay in the child, which is
       killed at ``call_timeout``.
     - Tool results cross a pipe: JSON values only, ``call_timeout`` adds a limit
       to slow tools, and the child starts slower than an import.

A child does not run as a different user. It is not a security boundary by
itself; pass a sandbox to bound what it may read, write, and connect to::

    from zett_agent.jit_harness import JITHarnessExtension, SubprocessToolHost
    from zett_agent.sandbox import NetworkMode, SandboxPolicy, create_sandbox

    policy = SandboxPolicy(network=NetworkMode.DISABLED)
    jit = JITHarnessExtension(
        Path.home() / ".myapp" / "jit",
        host_factory=lambda: SubprocessToolHost(sandbox=create_sandbox(policy), call_timeout=60),
    )

The host is built once per session. Inside the child, ``harness.call_tool`` and
``harness.call_tool_sync`` become frames to the agent, which runs the loaded tool
there. Nested calls therefore still pass through extensions, events, and
profiling, and a repository tool can compose base tools exactly as it does in
process. Two limits follow from the pipe:

* A tool returns ``None``, booleans, numbers, strings, lists, and dictionaries of
  those. Anything else raises :class:`~zett_agent.jit_harness.host.ToolNotSerializable`
  rather than being turned into a string.
* ``print`` in a tool goes to standard error, so it never corrupts the protocol.

If the child exits, the call raises
:class:`~zett_agent.jit_harness.host.JITHostCrashed` and the next ``repo_load``
starts a fresh one, even when no file changed. A tool that runs longer than
``call_timeout`` raises :class:`~zett_agent.jit_harness.host.JITHostTimeout` and
the child is killed.

A child is created by ``repo_load``, never by a tool call, and it stays alive
between turns so module state survives. Reloading after a source change starts a
new interpreter rather than importing again into the old one, so the previous
version's modules and their globals are gone. Closing the session
(``forget_session``) or exceeding ``max_sessions`` shuts the child down, and the
kill sweeps its whole process group, so a process a tool started does not outlive
it. ``JITHarnessExtension(root, max_sessions=8)`` is the default: a new session
releases the least recently used session that has no request in flight.

.. _jit-load:

Load, and load again
--------------------

``repo_load`` is safe to call whenever the model likes, and it always answers
with the tool names and the notes. What it does depends on what is on disk:

* **Not loaded yet.** It imports every module and registers each tool.
* **Loaded, and no Python file changed.** It imports nothing again. Importing
  would only run the module bodies a second time, with their side effects. Tools
  hidden by ``repo_unload`` become callable again, so calling it twice leaves the
  same state as calling it once.
* **Loaded, and a Python file changed.** It imports the whole package afresh and
  replaces the previous version in one step, so an edit to any file takes effect
  together with every other.

Changes are found by comparing the content of every Python file, the package
root included, so an edit of the same length is seen and a file that was added or
removed counts too. Notes are read again on every call and never need importing.
If anything fails, the previous version stays registered and the error names the
file and line, relative to the repository:

.. code-block:: text

   twice.py:3 in <module>: value = 1 / 0
   ZeroDivisionError: division by zero

A failed load always raises ``ValueError``, whatever went wrong inside; the model
reads the location and repairs the file with its own file tool. Module bodies and
imports run on load, so their external side effects cannot be rolled back.
Executions already started keep their original implementation; later calls use
the new one. Loading also clears the unloaded state of the repository.

Compose with the harness
------------------------

Use ``await harness.call_tool(name, **arguments)`` in async tools. For ordinary
``def`` functions, use the blocking composition entry point:

.. code-block:: python

   @tool(guidelines="Use to add an integer to itself.")
   def twice(value: int) -> int:
       """Add the same integer twice."""
       return harness.call_tool_sync("add", left=value, right=value)

Both paths check the current registration and execute parameter validation,
middleware, and necessary tool hooks. Dependencies return Python values, not
JSON strings, unless a result hook deliberately transforms them. Exceptions
propagate to the outer tool unless your Python catches them. Only the outer
call's result is recorded in the model conversation; no nested execution trace
is emitted. A dependency that requests user input or approval still performs
that interaction.

Synchronous code runs off the event loop. Do not call ``call_tool_sync`` from
async code, or use it outside a running synchronous tool. Composition depth is
bounded by ``max_call_depth`` (32 by default). Cancellation cannot forcibly
stop arbitrary synchronous Python or undo side effects; give blocking I/O
its own timeouts and avoid detached background work. While a module body runs,
it cannot dispatch tools or change the harness.

Composition inside a tool is programmatic tool calling: the model writes code
that calls tools, instead of asking for one call per model turn. A single outer
call can replace dozens of turns, nested results stay in Python and never reach
the conversation, and every nested call still reaches profiling and events.
Dependencies must be loaded, hooks and argument validation still apply, and the
returned value should be JSON-friendly because it is what the model reads.

Save a note
-----------

When a task takes many exploratory steps, the guidance asks the model to consider
keeping what worked. Choose a tool when the steps are fixed, and a Markdown
note when the path still needs judgment. A note is a Markdown file, similar to a
``SKILL``, at ``resources/<name>.md``:

.. code-block:: markdown

   ---
   kind: workflow
   description: Use when a test fails and the cause is not yet known.
   ---
   # Investigate a failing test

   1. Reproduce the smallest failing test with the available test runner.
   2. Read the failing assertion and the relevant source, not the whole repository.
   3. Verify the diagnosis by rerunning that test and related regression tests.

The front matter takes ``description``, a hint of up to 512 characters, and
``kind``, which says what the text is for and how the model should treat it:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``kind``
     - Meaning
   * - ``workflow``
     - An ordered procedure that worked, followed step by step once the model has
       checked that it applies. The default when ``kind`` is omitted.
   * - ``skill``
     - Know-how for a tool, library, or technique: how to use it well and which
       mistakes to avoid. Applied where it fits.
   * - ``instruction``
     - A standing rule or preference to honor whenever its description matches the
       work at hand.
   * - ``memory``
     - Distilled facts and key points from earlier work, such as findings,
       decisions, and locations. It can be out of date, so the model verifies it
       before relying on it.

A note is not a tool. ``repo_load`` lists each one as a
:class:`~zett_agent.jit_harness.results.ResourceInfo`, which has no body:

.. code-block:: json

   {"repo": "kit", "path": "/data/jit/s1/kit", "status": "loaded",
    "tools": ["sum_doubled"],
    "resources": [
      {"name": "triage", "kind": "workflow",
       "description": "Use when a test fails and the cause is not yet known.",
       "path": "/data/jit/s1/kit/resources/triage.md"}]}

The model opens the file with its own file tools when a description fits the
task, and reads only what it needs. Nothing about the body is registered or
injected into a request, and the harness never reads it past the front matter.
Follow a workflow only after checking that it applies and is safe: it cannot
grant permissions or make an unloaded dependency available. A note may share a
name with a tool, because only one of them is a tool. Every description is
repeated in each ``repo_load`` answer, so a repository lists at most 50 notes;
more fails the load, and so does an unknown ``kind`` (the error lists the valid
ones) or a note with no front matter.

Validate a proposed optimization against representative tasks and account for
the cost of creating and testing it. Guidance encourages this behavior; it does
not guarantee that a model will learn, or that a saved path is globally optimal.

Resume a session
----------------

The files are the saved state. The harness keeps nothing else: there is no
snapshot to write, and module globals, profiles, and recent errors are not
saved. When a session's first request finds exactly one repository in its
directory, the extension adopts it, loads it, and tells the model its name and
path in the guidance, so a resumed session has its tools without asking. A
repository that fails to load is still adopted, so ``repo_load`` can show the
model the error. Several candidate directories are not guessed between.

Release an idle session with ``jit.forget_session(session_id)``. It removes the
repository's modules and import hook from the interpreter and keeps the files.
``harness.close()`` does the same for a standalone harness. Back up, move, or
delete a repository like any directory; deleting the session directory forgets
what the model learned.

Choose whether to manage base tools
-----------------------------------

``manage_base_tools=True`` (the default) lets the harness unload/reload and profile
application-owned local tools, including tools supplied by extensions. Tools
registered later during setup or replaced by a policy hook are adopted before
model/tool dispatch. Synchronization does not restore tools removed by a policy.

To protect the base toolbox and opt out of its profiling:

.. code-block:: python

   jit = JITHarnessExtension(root, manage_base_tools=False)
   # The standalone JITHarness constructor accepts the same option.

In this mode base tools keep their normal request handlers, ``unload_tool`` and
``load_tool`` reject base names, and no base-call measurements or recent base
errors are collected. Repository tools can still compose currently registered base
tools, with validation, middleware, and hooks intact, and keep their own profiling.
A parent tool's elapsed time still includes time spent in base dependencies.
Calling an original application ``AgentTool`` directly bypasses the harness, so it
cannot be counted or prevented by unloading a request registration.

The Python API
--------------------------------------

``JITHarness`` is the same toolbox without the model: ``JITHarness(directory,
tools=[...])`` with ``create_repo(name)``, ``load_repo()``, ``unload_repo()`` and
``close()``, plus ``repo`` and ``repo_path``. ``unload_tool(name)`` and
``load_tool(name)`` hide or restore one tool, base or repository, from your own
code; the model has no tool for them. ``list_tools()`` and ``describe(name)`` read
a compact catalog of names, origins, states, and descriptions. Measurements stay
on their own methods: ``profile_stats(name)``, ``profile(name)``, and
``tool_errors(name)``.

.. code-block:: python

   from pathlib import Path

   from zett_agent.jit_harness import JITHarness

   harness = JITHarness("jit", tools=[add])
   repo = Path(harness.create_repo("kit")["path"])
   (repo / "twice.py").write_text(source)
   harness.load_repo()
   print(await harness.call_tool("twice", value=21))

Use profiling to build useful tools
-----------------------------------

Every harness records session-local measurements for base, repository, and
management tools. Guidance asks the model to complete the user's task with minimal
total token usage and end-to-end time, while preserving correctness,
completeness, safety, and answer quality, and to count the costs of creating and
testing a tool in that trade-off. To make that trade-off the model can read the
summary below through ``profile_tools``; your application reads the full
measurements.

.. _jit-profile-tool:

What the model sees
~~~~~~~~~~~~~~~~~~~

``profile_tools(names)`` takes the names of up to 10 tools and answers with one
row for each, in the order asked. It works for any tool, the application's as well
as the ones the model wrote, so the model can compare a tool it would replace with
its own. A tool that was never called gets a row of zeros: unused is an answer.
Notes are not tools and have no row.

Without ``names`` it answers with an overview: every tool of the repository,
called or not, and every application tool that has been called, most called first,
at most 20 rows; ``omitted`` counts the rest. The management tools are never
measured, and asking for one is an error.

.. code-block:: json

   {"tools": [
     {"name": "sum_doubled", "origin": "python", "status": "loaded", "calls": 12,
      "success_rate": 0.917, "failed": 1, "avg_ms": 4.2, "avg_output_chars": 36,
      "recent_errors": ["ValueError: values must be integers"]},
     {"name": "summarize", "origin": "python", "status": "loaded", "calls": 0,
      "success_rate": null, "failed": 0, "avg_ms": null, "avg_output_chars": null,
      "recent_errors": []}
   ], "omitted": 0}

``success_rate`` is completed over completed plus failed; rejected and cancelled
attempts say nothing about the tool and are left out. ``avg_output_chars`` is what
a call handed back, and the message of a failed call counts because the model
reads it too. A tool asked for by name carries up to three recent distinct errors,
which is where to look before fixing it; in the overview each carries one. The
numbers live in memory and start again from zero with the process, and a few calls
are thin evidence, which the guidance says. Application tools are measured only
with ``manage_base_tools=True``; with it off, asking for one is an error rather
than a row of zeros that would read as unused. The same summary is
:meth:`~zett_agent.jit_harness.runtime.JITHarness.profile_tools` in Python.

What your application sees
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   harness = jit.get_harness("builder")
   report = harness.profile("twice", offset=0, limit=100)
   stats = report.stats[0]
   print(stats.call_count, stats.average_duration_ms)
   print(stats.average_input_chars, stats.average_output_chars)
   for call in report.calls:
       print(call.call_id, call.caller_tool_name, call.status, call.duration_ms)

Aggregate statistics count finished calls per outcome plus mean duration and
mean input and output characters. Each finished record carries its invocation
id, tool name, origin, outcome, elapsed time, measured sizes, the composing
caller when the call was nested, and an error type when it failed. Lifetime
totals are not reset by a reload or unload. Look for repeated caller/dependency
relationships a focused tool could compose, large outputs that can be reduced,
and slow or failure-prone implementations. Profiling is evidence about
execution, not a quality score.

Every record also carries a :class:`~zett_agent.jit_harness.enums.ToolOrigin`:

* ``base`` — an application-owned tool from ``tools=[...]`` or a tool-providing
  extension. It can be unloaded and profiled when base management is enabled.
* ``python`` — a tool the model defined in its repository with ``@tool``. An
  unloaded tool keeps ``python``; its call status becomes ``rejected``.
* ``text`` — a Markdown resource of any kind. Calls measure reading a page, not
  acting on its content.
* ``management`` — ``repo_create``, ``repo_load``, ``repo_unload``, or ``profile_tools``.
* ``unknown`` — a defensive fallback for a name that was in no registry when the
  attempt was recorded. Normal dispatch does not produce it.

The ``status`` field is a :class:`~zett_agent.jit_harness.enums.ToolCallStatus`:

* ``completed`` — the handler returned. A later serialization or lifecycle-hook
  failure does not change the record, so ``completed`` means the tool itself ran.
* ``failed`` — the tool started and raised: argument validation, handler code, or
  an uncaught nested dependency error.
* ``cancelled`` — the awaiting caller was cancelled. A synchronous worker may
  still be running, because Python cannot forcibly stop it.
* ``rejected`` — the harness refused the call before any handler body ran: the
  tool is unloaded, missing its base binding, no longer registered in the current
  request, or the nested call exceeded ``max_call_depth``.

Use these measurement rules when interpreting results:

* Input length is compact Unicode JSON, including property names and punctuation.
  Text output uses plain string length; structured output uses compact JSON.
  Multimodal output counts text blocks only, excluding image bytes and URLs.
  These are characters, not bytes, tokens, or exact provider payload sizes.
* A missing measurement is ``None``, not zero. Character averages use measured
  samples only. Error output counts the error-message length; cancellation has
  no output measurement.
* Measurements stop at 8,000 characters, so a reported 8,000 means "at least
  this many".
* Duration is elapsed dispatch time, including validation, synchronous worker
  dispatch, and nested calls. Surrounding middleware, lifecycle hooks, and user
  approval waits are outside the boundary. A parent includes its children, so
  summing both double-counts nested work.
* A middleware short-circuit is not counted, because the tool never executed.
  Unknown names, stale request references, and rejected module-loading calls
  also create no record.

The profile retains a bounded window of finished attempts in memory, not their
input/output contents or exception messages: the newest ``call_capacity``
records (2,000 by default) survive, oldest dispatch first, while lifetime
aggregates keep counting every attempt. Profiles survive later requests in the
same in-memory harness but are not saved: a new process starts fresh. Export a
page with ``report.model_dump_json()`` and read it back with
``JITHarnessProfile.model_validate_json(...)``.

Diagnose recent errors
----------------------

Each session has a separate LRU cache of distinct errors, read by your
application with ``harness.tool_errors``. The defaults retain 100 entries and at
most 4,000 characters of each message:

.. code-block:: python

   jit = JITHarnessExtension(root, error_capacity=100, error_message_chars=4000)
   # After a request:
   report = jit.get_harness("builder").tool_errors("twice", limit=20)
   for error in report.errors:
       print(error.error_type, error.message, error.occurrences)
       print(error.last_call_id, error.caller_tool_name)

Both configuration values must be positive integers. Capacity is global to one
harness, not a separate allowance for every tool. Errors deduplicate by tool name,
exception type, and complete error message. A repeated error increments
``occurrences`` and moves that entry to the most recent position; when the cache
is full, the least recently occurring entry is evicted. Reads do not refresh
recency, and successful calls do not clear old evidence.

Records include the error type, bounded message, last occurrence time, count, and
the latest invocation's caller and call ID. ``message_truncated`` indicates
whether only a prefix was retained. ``occurrences`` covers the entry's current
cache residence; lifetime failed/rejected counts remain available through
profiling. An empty report is not proof that a tool never failed. Cancellations
are recorded in profiling rather than the error cache. A failed ``repo_load`` is
retained too. ``harness.tool_errors`` follows ``next_offset`` and takes ``limit``
from 1 to 1000, defaulting to 5.

.. warning::

   Exception messages may include sensitive application data. Arguments,
   tracebacks, and exception objects are not stored, but the error text itself
   is retained. Use applications and tools that redact secrets in their error
   messages.

Observe repository and tool calls
---------------------------------

``JITHarnessExtension`` publishes two CUSTOM events so a UI or log can follow the
harness as it works, including calls that never surface as ``TOOL_STARTED``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Event name
     - Emitted when
   * - ``jit_tool_started``
     - A repository tool or ``repo_*`` call is about to run.
   * - ``jit_tool_completed``
     - That call returned or raised, adding ``success`` and ``error_type``.

Both events fire for calls the model requested and for calls one repository tool
composes through ``harness.call_tool``, so a composed dependency is visible even
though its result never reaches the conversation. Application base tools are
skipped: the runtime already publishes ``TOOL_STARTED`` and ``TOOL_COMPLETED``
when the model requests them.

.. code-block:: python

   from zett_agent.events import AgentEventType

   async for event in client.stream("Build and use a helper tool"):
       if event.type is AgentEventType.CUSTOM and event.name == "jit_tool_started":
           print("started", event.payload["tool_name"], event.payload["origin"])
       elif event.type is AgentEventType.CUSTOM and event.name == "jit_tool_completed":
           print("finished", event.payload["tool_name"], event.payload["success"])

Each payload carries ``session_id``, ``tool_call_id``, ``tool_name``, and
``origin``; the completed event adds ``success`` and ``error_type``. The event
stays small on purpose: arguments, results, and error messages are not copied
into it. ``tool_call_id`` is the join key, matching both the conversation's
``ToolCall`` and the ``call_id`` of the record in
:meth:`~zett_agent.jit_harness.runtime.JITHarness.profile`.

Events pair up for calls that reach dispatch. A call the harness rejects before
running any handler emits no pair of its own; read ``profile()`` for its
``rejected`` record and ``tool_errors()`` for the message. Names this harness has
never seen are ignored rather than reported as JIT calls.

Try the complete offline example
--------------------------------

The scripted model creates a repository, writes ``twice.py`` through a file tool,
loads it, and invokes ``twice``. A second extension with the same root and
session id then adopts the files and uses the tool again. The program also checks
a synchronous → asynchronous → base-tool pipeline, empty inputs, a failed reload
that keeps the previous version, dependency unload, a listed note, and a restart
from the files alone. It uses a temporary directory and makes no network
requests:

.. literalinclude:: ../_examples/jit_harness.py
   :language: python
   :pyobject: main

.. code-block:: console

   uv run python docs/_examples/jit_harness.py

Expected output:

.. code-block:: text

   JIT tool created and reused: 42
   JIT invariants checked: composition, failed reload, unload, restart from files
   JIT profiling checked: per-tool totals and nested call records
   JIT diagnostics checked: recent dependency error retained

Download :download:`the complete program <../_examples/jit_harness.py>`.
For exact contracts, see :class:`~zett_agent.jit_harness.runtime.JITHarness` and
:class:`~zett_agent.jit_harness.extension.JITHarnessExtension`.
