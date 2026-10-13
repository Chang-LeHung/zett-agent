Build and reuse JIT capabilities
========================================

A conversation can need more than a fixed toolbox. ``JITHarnessExtension`` lets
a trusted model write a reusable Python tool or Markdown workflow, compose
existing tools, improve its capabilities, and unload ones it no longer needs. Your
application can save that toolbox and start a new session from it.

JIT means building the tool environment when a task needs it, not a Python
just-in-time compiler. This feature uses ordinary function calling on both
Responses and Chat Completions. It does not require native ``tool_search``,
though a Responses endpoint can defer the inspection tools described in
`Control what the model sees`_.

.. warning::

   Generated Python runs with your application's privileges. Imports and
   module-level statements execute when source is activated, before any tool
   call. This is not a sandbox. Enable it only when trusted code execution is
   appropriate, and review snapshots before activating source from elsewhere.

Enable a session-local toolbox
------------------------------

Provide a model and the base tools your application owns:

.. code-block:: python

   from zett_agent.agent import AgentRunConfig
   from zett_agent.client import create_agent
   from zett_agent.extensions.memory import InMemoryMessageAccumulator
   from zett_agent.jit_harness import JITHarnessExtension

   jit = JITHarnessExtension()
   client = await create_agent(
       model,
       config=AgentRunConfig(session_id="builder"),
       tools=[add],
       extensions=[InMemoryMessageAccumulator(), jit],
   )
   reply = await client.run("Build and use a reusable tool that doubles an integer.")

Run this inside an async function. ``model`` is a configured function-calling
provider and ``add`` is your typed base tool. The extension supplies current
tool guidance, so an additional ``ToolGuidelinesExtension`` is not necessary.
An explicit extension list replaces optional defaults; include history as above
or use :doc:`session persistence <sessions>`.

The model decides whether to create a tool. Creation does not test the tool
body: successful registration means the Python module loaded and the tool
definition validated, not that the implementation solves the task.

Learn from expensive workflows
----------------------------------------

When a task involves many exploratory steps or substantial time and tokens,
the harness guidance asks the model to consider retaining what worked.
Choose Python when stable steps can be automated; choose text when the path
still needs judgment, such as selecting relevant evidence or interpreting errors.
Neither form should become a dump of the conversation. Keep the useful path,
applicability, dependencies, known pitfalls, and outcome checks.

A text tool is a reusable Markdown procedure, similar to a ``SKILL``:

.. code-block:: python

   from zett_agent.jit_harness import JITHarness, ToolKind

   harness = JITHarness()
   harness.create_tool(
       kind=ToolKind.TEXT,
       name="investigate_test_failure",
       description="Use when a test fails and the cause is not yet known.",
       source="""# Investigate a failing test

   1. Reproduce the smallest failing test with the available test runner.
   2. Read the failing assertion and the relevant source, not the whole repository.
   3. Compare the expected behavior with the observed inputs and result.
   4. Verify the diagnosis by rerunning that test and related regression tests.

   Record a confirmed cause, not a guess. Do not copy credentials into reports.
   """,
   )
   page = await harness.call_tool("investigate_test_failure")
   print(page["content"])

The model uses the same ``create_tool`` with ``kind="text"``. ``name`` and a
short ``description`` are required; ``entrypoint`` must be omitted. Python
creation remains the default, so existing ``create_tool(source, entrypoint=...)``
calls keep working.

Loaded text tools expose a reader with ``offset=0`` and ``max_chars=2000``.
The model calls it to read relevant steps, then follows them with available tools.
Reading text does not execute Python or automatically run a workflow. The full
body is not injected into each model request: only the tool's selection hint,
reader schema, and guidance are offered. Long workflows use the same exact,
budgeted pages as source reads.

``update_tool(name, markdown, description=...)`` replaces a text workflow without
changing its kind or name; omitting ``description`` preserves its current hint.
Use ``read_tool_source`` to inspect the full path before editing.
Unloading hides the reader while retaining its content; loading restores it.
Both Python and text tools are saved in the application's snapshot for reuse
in later sessions. Learning remains in memory until the application saves it.

Validate a proposed optimization against representative tasks and account for
the cost of creating and testing it. Guidance encourages this behavior; it
does not guarantee that a model will learn, or that a saved path is globally optimal.

Choose whether to manage base tools
----------------------------------------

``manage_base_tools=True`` (the default) lets the harness unload/reload and profile
application-owned local tools, including tools supplied by extensions. Tools
registered later during setup or replaced by a policy hook are adopted before
model/tool dispatch. Synchronization does not restore tools removed by a policy.

To protect the base toolbox and opt out of its profiling:

.. code-block:: python

   jit = JITHarnessExtension(manage_base_tools=False)
   # The standalone JITHarness constructor accepts the same option.

In this mode base tools keep their normal request handlers. ``unload_tool`` and
``load_tool`` reject base names; no base-call measurements or recent base errors
are collected. Dynamic Python tools can still compose currently registered base
tools, with argument validation, middleware, and tool hooks intact. Dynamic and
management calls retain their own profiling. A parent tool's elapsed time still
includes time spent in base dependencies.

Base implementations cannot be updated or serialized as Python source in either
mode. A snapshot retains base names and unload state, and the application must
provide implementations again in each request. This option is not persisted in
snapshots: the receiving application chooses it. When loading a snapshot with
base management disabled, existing base unload marks are retained in the snapshot
but not enforced; dynamic unload marks are still enforced. Later loading that
snapshot with management enabled reactivates the retained base unload marks.

Profiling observes calls routed through the Agent/harness. Calling an original
application ``AgentTool`` directly bypasses the harness, so it cannot be counted
or prevented by unloading a request registration.

The model's management tools
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Tool
     - Behavior
   * - ``create_tool(source, ...)``
     - Create Python with ``entrypoint``, or text with ``kind="text"``, ``name``, and ``description``. ``files`` adds package modules.
   * - ``update_tool(name, ...)``
     - Replace or add files, source, or entry point of a dynamic tool; ``remove`` deletes a file. Everything sent compiles as one version.
   * - ``unload_tool(name)``
     - Hide the callable registration while retaining source and its name.
   * - ``load_tool(name)``
     - Restore a retained tool without regenerating code.
   * - ``list_tools(offset=0, limit=20)``
     - Read a small catalog page: names, states, and description previews only.
   * - ``inspect_tool(name)``
     - Read the definition, parameters, and package file sizes, without source, profiling, or history.
   * - ``read_tool_source(name, path=None, offset=0, max_chars=2000)``
     - Read an exact fragment of one package file, defaulting to the entry point module.
   * - ``read_tool_schema(name, offset=0, max_chars=2000)``
     - Read JSON Schema fragments if the definition reports ``parameters_omitted``.
   * - ``inspect_tool_stats(name)``
     - Read only that tool's lifetime aggregate measurements.
   * - ``inspect_tool_calls(name, offset=0, limit=5)``
     - Read compact single-call measurements without source or aggregates.
   * - ``inspect_tool_errors(name=None, offset=0, limit=3, max_chars=512)``
     - Read recent message previews and occurrence counts, optionally filtered by tool.

Management names are reserved. The model cannot overwrite or unload these
tools, and cannot replace application-provided base tools. All management
operations use the runtime's serial scheduling mode. If a response includes
ordinary parallel calls too, those finish before the serial management calls.

The last five read retained details and are the *inspection* tier:
``read_tool_source``, ``read_tool_schema``, ``inspect_tool_stats``,
``inspect_tool_calls``, and ``inspect_tool_errors``. The others are the control
loop that builds, changes, and lists capabilities. `Control what the model
sees`_ explains how to expose the inspection tier.

State-changing tools -- ``create_tool``, ``update_tool``, ``unload_tool``, and
``load_tool`` -- acknowledge only ``name`` and ``status``, for example
``{"name": "twice", "status": "loaded"}``. They do not return profiling, metadata,
source, or call history. Profiling is still recorded normally and can be queried
when useful through ``inspect_tool_stats``, ``inspect_tool_calls``, or the Python
``profile()`` API.

Each query has one purpose. Discovery does not fetch profiling; parameter
inspection does not fetch source or history. Model-facing query responses stay
within an 8,000-character serialized JSON budget, including escaping and
punctuation. ``limit`` is 1–100 and ``max_chars`` is 1–4,000. Both are maxima:
the budget can shorten a page, so always follow its ``next_offset``.

Catalog descriptions preview up to 200 characters; definition descriptions
preview up to 500. Source and schema fragments are exact, not summaries:
concatenate them in order and compare ``content_id`` between pages if the tool
changes during reading. An oversized schema is explicitly omitted from
``inspect_tool``, with ``parameters_omitted`` and ``parameters_chars``; use
``read_tool_schema`` to recover complete valid JSON. Base tools have no retained
Python source, so ``read_tool_source`` rejects them.

Application methods remain complete: ``harness.list_tools()`` returns the full
catalog, ``definition(name)`` returns full metadata and schema, ``source(name)``
returns full dynamic source, and ``inspect_tool(name)`` returns schema,
guidelines, and source. Measurements stay on their own methods:
``profile_stats(name)`` returns aggregates and ``profile(name)`` returns
individual call records. Those methods do not apply the model response budget.

Control what the model sees
----------------------------------------

Every definition and instruction is sent on every request. The inspection tier
is the part most sessions never need, so
``JITHarnessExtension(inspection_tools=...)`` decides how it reaches the model:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Mode
     - Behavior
   * - ``InspectionToolExposure.DEFERRED`` (default)
     - Mark the inspection definitions as deferred. A provider that supports
       deferred loading (OpenAI Responses) fetches one on demand through its
       hosted ``tool_search`` endpoint. Protocols that ignore the flag — Chat
       Completions, Anthropic, Google, and Ollama — still receive every
       definition, so no capability is lost there: the model can always read
       source, schemas, measurements, and error history.
   * - ``InspectionToolExposure.ALWAYS``
     - Send every management definition with the first request, exactly as if
       the tier did not exist.
   * - ``InspectionToolExposure.HIDDEN``
     - Never expose the inspection tools. The application keeps ``profile``,
       ``profile_stats``, ``tool_errors``, ``source``, and ``inspect_tool``,
       but the model cannot inspect code or measurements by itself.

The choice also shapes the prompt: hidden inspection tools are never named, and
an endpoint that may defer them is told that a search may be needed. Nothing
else changes; the tools keep their names, serial scheduling, and output budgets,
and the management names stay reserved in every mode.

Measured on the default toolbox with ``cl100k_base`` tokenization, one request
costs roughly:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Responses
     - Chat Completions
   * - ``ALWAYS``
     - 3,230 tokens
     - 3,230 tokens
   * - ``DEFERRED`` (default)
     - 2,370 tokens
     - 3,250 tokens
   * - ``HIDDEN``
     - 2,080 tokens
     - 2,080 tokens

Each figure covers the control-tool definitions plus the instruction block,
before tool guidance and your own tools. Deferring saves the inspection
definitions (about 880 tokens); hiding removes those definitions and their
instructions (about 290 tokens more). Keep the default when the model should be
able to diagnose its own tools, and hide when it should not. Hiding is also the
lever that matters on Chat Completions, where deferral has no effect because the
protocol always receives every definition.

Deferral depends on a provider feature, so it never changes Chat Completions
behaviour. Hiding is a pure application decision, and the harness API stays
complete in every mode.

These control definitions are larger than they were before package files and
programmatic guidance existed. Most of that size is the JSON Schema rendering
itself: every optional argument is written as an ``anyOf`` union and every
property carries a generated ``title``, which together account for roughly 280
tokens of the figures above across the eleven tools.

Write and compose tools
-----------------------

Python modules receive ``tool``, ``ToolExecutionMode``, and ``harness``. Source
can contain imports, helper functions, and classes. Pass the variable name of
one decorated entry point, for example ``entrypoint="twice"``:

.. literalinclude:: ../_examples/jit_harness.py
   :language: python
   :start-after: SOURCE =
   :end-before: '''

The tool contract is the same as :doc:`tools`: annotated arguments, a
description, and non-empty guidelines. JIT tools are ordinary functions, not
deferred or client-side search tools.

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
its own timeouts and avoid detached background work.

Programmatic tool calling
----------------------------------------

Composition inside a tool is programmatic tool calling: the model writes code
that calls tools, instead of asking for one tool call per model turn. A single
outer call can then replace dozens of turns.

.. code-block:: python

   from .batch import summarize

   @tool(guidelines="Use to summarize every requested file in one call.")
   async def summarize_files(paths: list[str]) -> str:
       """Read many files in code and return one compact summary."""
       return await summarize(harness, paths)

with ``batch.py`` holding the loop:

.. code-block:: python

   async def summarize(harness, paths):
       lines = []
       for path in paths:
           text = await harness.call_tool("read_file", path=path)
           lines.append(f"{path}: {len(text)} characters")
       return "\n".join(lines)

What it buys:

* **Fewer model turns.** The loop, the branching, and the retries happen in
  Python, so the conversation grows once instead of once per tool call.
* **Smaller context.** Nested results stay in Python. Only the outer tool's
  return value becomes a ``ToolMessage``; intermediate payloads are never sent
  back to the model, and no execution trace is appended.
* **Observability is unchanged.** Every nested call is measured by profiling
  and emits ``jit_tool_started`` / ``jit_tool_completed`` with its origin, so
  the application still sees what ran.

Limits to design around:

* Dependencies must be loaded, and depth is bounded by ``max_call_depth``.
* Tool hooks and middleware still run for nested calls, so a dependency that
  requests user input or approval still interrupts.
* Argument validation applies to nested calls, and exceptions propagate unless
  your Python catches them.
* Keep the returned value JSON-friendly: it is what the model reads.

Split a tool across modules
----------------------------------------

A large tool can be a small package instead of one long module. The entry point
takes the role of the package root, and additional files are reached with
relative imports:

.. code-block:: python

   jit.get_harness("builder").create_tool(
       source=entry_source,
       entrypoint="summarize_files",
       files={
           "batch.py": batch_source,
           "parsers/__init__.py": parser_source,
       },
   )

Nested directories become subpackages, so ``from .parsers import parse`` and
``from .parsers.lines import split`` both work. Path segments must be importable
Python names, ``__init__.py`` at the root is the entry point itself, and a
package holds at most 32 files of 200,000 characters each.

Reading and editing stay per file:

* ``inspect_tool(name)`` lists the package files with their character counts,
  entry point included as ``__init__.py``.
* ``read_tool_source(name, path="batch.py")`` pages one file; the default is the
  entry point.
* ``update_tool(name, files={...}, remove=[...])`` adds or replaces the files you
  send and deletes the ones you name. Files you leave out keep their content, so
  a small edit costs a small update.
* Everything in one call is validated and compiled as a single new version. A
  cross-file refactor therefore applies at once, and a failure such as a broken
  import leaves the previous version installed.

Two consequences matter. First, imports and module bodies run when the tool is
registered, exactly as a normal import would, so a ``SyntaxError`` or
``ImportError`` in a helper surfaces at creation or update time. Second, the
package is not retained in ``sys.modules`` afterwards, so import what a tool
needs at module level rather than lazily inside a function.

Packages survive ``unload_tool`` and snapshots. A snapshot that contains at
least one multi-file tool writes ``format_version: 2``; single-module tools keep
writing version 1, so existing files stay unchanged and remain readable.

Unload without losing source
----------------------------

``unload_tool`` removes a tool from the current callable definitions immediately
and from later requests in the same session. It does not erase earlier
conversation messages, revoke Python privileges, or terminate an already
started invocation. Dependent tools remain registered; calling an unloaded
dependency raises an explicit error. Loading restores the definition.

``unload_tool`` returns ``status="unloaded"``; ``load_tool`` returns
``status="loaded"``. Snapshots store hidden tool names in ``unloaded_tools``.

Updating an unloaded tool preserves its unloaded state. An update cannot
rename a tool. Existing invocations finish with the implementation they
started; subsequent calls use the validated replacement. Failed source loads
leave the old registration intact, but cannot roll back module-level side
effects such as file writes.

Use profiling to build useful tools
----------------------------------------

Every harness records session-local measurements for base, dynamic, and
management tools. The model reads aggregate counters through ``inspect_tool_stats``
and single-call measurements through ``inspect_tool_calls``. Guidance asks it to
complete the user's task with minimal total token usage and end-to-end time,
while preserving correctness, completeness, safety, and answer quality.
It encourages JIT optimization when useful, including the costs of inspection,
tool creation, and validation in the trade-off. Existing tools or a direct
solution remain preferable when they already serve the task efficiently.
Relevant measurements help the model decide what to improve and verify the result.

Look for repeated caller/dependency relationships that a focused tool could
compose, large outputs that can be reduced before they reach the model, and
slow or failure-prone implementations. Profiling is evidence about execution,
not a quality score: a rarely called tool may still be essential, and fewer
characters alone do not mean a better answer.

Application code can inspect the same data:

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
totals are not reset on update or unload.

Every record also carries a :class:`~zett_agent.jit_harness.enums.ToolOrigin`
naming which registry served the call:

* ``base`` — an application-owned tool from ``tools=[...]`` or a tool-providing
  extension. It can be unloaded/reloaded and profiled when base management is
  enabled. Its implementation has no retained source and cannot be updated or
  serialized; snapshots retain its name and unload state only.
* ``python`` — a dynamic tool the model created with ``create_tool`` or replaced
  with ``update_tool``. An unloaded dynamic tool keeps ``python``; its call
  status becomes ``rejected``.
* ``text`` — a saved Markdown workflow. Calls measure reading a workflow page,
  not executing its steps or the task's eventual quality.
* ``management`` — one of the reserved control functions such as ``create_tool``
  or ``inspect_tool``.
* ``unknown`` — a defensive fallback for a name that was in no registry when the
  attempt was recorded. Normal dispatch does not produce it.

``list_tools`` and ``inspect_tool`` use the same value for each tool's catalog
``origin``; there only ``base`` and ``python`` occur. Enum members compare equal
to their string values, so ``origin == "python"`` still works and JSON exports
contain the plain ``"base"``, ``"python"``, ``"management"``, and ``"unknown"``
strings shown above.

The ``status`` field is a :class:`~zett_agent.jit_harness.enums.ToolCallStatus`
describing how the attempt ended:

* ``completed`` — the handler returned. A later serialization or lifecycle-hook
  failure does not change the record, so ``completed`` means the tool itself ran.
* ``failed`` — the tool started and raised: argument validation, handler code, or
  an uncaught nested dependency error.
* ``cancelled`` — the awaiting caller was cancelled. A synchronous worker may
  still be running, because Python cannot forcibly stop it.
* ``rejected`` — the harness refused the call before any handler body ran: the
  tool is unloaded, missing its base binding, no longer registered in the current
  request, or the nested call exceeded ``max_call_depth``.

Recent-error records reuse the same enum but only ever hold ``failed`` or
``rejected``; constructing one with a success or cancellation status is rejected.

Use these measurement rules when interpreting results:

* Input length is compact Unicode JSON, including property names and punctuation.
  Text output uses plain string length; structured output uses compact JSON.
  Multimodal output counts text blocks only, excluding image bytes and URLs.
  These are characters, not bytes, tokens, or exact provider payload sizes.
* A missing measurement is ``None``, not zero. Character averages use measured
  samples only. Error output counts the error-message length; cancellation has
  no output measurement.
* Measurements stop at 8,000 characters, the same budget as a model-facing
  response, so a reported 8,000 means "at least this many". Beyond that point,
  measuring exactly would cost dispatch time without changing a decision.
* Duration is elapsed dispatch time, including validation, synchronous worker
  dispatch, and nested calls. Surrounding middleware, lifecycle hooks, user
  approval waits, and input/output measurement are outside the boundary.
  A parent includes its children, so summing both double-counts nested work.
  Cancelled synchronous dispatch stops timing when its await is cancelled;
  Python cannot forcibly stop the worker, which may finish later.
* A middleware short-circuit is not counted, because the tool never executed.
  Unknown names, stale request references, and rejected module-loading calls
  also create no record.

The profile retains a bounded window of finished attempts in memory, not their
input/output contents or exception messages: the newest ``call_capacity``
records (2,000 by default) survive, oldest dispatch first, while lifetime
aggregates keep counting every attempt. Pass another ``call_capacity`` to
``JITHarness`` or ``JITHarnessExtension`` when a long session needs a smaller
window. This does not change normal conversation history or add nested
execution-trace events. Separate session profiles remain isolated. They survive
later requests in the same in-memory harness but are not included in executable
snapshots: activating a snapshot starts fresh measurements. In-flight calls are
not published; their record appears when the attempt ends.

The model's ``inspect_tool_calls`` returns ``calls``, ``total_calls``, and
``next_offset``. Its default is five records, with the tool name, null fields,
and request metadata omitted. Caller relationships and error types appear only
when applicable. It does not return source, schemas, or aggregate statistics.

The application's ``harness.profile`` uses ``offset``, ``limit`` (default 100),
and ``next_offset``; call records are paginated here rather than through
``inspect_tool``. Aggregates always cover the full observed lifetime, not only
the current page. Export after a request if you need stable pagination while
reading all records:

.. code-block:: python

   json_page = report.model_dump_json()
   # Store pages using your application's storage.
   from zett_agent.jit_harness import JITHarnessProfile

   restored_page = JITHarnessProfile.model_validate_json(json_page)

For long sessions, record retention consumes memory in proportion to the number
of calls. Export what you need and release idle harnesses with ``forget_session``.
Profile export does not activate code or seed a new harness's measurements.

Diagnose recent errors
----------------------------------------

``inspect_tool_errors`` gives the model recent error messages to decide what
to do next: retry with corrected arguments, load a dependency, repair dynamic
Python source, or explain an application failure. It is a read-only tool;
querying errors does not retry a call, update source, or save a snapshot.

Each session has a separate LRU cache of distinct errors. The defaults retain
100 entries and at most 4,000 characters of each message:

.. code-block:: python

   jit = JITHarnessExtension(error_capacity=100, error_message_chars=4000)
   # After a request:
   report = jit.get_harness("builder").tool_errors("twice", limit=20)
   for error in report.errors:
       print(error.error_type, error.message, error.occurrences)
       print(error.last_call_id, error.caller_tool_name)

Both configuration values must be positive integers. Capacity is global to
one harness, not a separate allowance for every tool. Errors deduplicate by
tool name, exception type, and complete error message. A repeated error
increments ``occurrences`` and moves that entry to the most recent position;
when the cache is full, the least recently occurring entry is evicted.
Reads do not refresh recency, and successful calls do not clear old evidence.

Records include the error type, bounded message, last occurrence time, count,
and the latest invocation's caller and call ID. ``message_truncated`` indicates
whether only a prefix was retained. Different messages with the same truncated
prefix remain distinct. If an exception cannot be formatted, its type and
context are retained with ``message=None``.

``occurrences`` covers the entry's current cache residence; an evicted error
starts at one if it reappears. Lifetime failed/rejected counts remain available
through profiling and never reset on eviction. An empty error report is not
proof that a tool never failed. Cancellations are recorded in profiling rather
than the error cache. Failed management calls, including source registration
or update errors, are retained too.
The cache follows the dispatch profiling boundary: surrounding middleware or
lifecycle-hook failures are not captured, and stale request references cannot
add diagnostics to a later request.

The model receives three entries with message previews of at most 512 characters
by default. Raise ``max_chars`` only when the preview is insufficient, or lower
``limit`` for longer messages. ``message_truncated`` remains true when the cache
or the response preview clipped a message. The response carries ``errors``,
``total_matches``, and ``next_offset``; each error carries its tool, exception
type, message preview, occurrence count, and latest call ID. Cache configuration
and duplicated request metadata are omitted.

The application's ``harness.tool_errors`` returns the complete retained messages,
newest occurrence first. Follow ``next_offset`` to retrieve more retained entries;
``limit`` is 1–1000, defaulting to 5. Its full report includes
capacity, global ``retained_errors``, and filtered ``total_matches``. Error
details survive later requests, updates, and unload in this in-memory session,
but are not part of executable snapshots. They can be separately exported
with ``report.model_dump_json()`` and read with
``ToolErrorReport.model_validate_json(...)``.

.. warning::

   Exception messages may include sensitive application data. Arguments,
   tracebacks, and exception objects are not stored, but the error text itself
   is retained. Use applications and tools that redact secrets in their error
   messages. Diagnostics can also contain untrusted text; model guidance treats
   them as data, not instructions to execute.

Observe management and dynamic tool calls
-----------------------------------------

``JITHarnessExtension`` publishes two CUSTOM events so a UI or log can follow the
harness as it works, including calls that never surface as ``TOOL_STARTED``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Event name
     - Emitted when
   * - ``jit_tool_started``
     - A management or dynamic tool call is about to run.
   * - ``jit_tool_completed``
     - That call returned or raised, adding ``success`` and ``error_type``.

Both events cover all management tools, dynamic Python tools, and text workflow
readers the model created. They fire for calls the model requested and for calls one dynamic
tool composes through ``harness.call_tool``, so a composed dependency is visible
even though its result never reaches the conversation. Application base tools are
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
``origin``; the completed event adds ``success`` and ``error_type``. ``origin``
uses the same values as profiling: ``management`` identifies control calls,
``python`` executable tools, and ``text`` workflow reads.

The event stays small on purpose: arguments, results, and error messages are not
copied into it. ``tool_call_id`` is the join key, matching both the conversation's
``ToolCall`` and the ``call_id`` of the record in
:meth:`~zett_agent.jit_harness.runtime.JITHarness.profile`. Use that to read
duration, input/output sizes, and the composing caller.

Events pair up for calls that reach dispatch. A call the harness rejects before
running any handler — an unloaded dependency, a missing base binding, or a
depth-limit breach — emits no pair of its own; read ``profile()`` for its
``rejected`` record and ``tool_errors()`` for the message. Names this harness has
never seen are ignored rather than reported as JIT calls, and a lifecycle hook in
another extension that raises before the handler can leave a started event
without its completion.

Save and use in a new session
----------------------------------------

Saving is application-controlled and never automatic:

.. code-block:: python

   from zett_agent.jit_harness import JITHarnessSnapshot

   jit.get_harness("builder").save("harness.json")
   snapshot = JITHarnessSnapshot.load("harness.json")
   reused = JITHarnessExtension(snapshot)
   other = await create_agent(
       model,
       config=AgentRunConfig(session_id="next-session"),
       tools=[add],
       extensions=[InMemoryMessageAccumulator(), reused],
   )
   reply = await other.run("Use the saved toolbox.")

Each session receives an independent copy of the template. Later requests in
the same session retain its evolution; modifying one session does not change
another or overwrite the file. The save destination's parent directory must
exist. Replacement is atomic, so an interrupted write does not expose a
partially written JSON file.

Snapshots contain format-versioned Python source, entry points, tool metadata,
Markdown workflows, base-tool names, and unload state. They do not contain conversation history,
callables, credentials, connections, or mutable Python globals. Reloading
recreates module globals from source. Python dependencies must be installed
in the destination application's environment.
Python records remain in ``tools``; Markdown records use ``text_tools``.
Older Python-only format-1 snapshots load with an empty ``text_tools`` collection.

``JITHarnessSnapshot.load`` and ``from_json`` only validate data; they do not
execute source. ``JITHarness(snapshot, tools=[...])`` activates code directly,
or the extension activates it on the new session's first request. Unloaded
tools' modules are activated too; unload controls invocation, not loading
side effects. Missing base tools do not prevent activation, but calling one
fails explicitly. Bind base tools through ``tools=[...]`` or a tool-providing
extension. Request-scoped bindings are released after each request, so calls
outside that request cannot reuse closed application resources.

``JITHarnessExtension`` drives that binding through the harness's own lifecycle
API: ``attach(context, management_tools)`` binds one request and mounts its
tools, ``detach(context)`` releases it, and the ``active`` property reports
whether a request is currently bound. The extension calls them in matching
pairs; an application normally reads ``active`` at most. Attaching twice raises,
while detaching an older request is ignored so late cleanup cannot dismantle the
request that is running now.

Use ``snapshot.to_json()`` and ``JITHarnessSnapshot.from_json(...)`` for your
own storage. For long-lived servers, ``jit.forget_session(session_id)`` releases
an idle in-memory toolbox without deleting snapshots or conversation history.

Try the complete offline example
--------------------------------

The scripted model creates ``twice``, invokes it, and reuses its saved source
in an independent session. The program also checks a synchronous → asynchronous
→ base-tool pipeline, empty inputs, failed updates, dependency unload, and
snapshot isolation. It uses a temporary directory and makes no network requests.
It also saves, unloads, restores, and reads a reusable text workflow.
The scripted model also inspects profiling after a tool call, and the program
checks aggregate and nested-call measurements:

.. literalinclude:: ../_examples/jit_harness.py
   :language: python
   :pyobject: main

.. code-block:: console

   uv run python docs/_examples/jit_harness.py

Expected output:

.. code-block:: text

   JIT tool created and reused: 42
   JIT invariants checked: composition, failed update, unload, snapshot isolation
   JIT profiling checked: per-tool totals and nested call records
   JIT diagnostics checked: recent dependency error retained

Download :download:`the complete program <../_examples/jit_harness.py>`.
For exact contracts, see :class:`~zett_agent.jit_harness.runtime.JITHarness`,
:class:`~zett_agent.jit_harness.snapshot.JITHarnessSnapshot`, and
:class:`~zett_agent.jit_harness.extension.JITHarnessExtension`.
