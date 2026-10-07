Troubleshooting
===============

Start with the exception and the last event you received. A tool error can be
recoverable; a provider or lifecycle error ends the request. Use
:doc:`application` for failure boundaries and :doc:`testing` to reproduce the
behavior without a live model. Never include credentials or full private
prompts in a public error report.

A decorated tool fails at import time
------------------------------------------------------------

``A tool needs at least one non-empty guideline`` means the function has a
description but no usage guidance. Add ``@tool(guidelines="When to use it.")``
or a nonempty ``Guidelines:`` docstring section. Every argument also needs a
type annotation. ``Args`` entries must match real parameter names; unknown
entries and unnamed variadic parameters are rejected. See :doc:`tools`.

The endpoint rejects reasoning or tool search
------------------------------------------------------------

The adapter does not negotiate model capabilities. Verify the endpoint supports
the selected protocol and reasoning level. On Responses, the current adapter
maps the highest levels to ``high``; on Chat Completions an unsupported named
level may be rejected. ``OFF`` can omit the setting rather than disabling
provider-default reasoning.

``ToolSearchExtension`` needs a compatible Responses tool-search endpoint,
not just ordinary tool calling. Use ``response=True`` and a model supporting
that protocol, or omit tool search and expose tools normally. Skill loading
does not require Responses. See :doc:`providers` and :doc:`on-demand`.

The second turn forgot the first
--------------------------------

History comes from an extension. Passing ``extensions=[...]`` replaces the
defaults, so add :class:`~zett_agent.extensions.memory.InMemoryMessageAccumulator`
or a persistence extension, and keep using the same ``session_id``. Two clients
with different accumulator instances do not share memory just because the IDs
match.

The model never sees a message you added
----------------------------------------

Add context in a lifecycle hook, not by editing shared state:

* ``on_state`` — restore history and inject system instructions.
* ``on_message`` — replace the pending user input before it is sent.
* ``before_model`` — adjust context before each model step.

Use the ``context`` passed to the hook. See
:doc:`../concepts/lifecycle` for exactly when each hook runs.

Your custom event never reaches the UI
--------------------------------------

Internal notifications and UI events are different channels. ``context.publish``
delivers an :class:`~zett_agent.extensions.events.ExtensionEvent` to extensions;
``await context.emit(AgentEvent(...))`` queues something for ``Agent.stream``.
Use the second one for anything the user should see. See
:doc:`../extending/events`.

An approval stays pending
-------------------------

The response must match the request's session ID, external event name, and
payload shape. If no registered extension accepts it, the request keeps waiting.
Do not block the loop with ``input()`` — use an async UI or
``asyncio.to_thread`` for terminal input. A cancelled request invalidates its
pending routes, so a late reply is ignored.

Reasoning or cache numbers are missing
--------------------------------------

Reasoning text appears only when the model returns it; the effort level is a
request preference, not a promise. Usage is reported on ``MODEL_COMPLETED``, not
on every text fragment. ``cache_hit_rate`` is ``None`` both when the provider
reported nothing and when there were no input tokens;
``ModelUsage.cache_reported`` tells the two apart. Gateways that translate
protocols often drop the cache breakdown, so prefer a Responses adapter when
cache visibility matters.

Retries duplicated text
-----------------------

Built-in retries stop after the first streamed event, so a retry should never
replay visible output. Check application-level retries and make sure you are not
appending both the final message text and every ``TEXT_DELTA`` fragment to the
same buffer.

Project instructions or a skill did not appear
------------------------------------------------------------

``AgentsMdExtension`` loads the chosen directory and its ancestors, not every
nested file in the workspace. Check ``sources()`` and ``instructions()``, and
verify the extension is in your explicit list. Empty or unreadable files are
skipped. ``directory`` does not change tool paths. See :doc:`project-instructions`.

For skills, check ``extension.skills``. Discovery occurs at construction time;
new files or catalog metadata need a new extension. Existing skill bodies are
read on demand, but editing a file does not replace instructions already in
history. Pass explicit project roots rather than assuming defaults include
the working directory. See :doc:`on-demand`.

An MCP server has no tools or fails before generation
------------------------------------------------------------

Check ``extension.servers`` and ``extension.config_path``. A missing config
loads no servers, whereas a bad config raises. Discovery connects on every
request; a server that cannot be reached fails setup rather than silently
disappearing. By default a remote ``search`` tool on ``catalog`` is named
``catalog__search``. See :doc:`mcp` for transports and namespaces.

The runtime rejects a second request
------------------------------------------------------------

Two requests for the same session cannot overlap. Queue them or finish/close
the first stream, including cleanup. Breaking out of a bare async iterator
without ``aclosing`` can leave unfinished work. Different session IDs may run
concurrently, but separate runtimes and workers do not share an admission lock.
See :doc:`application`.

Compaction did not run
----------------------

It needs a crossed size threshold *and* an older completed turn it can replace.
The current turn is kept whole, so a single oversized turn is left alone. When
the generated summary would not be smaller than the messages it replaces,
``COMPACTION_COMPLETED`` reports ``applied=False``. See
:doc:`../examples/compaction` for a reproducible test.

The page has no navigation
--------------------------

If you are reading this locally, rebuild with ``make docs-serve`` and open the
printed URL. The left sidebar is global; on a narrow screen, open it with the
navigation toggle.
