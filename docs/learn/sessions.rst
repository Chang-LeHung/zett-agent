Keep and restore conversations
==============================

Each request starts with fresh context. A history extension restores previous
dialogue, so you decide how long a conversation lives and where it is stored.
Default clients remember turns in memory; they do not persist them to disk.

Identity is not storage
-----------------------

:class:`~zett_agent.agent.AgentRunConfig` names the conversation:

.. code-block:: python

   from zett_agent.agent import AgentRunConfig

   config = AgentRunConfig(session_id="project-notes")

The same ``session_id`` across requests is what lets a history extension find the
right messages. ``request_id`` is only a correlation label for one request; it is
not an idempotency key, and reusing it does not deduplicate anything.

Pick a history extension
------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Extension
     - Lifetime
   * - :class:`~zett_agent.extensions.memory.InMemoryMessageAccumulator`
     - One process; the default when you do not pass ``extensions``.
   * - :class:`~zett_agent.extensions.sqlite.SQLiteSessionExtension`
     - Survives restarts; stores raw history and compaction checkpoints.
   * - Your own implementation
     - See :doc:`../extending/storage-adapter`.

Keep history in a script
------------------------

.. code-block:: python

   from zett_agent.client import create_agent

   client = await create_agent(model)
   await client.run("My preferred language is Python.")
   reply = await client.run("What language did I mention?")

With a real model, the answer wording varies. The important contract is that
the second request receives the earlier user and assistant messages. Reuse
the client or the same accumulator instance to retain that memory. Creating
a new client with a new default accumulator starts fresh, even if you reuse
the session ID. A different ID selects an independent history.

Configure a durable conversation
--------------------------------

.. code-block:: python

   from zett_agent.agent import AgentRunConfig
   from zett_agent.extensions.sqlite import SQLiteSessionExtension
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   storage = SQLiteSessionExtension("sessions.sqlite")
   config = AgentRunConfig(session_id="project-notes")
   try:
       client = await create_agent(
           model,
           config=config,
           extensions=[storage, ToolGuidelinesExtension()],
       )
       await client.run("Remember my preference for Python.")
   finally:
       await storage.close()

This creates or updates a real file relative to the process working directory.
Choose an application-owned path and keep it stable across restarts. The
extension also accepts ``":memory:"`` for an ephemeral SQLite database.
Omitting the path uses its user-data default; use explicit paths in tests and
multi-user applications.

After reopening the same database, create a client with the same ``config``
and send the next user message. Current system instructions and tools come
from the new client; persisted system prompts do not accumulate in model
context. Close the provider separately when it is no longer needed.

Restore across processes
------------------------

.. literalinclude:: ../_examples/sessions.py
   :language: python
   :pyobject: main

This program writes to a temporary database, closes it, opens a new client, and
restores the first turn — the pattern to copy for a real application. Close the
extension when finished with ``await storage.close()``.

.. warning::

   Use one history-restoring extension per agent. Combining SQLite and in-memory
   restore makes it unclear which messages the model sees.

Two views of the same conversation
----------------------------------

You will usually want history for a UI and history for the model, and they are
not identical once compaction runs.

* ``await storage.list_raw_messages(session_id)`` returns the original records in
  order — what you show in a transcript.
* ``await storage.storage.load(session_id)`` returns the model context: the latest
  checkpoint plus the raw messages after it.

Compaction adds a checkpoint; it never rewrites the raw log. Pagination and
listing helpers on the storage extension cover the rest (``after_sequence`` is
exclusive, ``through_sequence`` is inclusive).

The raw log includes system messages and tool-only assistant messages, not
just visible chat bubbles. A UI should select the roles and fields it wants
to render instead of displaying every record as user-facing text.

.. code-block:: python

   records = await storage.list_raw_messages(
       "project-notes", after_sequence=0, limit=50,
   )
   for record in records:
       print(record.sequence, record.message.role, record.message)

``storage.storage.load`` returns a ``SessionView`` with ``snapshot``,
``raw_tail``, and a ``messages`` projection. During restore, the persistence
extension additionally omits incomplete assistant/tool batches left by an
interrupted process, so an endpoint never receives an orphan tool result.
The original records stay available in the raw log for diagnosis.

List, label, and forget conversations
-------------------------------------

.. code-block:: python

   sessions = await storage.list_sessions(limit=20, offset=0)
   await storage.update_session("project-notes", title="Project notes")
   summary = await storage.get_session("project-notes")

``create_session`` can create an empty session before the first run.
``parent_session_id`` links a delegated conversation to its parent;
``session_type`` is an application-defined integer classification, not a
permission or runtime mode.

To intentionally delete the stored session and its log/checkpoints, call
``await storage.delete_session(session_id)``. This is deletion, not archival;
stop active requests first and apply your retention/backup policy. In-memory
history has ``accumulator.clear(session_id)`` instead. Session IDs do not
control access — authenticate the owner before any history operation.

Interrupted runs are not transactions
-------------------------------------

Persistence appends completed message records as the request proceeds. A
later error or cancellation does not undo a user message, finished tool result,
or external side effect. Partial text deltas are not final assistant records.
After a restart, let the restore projection select valid context rather than
reconstructing it from UI fragments. For retry and idempotency decisions,
see :doc:`application`.

Running out of context
----------------------

Add :class:`~zett_agent.extensions.compaction.CompactionExtension` to summarize
older turns when the conversation approaches the model's limit. Recent turns are
kept whole, including their tool calls and results, and a summary that would not
be smaller is discarded. See :doc:`../examples/compaction`.

Next: understand :doc:`../concepts/context`, or move to
:doc:`../extending/index` to build your own behavior.
