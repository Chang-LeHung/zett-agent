Keep and restore conversations
==================================

Session identity is not history storage
-------------------------------------------

``AgentConfig.session_id`` identifies a conversation. ``request_id`` is an
optional correlation ID for one invocation, not an idempotency key. Reusing a
request ID does not deduplicate messages. The caller must define any retry/edit
policy that requires replacing an earlier user submission.

Each request creates fresh AgentState. The default in-memory extension restores
messages from earlier requests in the same process. To survive restarts, configure
SQLite persistence explicitly.

Restore using a new client and connection
---------------------------------------------

.. literalinclude:: ../_examples/sessions.py
   :language: python
   :pyobject: main

This program writes to a temporary database, closes the first connection, creates
a new client, restores the first turn, queries history, renames the session, and
deletes it. The assertions check the stored roles and restored view.

Run and download it from :doc:`../examples/sessions`. For a real application,
replace the temporary path with an application-owned database path and omit the
demo's deletion. Never run destructive cleanup on the default user database.

Raw Log versus model context
--------------------------------

* Use ``list_raw_messages`` for the conversation UI. It returns original messages
  in sequence order, including tools and internal agent input.
* Use ``load`` / SessionView for model context: latest checkpoint plus its raw tail.
* Compaction creates a new snapshot; it does not rewrite Raw Log history.

``after_sequence`` is exclusive; ``through_sequence`` is inclusive. ``offset``
and ``limit`` apply after filtering. Session listings are ordered by latest
activity, not message sequence.

.. warning::

   Explicit extensions replace optional defaults. Use one history-restoring
   extension per agent; do not combine SQLite and in-memory restore casually.
   Add ToolGuidelinesExtension separately if tool prompt guidance is needed.

Next: :doc:`../concepts/context`, :doc:`../examples/compaction`,
:doc:`../extending/storage-adapter`.
