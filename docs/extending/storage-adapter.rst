Implement a storage adapter
===============================

Keep lifecycle integration in the base extension
----------------------------------------------------

Do not duplicate context restoration, Raw Log events, or compaction bookkeeping
inside a new database extension. Implement the SessionStorage contract and pass
it to BaseSessionPersistenceExtension. The base already knows when to load, append,
snapshot, and clean up request bookkeeping.

The three runtime storage operations
----------------------------------------

.. list-table:: SessionStorage boundary
   :header-rows: 1
   :widths: 18 42 40

   * - Method
     - Returns
     - Invariants
   * - load
     - SessionView: optional checkpoint and ordered raw tail
     - A consistent view; unknown sessions are empty.
   * - append
     - Assigned per-session sequence number
     - Original messages are immutable; appends advance sequence atomically.
   * - snapshot
     - ContextSnapshot with a new version
     - Check expected_version and advancing Raw Log boundary; never rewrite raw messages.

CRUD for session titles and paginated UI history is provided by concrete storage
implementations such as SQLiteSessionStorage. It is separate from the three
operations required by the runtime persistence extension.

A complete adapter example
------------------------------

This adapter wraps SQLite with an audit trail. It intentionally delegates actual
transactions to the tested SQLite implementation; it is not a claim to implement
a different database. A PostgreSQL or remote adapter would replace these bodies
while preserving the same ordering/version guarantees.

.. literalinclude:: ../_examples/storage_adapter.py
   :language: python
   :pyobject: AuditedStorage

The extension itself stays small:

.. literalinclude:: ../_examples/storage_adapter.py
   :language: python
   :pyobject: AuditedPersistence

Run the :doc:`complete program <../examples/storage-adapter>` to verify that one
request causes ``load, append:system, append:user, append:assistant`` and no
unnecessary snapshot.

Snapshot consistency checklist
----------------------------------

* Read the checkpoint and its tail consistently; do not combine different database versions.
* Treat compacted_through_sequence as an inclusive represented boundary.
* Start the replay tail strictly after that boundary.
* Reject stale expected_version rather than silently overwriting a checkpoint.
* Do not store a checkpoint after every successful response.
* Preserve message roles, attributes, replay blocks, timing, metadata, tags, and usage.
* Do not treat request_id as an automatic idempotency key.

Use independent tests for empty sessions, two-message turns, tool round trips,
compaction plus a later tail, conflicts, unknown sessions, and database reopening.
Use temporary databases and release all connection pools.

Supporting checkpoint type
------------------------------

The snapshot contract uses this class from its defining module:

.. code-block:: python

   from zett_agent.extensions.compaction import CompactedMessage

   checkpoint = CompactedMessage(content="The user prefers Python.")

CompactedMessage is a specialized system message marking a context checkpoint.
Its ``content`` contains the summary; inherited ``attributes`` holds optional
message annotations. It is not exported from the package root. Store this
checkpoint separately from the original Raw Log, together with the inclusive
sequence boundary described above.

Related API: :class:`~zett_agent.extensions.persistence.SessionStorage`,
:class:`~zett_agent.extensions.persistence.BaseSessionPersistenceExtension`,
:class:`~zett_agent.storage.SQLiteSessionStorage`, :class:`~zett_agent.extensions.persistence.SessionView`.
