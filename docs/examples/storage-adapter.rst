Connect a storage adapter
=============================

Wrap an existing storage backend to record persistence operations, while
letting the persistence extension manage conversation history.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/storage_adapter.py

The output includes:

.. code-block:: text

   load, append:system, append:user, append:assistant

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Adapt the store, not the agent loop
------------------------------------------

``AuditedStorage`` delegates reads and writes to SQLite while recording their
order. The extension only needs to construct that adapter:

.. literalinclude:: ../_examples/storage_adapter.py
   :language: python
   :pyobject: AuditedPersistence

The complete source implements the storage methods for loading history,
appending messages, and saving a compaction checkpoint. If you replace SQLite
with another database, preserve sequence ordering, transactions, and version
checks; those guarantees matter when restoring a conversation safely.

Complete source
-------------------

:download:`Download storage_adapter.py <../_examples/storage_adapter.py>`.

.. literalinclude:: ../_examples/storage_adapter.py
   :language: python
   :linenos:

Next: :doc:`../extending/storage-adapter` for the contract, or
:doc:`../learn/sessions` if SQLite already meets your needs.
