Implement session storage from scratch
=======================================

Keep every session in one JSON file and let the persistence extension own the
lifecycle: restore context, append the immutable Raw Log, and save a compaction
checkpoint.

Run it
------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/custom_storage.py

The output is:

.. code-block:: text

   Stored 3 messages; reopened 3 from a new instance

The program uses a deterministic offline model and a temporary directory. It
requires no API key and is executed by the documentation test suite.

What the program shows
----------------------

``JsonFileStorage`` implements three asynchronous operations — ``load``,
``append``, and ``snapshot`` — over a dictionary that is written to disk after
every change. ``JsonPersistence`` only passes it to the base extension, which
supplies everything else: when to restore, what to append, and when to compact.

The final assertion reopens the file with a fresh storage instance. That is the
part a real deployment cares about: the next process must see the same
conversation, in the same order, with the same compaction boundary.

Complete source
---------------

:download:`Download custom_storage.py <../_examples/custom_storage.py>`.

.. literalinclude:: ../_examples/custom_storage.py
   :language: python
   :linenos:

Next: :doc:`../extending/storage-adapter` explains each operation and the
invariants to keep, or :doc:`../learn/storage` covers the SQLite, MySQL, and
PostgreSQL storage that ships with the runtime.
