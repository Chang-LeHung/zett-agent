Compact without losing history
==================================

Compress an old turn while preserving the original Raw Log. The tiny word-based token estimate is for this demonstration only.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/compaction.py

The output includes:

.. code-block:: text

   Raw messages: 4; checkpoint boundary: 2; raw tail: 2

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Ask for compaction
--------------------

``CompactionExtension`` also runs on demand: ``AgentClient.compact`` restores the
session, asks every extension to compact through ``on_compact``, and returns the
checkpoint it stored. No input message is appended and no primary model call is
made, so a UI can summarize a conversation the moment a reader asks for it::

   stored = await client.compact()
   print(stored.summary if stored is not None else "nothing to compact")

The events of the pass reach the client's ``event_dispatcher`` exactly as a
streamed run does, which is how the ``Compaction started`` lines above are
produced.

Complete source
-------------------

:download:`Download compaction.py <../_examples/compaction.py>`.

.. literalinclude:: ../_examples/compaction.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
