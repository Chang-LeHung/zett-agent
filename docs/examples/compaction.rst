Compact without losing history
==================================

Compress an old turn while preserving the original Raw Log. The tiny word-based token estimate is for this demonstration only.

Run it
----------

From the repository root:

.. code-block:: console

   uv run --directory backend/zett-agent python docs/_examples/compaction.py

The output includes:

.. code-block:: text

   Raw messages: 4; checkpoint boundary: 2; raw tail: 2

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download compaction.py <../_examples/compaction.py>`.

.. literalinclude:: ../_examples/compaction.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
