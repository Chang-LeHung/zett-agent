Observe usage and timing
============================

Receive usage and reasoning/content timing through internal message events. The token counts are fixtures, not monetary costs.

Run it
----------

From the repository root:

.. code-block:: console

   uv run --directory backend/zett-agent python docs/_examples/observability.py

The output includes:

.. code-block:: text

   Reasoning and content boundaries recorded

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download observability.py <../_examples/observability.py>`.

.. literalinclude:: ../_examples/observability.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
