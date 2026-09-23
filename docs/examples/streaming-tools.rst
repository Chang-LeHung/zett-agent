Stream a tool round trip
============================

Observe reasoning, execute add, and return its result to the next model step.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/streaming_tools.py

The output includes:

.. code-block:: text

   Answer: 42

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download streaming_tools.py <../_examples/streaming_tools.py>`.

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
