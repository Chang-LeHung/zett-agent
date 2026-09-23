Cancel and reuse an agent
=============================

Close an interrupted stream, clean up extension state, and complete a new request.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/cancellation.py

The output includes:

.. code-block:: text

   Cancelled request; extension state cleared

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download cancellation.py <../_examples/cancellation.py>`.

.. literalinclude:: ../_examples/cancellation.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
