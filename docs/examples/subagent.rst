Run an isolated child agent
===============================

Run a child with read-only tools and explicitly shared temporary storage. Tool restrictions are not an operating-system sandbox.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/subagent.py

The output includes:

.. code-block:: text

   Two isolated sessions; child links to parent

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download subagent.py <../_examples/subagent.py>`.

.. literalinclude:: ../_examples/subagent.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
