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

Choose the child's capabilities explicitly
------------------------------------------

Give a child profile its own instructions, model, and tools. The parent can
delegate work to the profile without giving it every capability the parent
has:

.. literalinclude:: ../_examples/subagent.py
   :language: python
   :pyobject: main

The example stores the parent and child as separate sessions in the same
temporary database. A parent-session link lets your application show how
they relate. Read-only filesystem tools restrict the exposed operations;
they do not isolate the process or prevent arbitrary code from writing.

Complete source
-------------------

:download:`Download subagent.py <../_examples/subagent.py>`.

.. literalinclude:: ../_examples/subagent.py
   :language: python
   :linenos:

Next: :doc:`../extending/built-ins` for the subagent profile options, or
:doc:`goal` for evaluation-driven continuation.
