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

Close the stream when you leave it
------------------------------------------

Use ``aclosing`` around a stream you might stop early. The example breaks on
the first text delta, exits the context manager, and then reuses the client
for a complete request:

.. literalinclude:: ../_examples/cancellation.py
   :language: python
   :pyobject: main

If an extension retains request-local state, clean it on cancellation as well
as success and error. ``WorkTracker`` demonstrates all three paths. Closing a
stream does not roll back a tool action that already completed; design
side-effecting tools with that distinction in mind.

Complete source
-------------------

:download:`Download cancellation.py <../_examples/cancellation.py>`.

.. literalinclude:: ../_examples/cancellation.py
   :language: python
   :linenos:

Next: :doc:`../learn/streaming` for consumer patterns, or
:doc:`../extending/state` for extension cleanup.
