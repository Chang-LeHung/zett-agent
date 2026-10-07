Observe usage and timing
============================

Record token usage, prompt-cache hits, and reasoning/content timing without
changing how your application runs an agent. The model's counts are fixtures;
the example does not measure provider prices.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/observability.py

The output includes:

.. code-block:: text

   Reasoning and content boundaries recorded

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Record the completed message
----------------------------

Use an extension when you want the finalized message, usage, and timing
together, rather than each display delta:

.. literalinclude:: ../_examples/observability.py
   :language: python
   :pyobject: UsageObserver

The example retains records in a list and checks a 60% cache-hit rate. For a
long-running service, send records to a bounded queue or telemetry sink
instead of keeping an unbounded list in memory. Reported cache usage depends
on the provider; missing usage is not a measured zero.

Complete source
-------------------

:download:`Download observability.py <../_examples/observability.py>`.

.. literalinclude:: ../_examples/observability.py
   :language: python
   :linenos:

Next: :doc:`../learn/providers` for prompt caching, or
:doc:`../concepts/events` for choosing between display and lifecycle events.
