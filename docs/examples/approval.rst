Pause for external approval
===============================

Correlate an external reply with a waiting tool. The simulated UI answers immediately; duplicate replies are rejected. No export is actually performed.

Run it
----------

From the repository root:

.. code-block:: console

   uv run --directory backend/zett-agent python docs/_examples/approval.py

The output includes:

.. code-block:: text

   Response accepted; duplicate rejected

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download approval.py <../_examples/approval.py>`.

.. literalinclude:: ../_examples/approval.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
