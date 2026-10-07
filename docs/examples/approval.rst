Pause for external approval
===============================

Correlate an external reply with a waiting tool. The simulated UI answers immediately; duplicate replies are rejected. No export is actually performed.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/approval.py

The output includes:

.. code-block:: text

   Response accepted; duplicate rejected

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Connect an approval UI
----------------------

The example publishes an approval request, pauses the tool, and sends a
correlated reply back through the client. In a real application, replace the
simulated reply with your UI's accept or reject action.

.. literalinclude:: ../_examples/approval.py
   :language: python
   :pyobject: main

Keep the session, request, and tool-call identifiers with the prompt you show
the user. A reply for another request must not authorize this one. The example
checks that a duplicate reply is rejected and does not perform an actual
export.

Complete source
-------------------

:download:`Download approval.py <../_examples/approval.py>`.

.. literalinclude:: ../_examples/approval.py
   :language: python
   :linenos:

Next: :doc:`../extending/events` for response routing, or
:doc:`../extending/built-ins` for ``AskUserExtension`` and ``PlanModeExtension``.
