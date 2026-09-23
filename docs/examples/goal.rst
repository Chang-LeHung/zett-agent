Iterate toward a goal
=========================

A ``goal_mode`` external event selects one future request. A private evaluator
then reports incomplete and finally achieved. This demonstrates control flow,
not real-model evaluation quality.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/goal.py

The output includes:

.. code-block:: text

   Implementation and tests are complete.

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download goal.py <../_examples/goal.py>`.

.. literalinclude:: ../_examples/goal.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
