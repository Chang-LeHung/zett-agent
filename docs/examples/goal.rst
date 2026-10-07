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

Separate the worker from the reviewer
------------------------------------------

The application selects goal mode for a specific upcoming request. A reviewer
checks the worker's result and can request another attempt:

.. literalinclude:: ../_examples/goal.py
   :language: python
   :pyobject: main

Here the reviewer deliberately reports missing tests once, then accepts the
second answer. In a real workflow, have it inspect concrete evidence rather
than accepting the worker's claims. Bound continuation with an iteration
limit and give the user a way to cancel; an evaluator is not a guarantee of
correctness.

Complete source
-------------------

:download:`Download goal.py <../_examples/goal.py>`.

.. literalinclude:: ../_examples/goal.py
   :language: python
   :linenos:

Next: :doc:`subagent` for ordinary delegation, or
:class:`~zett_agent.extensions.goal.GoalExtension` for selection and limits.
