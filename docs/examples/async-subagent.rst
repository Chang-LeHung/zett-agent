Run subagents in the background
===================================

Start a child agent without waiting for it, keep the parent working, and read
the report when it is ready. Tool restrictions on a child are not an
operating-system sandbox.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/async_subagent.py

The output includes:

.. code-block:: text

   Background task finished; child session is separate

This program uses deterministic offline models. It requires no API key and is
executed by the documentation test suite.

How the model manages a task
------------------------------------------

:class:`~zett_agent.extensions.async_subagent.AsyncSubAgentExtension` registers
five tools:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Tool
     - Purpose
   * - ``task_start``
     - Start a task from a configured profile; returns its ``task_id`` at once.
   * - ``task_list``
     - Show every task of the session with its status.
   * - ``task_status``
     - Show one task without its report; never waits.
   * - ``task_output``
     - Return the report, waiting up to ``wait_seconds`` when asked.
   * - ``task_cancel``
     - Stop a running task.

A finished task is announced before the parent's next model call, or when the
session's next request starts, as an ordinary agent message. A task whose
result the model already read is not announced again.

.. literalinclude:: ../_examples/async_subagent.py
   :language: python
   :pyobject: main

Customize it by subclassing
------------------------------------------

Override only what you need instead of copying the extension:

.. code-block:: python

   class TerseNotices(AsyncSubAgentExtension):
       def build_tools(self, context, subagents):
           # Drop cancellation for this deployment.
           return tuple(t for t in super().build_tools(context, subagents) if t.name != "task_cancel")

       def completion_notice(self, task):
           return f"{task.task_id} is {task.status.value}."

       async def task_finished(self, task):
           metrics.record(task.subagent_type, task.elapsed)

``task_info`` shapes what each tool returns, ``task_started`` and
``task_finished`` observe the lifecycle (their errors are logged, never raised
into the task), and ``_create_child`` changes how each child agent is built.
Tasks live in memory for the lifetime of the extension object and are visible
only to the session that started them. Call ``await extension.close()`` when the
application shuts down.

Complete source
-------------------

:download:`Download async_subagent.py <../_examples/async_subagent.py>`.

.. literalinclude:: ../_examples/async_subagent.py
   :language: python
   :linenos:

Next: :doc:`subagent` for the blocking variant, or
:doc:`../extending/built-ins` to compare it with the other capabilities.
