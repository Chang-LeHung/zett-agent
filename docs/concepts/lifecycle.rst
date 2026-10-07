The request lifecycle
=====================

Knowing the order of hooks is how you pick the right one. Each stage finishes for
every extension before the next stage starts, and extensions run in ascending
``priority`` (lower numbers first).

.. code-block:: text

   new request
        │
   on_tool()       register the tools this request can call
        │
   on_state()      restore history and inject system instructions
        │
   on_message()    transform the pending user input
        │
   append input
        │
   before_run()
        │
   turn 1..n       before_turn() → model step → tools → after_turn()
        │
   after_run() → on_success() → RUN_COMPLETED

Because ``on_tool`` always completes before ``on_state``, a guidance extension
can describe every tool no matter which extension registered it. Priority only
changes the order *within* a stage.

One turn
--------

.. code-block:: text

   before_turn()
        │
   before_model(context, request)      adjust context before this model call
        │
   model streams deltas → final response
        │
   after_model()
        │
        ├─ no tool calls ─▶ after_turn() ─▶ final answer
        │
        └─ tool calls ────▶ before_tool() ─▶ run tool ─▶ after_tool()
                                                               │
                                           after_turn() ◀───────┘
                                                │
                                           next turn

Emit UI progress from any hook with ``await context.emit(AgentEvent(...))``. The
event becomes visible to whoever is consuming ``Agent.stream()``.

Which hook should I use?
------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Hook
     - Use it for
   * - ``on_tool``
     - Registering request-scoped tools.
   * - ``on_state``
     - Restoring history and adding system instructions.
   * - ``on_message``
     - Rewriting the user's input before it is sent.
   * - ``before_model``
     - Anything that must change immediately before a model call, including compaction.
   * - ``after_model``
     - Reading the response, usage, or recording a checkpoint.
   * - ``before_tool`` / ``after_tool``
     - Approving, rewriting, or observing one tool call.
   * - ``after_run`` / ``on_success``
     - Persisting results and releasing per-request state.
   * - ``on_error`` / ``RunCancelledEvent``
     - Cleaning up after a failure or cancellation.

Success, failure, cancellation
------------------------------

Cleanup needs all three paths:

* A model or hook error calls ``on_error`` and propagates the original exception.
* Cancelling the request publishes ``RunCancelledEvent``; ``on_error`` and
  ``on_success`` do not run.
* A tool failure is different: it becomes an unsuccessful tool result so the
  model can react, rather than failing the whole request.

See :doc:`../extending/hooks` for every hook and
:doc:`../extending/state` for a cleanup pattern that handles all three.
