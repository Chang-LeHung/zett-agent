The request lifecycle
=========================

Setup is a series of barriers
---------------------------------

For each hook stage, **all** registered extensions run in ascending numeric
priority before the next stage starts. Priority 10 runs before 100; equal values
retain registration order. Extensions must have unique names, including built-ins.

.. code-block:: text

   fresh AgentState + AgentContext
             |
        on_tool()        register request-local tools
             |
        on_state()       restore history and system instructions
             |
        on_message()     transform context.input_message
             |
        append input     publish MessageAppendedEvent exactly once
             |
        before_run()
             |
        model/tool loop
             |
        after_run() -> on_success() -> COMPLETED -> RUN_COMPLETED

Changing priority does not move ``on_message`` ahead of ``on_state``. It only
changes ordering **within** a stage. This is why a tool-guidance extension can
inspect tools registered by any extension, regardless of their registration order.

One model/tool cycle
------------------------

.. code-block:: text

   before_model()
          |
   before_model_events()     may yield CUSTOM / compaction events
          |
   assemble ModelRequest -> MODEL_STARTED -> deltas -> final ModelResponse
          |
   append AssistantMessage -> after_model() -> MODEL_COMPLETED
          |
   after_model_events()
          |
          +-- no tool calls --> check steering/internal inputs --> final answer
          |
          +-- tool calls --> before_tool() -> before_tool_events()
                                      |
                                TOOL_STARTED
                                      |
                                execute tool
                                      |
                         append ToolMessage -> after_tool()
                                      |
                          TOOL_COMPLETED / TOOL_FAILED
                                      |
                              after_tool_events()
                                      |
                          steering check / next tool / model

Hooks and their exact purpose are catalogued in :doc:`../extending/hooks`.
The :class:`~zett_agent.AgentExtension` reference also includes the complete
state diagram with internal notifications and exceptional paths.

Exceptional paths
---------------------

* A normal model or hook exception marks active work FAILED, calls ``on_error``,
  and propagates the original error.
* Task cancellation or generator closure marks active work CANCELLED and
  publishes ``RunCancelledEvent``. It does **not** call ``on_error`` or ``on_success``.
* Tool execution failures become unsuccessful ToolMessages so the model can react.
  Exceptions in preparation hooks are not tool results: they fail the request.
* Observers can fail after a terminal transition was committed. The original
  terminal state remains; a callback exception does not rewrite completed history.

Cleanup therefore needs success, failure, **and cancellation** paths. See
:doc:`../extending/state` for a complete pattern and tested example.
