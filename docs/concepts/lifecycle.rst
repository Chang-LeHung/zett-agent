The request lifecycle
=========================

Setup is a series of barriers
---------------------------------

For each hook stage, **all** registered extensions run in ascending numeric
priority before the next stage starts. Priority 10 runs before 100; equal values
retain registration order. Extensions must have unique names, including built-ins.

.. code-block:: text

   fresh AgentState + AgentRunContext
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
        turn 1..n    before_turn() -> model step -> tools -> after_turn()
             |
        after_run() -> on_success() -> COMPLETED -> RUN_COMPLETED

Changing priority does not move ``on_message`` ahead of ``on_state``. It only
changes ordering **within** a stage. This is why a tool-guidance extension can
inspect tools registered by any extension, regardless of their registration order.

One model/tool cycle
------------------------

.. code-block:: text

   before_turn()
          |
   assemble initial ModelRequest
          |
   before_model(context, request)
          |
   refresh ModelRequest -> MODEL_STARTED -> deltas -> final ModelResponse
          |
   append AssistantMessage -> after_model() -> MODEL_COMPLETED
          |
          +-- no tool calls --> after_turn() --> check steering/internal inputs --> final answer
          |
          +-- tool calls --> before_tool()
                                      |
                                TOOL_STARTED
                                      |
                                execute tool
                                      |
                         create ToolMessage -> after_tool()
                                      |
                            append ToolMessage
                                      |
                          TOOL_COMPLETED / TOOL_FAILED
                                      |
                          steering check / next tool / model
                                      |
                                 after_turn()

Hooks and their exact purpose are catalogued in :doc:`../extending/hooks`.
The :class:`~zett_agent.extensions.base.AgentExtension` reference also includes the complete
state diagram with internal notifications and exceptional paths.

Lifecycle hooks emit visible progress with ``await context.emit(event)``. All
internal work writes to the request's :class:`~zett_agent.event_queue.AgentEventQueue`; the
outer :meth:`~zett_agent.agent.Agent.stream` is the only AgentEvent async generator.
It drains events in order until RUN_COMPLETED and cancels the producer if the
caller closes the stream early.

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
