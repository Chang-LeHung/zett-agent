Lifecycle hook reference
============================

All hooks receive :class:`~zett_agent.AgentContext` unless noted. An ordinary
async hook returns None. The four ``*_events`` hooks are async generators yielding
AgentEvent. ``accept`` is synchronous and returns bool. Default implementations
are no-ops; override only what your extension needs.

Setup hooks
---------------

.. list-table:: Runs once per request, not once per Agent instance
   :header-rows: 1
   :widths: 25 35 40

   * - Hook
     - State at entry
     - Intended operation
   * - :meth:`~zett_agent.AgentExtension.on_tool`
     - Fresh registry already contains constructor tools.
     - Register dynamic tools through context.register_tool.
   * - :meth:`~zett_agent.AgentExtension.on_state`
     - All dynamic tools registered; new input not appended.
     - Restore history and insert system instructions.
   * - :meth:`~zett_agent.AgentExtension.on_message`
     - History ready; input in context.input_message.
     - Replace or validate current input; do not append it yourself.

Request hooks
-----------------

.. list-table:: Success and failure are separate paths
   :header-rows: 1

   * - Hook
     - Boundary and constraints
   * - :meth:`~zett_agent.AgentExtension.before_run`
     - Transformed input has been appended and published; initialize per-request work.
   * - :meth:`~zett_agent.AgentExtension.after_run`
     - A final answer is ready with no remaining queued input.
   * - :meth:`~zett_agent.AgentExtension.on_success`
     - All after_run hooks succeeded; precedes RUN_COMPLETED. Release success state.
   * - :meth:`~zett_agent.AgentExtension.on_error`
     - A non-cancellation request error is being propagated. Release failure state.

Model and tool hooks
------------------------

.. list-table:: May execute multiple times within one request
   :header-rows: 1

   * - Hook
     - Boundary and constraints
   * - :meth:`~zett_agent.AgentExtension.before_model`
     - Before prompt assembly; inspect or change context.state.messages and context.tools for this step.
   * - :meth:`~zett_agent.AgentExtension.after_model`
     - Complete AssistantMessage already appended; receives ModelResponse including usage.
   * - :meth:`~zett_agent.AgentExtension.before_tool`
     - Before one tool call. A raised exception fails the request before execution.
   * - :meth:`~zett_agent.AgentExtension.after_tool`
     - ToolMessage already appended, including execution failure results; inspect result.success.

Event-producing hooks
-------------------------

.. list-table:: Yield, do not return, AgentEvent objects
   :header-rows: 1

   * - Hook
     - Ordering
   * - :meth:`~zett_agent.AgentExtension.before_model_events`
     - After before_model, before ModelRequest assembly. Supports CUSTOM and the compaction lifecycle.
   * - :meth:`~zett_agent.AgentExtension.after_model_events`
     - After after_model and MODEL_COMPLETED, before tool dispatch or final-answer handling.
   * - :meth:`~zett_agent.AgentExtension.before_tool_events`
     - After before_tool, before TOOL_STARTED. Use for UI approval or preparation progress.
   * - :meth:`~zett_agent.AgentExtension.after_tool_events`
     - After after_tool and TOOL_COMPLETED/TOOL_FAILED, before steering selection or another tool.

Skipped or cancelled tools do not trigger after_tool_events. Custom output is
produced while the request is READY; it does not become a phase transition simply
because its payload says "working". Do not synthesize MODEL_STARTED or completion
events that are owned by the core loop.

Notification and input hooks
--------------------------------

.. list-table:: These hooks are not UI stream callbacks
   :header-rows: 1

   * - Hook
     - Contract
   * - :meth:`~zett_agent.AgentExtension.on_event`
     - Awaited for ExtensionEvent published inside the request; inspect concrete event types.
   * - :meth:`~zett_agent.AgentExtension.accept`
     - Receives config and ExternalEvent from external callers, potentially on another thread. Return whether delivery was accepted.

Use ExternalEventExtension for pending responses instead of writing your own
cross-thread Future handling. If overriding its on_event or on_error methods,
call super() so the base can wake and release pending waits.

Priority, naming, and exceptions
------------------------------------

``priority = 100`` is the default. Lower values run earlier within each hook
stage. Two instances with the same name cannot be registered. Give independently
configured instances explicit names through ``extension.name`` when needed.

If a hook fails, previously completed hooks are not rolled back. Cleanup must be
idempotent and tolerate partial initialization. Cancellation has its own internal
RunCancelledEvent, not an on_cancel method. See :doc:`state`.
