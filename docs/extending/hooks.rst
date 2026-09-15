Lifecycle hook reference
============================

All hooks receive :class:`~zett_agent.AgentRunContext` unless noted. Lifecycle
hooks are ordinary async functions returning None. ``accept`` is synchronous and
returns bool. Lifecycle defaults are no-ops; middleware defaults transparently
call the next layer. Override only what your extension needs.

Setup hooks
---------------

.. list-table:: Runs once per request, not once per Agent instance
   :header-rows: 1
   :widths: 25 35 40

   * - Hook
     - State at entry
     - Intended operation
   * - :meth:`~zett_agent.AgentExtension.on_tool`
     - Fresh local registry already contains constructor tools; server registry is empty.
     - Register dynamic tools through context.register_tool or context.register_server_tool.
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
     - Receives context and ModelRequest; inspect it or change context.state.messages, context.tools, and context.server_tools.
   * - :meth:`~zett_agent.AgentExtension.after_model`
     - Complete AssistantMessage already appended; receives ModelResponse including usage.
   * - :meth:`~zett_agent.AgentExtension.before_tool`
     - Before one tool call. A raised exception fails the request before execution.
   * - :meth:`~zett_agent.AgentExtension.after_tool`
     - Tool execution has finished but ToolMessage is not appended yet. Inspect the separate error argument and modify the result before context and persistence receive it.

Middleware hooks
----------------

.. list-table:: Wrap the operation instead of observing one side of it
   :header-rows: 1

   * - Hook
     - Boundary and constraints
   * - :meth:`~zett_agent.AgentExtension.on_model_request`
     - Wraps the provider stream. Pass a ModelRequest to ``call_next`` and close the returned iterator when leaving early.
   * - :meth:`~zett_agent.AgentExtension.on_tool_call`
     - Wraps one registered local tool handler. It may adjust arguments, transform or short-circuit the result, or raise a normal tool failure.

These hooks use the same extension priority ordering as lifecycle hooks, with
lower priorities forming the outer middleware layers. Provider-hosted tools run
inside ``on_model_request`` and do not use the local ``on_tool_call`` boundary.

Emitting outward events
--------------------------

Any lifecycle hook may publish visible progress with
``await context.emit(AgentEvent(...))``. The event enters the request-owned queue
at that exact point in hook priority order. Internal runtime methods and
extensions never expose nested AgentEvent async generators; only
:meth:`~zett_agent.Agent.stream` drains the queue and yields to callers.

Custom output does not become a phase transition simply because its payload says
"working". Do not synthesize MODEL_STARTED, TOOL_STARTED, or terminal events that
are owned by the core loop. Use ``context.publish`` instead for internal
extension-to-extension notifications that should not reach the UI.

``before_model`` receives ``(context, request)``. The runtime refreshes request
messages and both tool registries before each extension and again before
the provider call. Request fields are frozen; edit ``context.state.messages``,
``context.tools``, or ``context.server_tools`` to affect subsequent hooks and the
provider. The supplied request is a shallow view at entry, not a live view of
later registry changes.

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
