Middleware
==========

Every :class:`~zett_agent.AgentExtension` inherits
:class:`~zett_agent.MiddlewareHook`. Override these methods when the extension
must wrap the operation that crosses into a provider or local tool handler.
There is no separate middleware registry or constructor argument.

Two boundaries
--------------

``on_model_request(context, request, call_next)``
   Wraps every primary provider invocation. It is an async generator because
   model output is streamed. Pass the original request or a dataclass replacement
   into ``call_next(request)`` and close the returned stream when leaving early.

``on_tool_call(context, call, call_next)``
   Wraps the actual handler for one local tool invocation. It may edit
   ``call.arguments`` before awaiting ``call_next()``, transform the raw
   :data:`~zett_agent.ToolResult`, or return directly to short-circuit the
   handler. An exception follows ordinary tool error handling and is recorded
   as a failed ``ToolMessage``.

Provider-hosted server tools do not pass through ``on_tool_call`` because the
provider executes them inside ``on_model_request``.

Ordering
--------

Middleware uses the extension's existing priority order as an onion. Given
``extensions=[outer, inner]``, the order is ``outer before -> inner before ->
operation -> inner after -> outer after``. Lower priorities are outer layers;
equal priorities retain registration order.

.. code-block:: python

   from contextlib import aclosing
   from dataclasses import replace
   from time import monotonic

   from zett_agent import AgentExtension, SystemMessage

   class TraceExtension(AgentExtension):
       async def on_model_request(self, context, request, call_next):
           request = replace(
               request,
               messages=(*request.messages, SystemMessage(content="Trace this request.")),
           )
           async with aclosing(call_next(request)) as events:
               async for event in events:
                   yield event

       async def on_tool_call(self, context, call, call_next):
           started = monotonic()
           try:
               return await call_next()
           finally:
               record_duration(call.id, monotonic() - started)

   agent = await Agent.create(
       model,
       config=config,
       extensions=[TraceExtension()],
   )

Concurrency and cancellation
----------------------------

One extension instance can be entered simultaneously by different sessions.
Parallel local tools also execute a separate complete middleware chain in each
task. Keep invocation state in method-local variables or use synchronization for
shared mutable state. ``current_tool_call_id()`` remains task-local throughout
tool middleware and the handler.

Use ``try/finally`` around ``call_next`` for cleanup. Model middleware should
consume the inner stream under ``contextlib.aclosing`` so client cancellation or
an early consumer exit reaches the provider and all inner middleware promptly.
