Stream to a terminal or UI
==============================

Use callbacks when you do not need to control iteration
-----------------------------------------------------------

Subclass :class:`~zett_agent.AgentEventDispatcher` and override only the events
you want to display. Bind it through ``create_agent(..., event_dispatcher=...)``;
``client.run`` then dispatches events and still returns the final answer.

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: ConsoleEvents

``delta`` is incremental, while ``message`` on a completion event is complete.
Appending both to the same UI text buffer would duplicate the answer. A tool's
arguments may arrive as incomplete fragments; only complete ToolCall objects
are suitable for execution. The framework executes tools, not the UI.

Consume the stream directly when you need cancellation or forwarding
------------------------------------------------------------------------

.. code-block:: python

   from contextlib import aclosing

   async with aclosing(client.stream("Inspect the project")) as events:
       async for event in events:
           await send_to_ui(event)

Do not dispatch manually if a dispatcher is already bound. Callbacks are awaited
in order, so slow callbacks apply backpressure. Use a bounded application queue
if UI delivery must be decoupled; do not silently spawn unlimited tasks.

Event channels are different
--------------------------------

* **AgentEvent:** the stream consumed by a UI; includes CUSTOM events.
* **ExtensionEvent:** internal notifications delivered through ``context.publish``.
* **ExternalEvent:** application input routed through ``Agent.emit_external_event``.

An internal notification is not automatically a UI event. See
:doc:`../extending/events` for exact routing and examples.

Failures are not terminal success events
--------------------------------------------

``RUN_COMPLETED`` means the request completed successfully. Provider failures,
hook exceptions, and task cancellation propagate as exceptions; they are not
converted to synthetic RUN_COMPLETED events. Closing early releases the active
request and pending waits. Use ``aclosing`` even when breaking out of iteration.

Run :doc:`../examples/cancellation` to verify cleanup and reuse. A cancelled
conversation's already-appended history is not automatically rolled back.
