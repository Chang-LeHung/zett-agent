Stream to a terminal or UI
==========================

You can watch a request happen in two ways: register a dispatcher for the events
you care about, or consume the stream yourself when you need to cancel or
forward it.

Use callbacks for display
-------------------------

Subclass :class:`~zett_agent.dispatcher.AgentEventDispatcher` and override only
the events you want to render. Bind it through
``create_agent(..., event_dispatcher=...)``; ``client.run()`` still returns the
final answer.

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: ConsoleEvents

Two rules keep the display correct:

* ``delta`` is incremental, while ``message`` on a completion event is complete.
  Appending both to the same buffer duplicates the answer.
* Tool arguments can arrive as incomplete fragments. Only complete tool calls
  are executed, and the runtime does that — your UI never runs them.

``TOOL_STARTED`` can describe a batch: iterate ``event.tool_calls`` instead of
assuming there is one call. Completion and failure events identify the finished
call; correlate them by ID, since parallel calls can finish out of order.

Consume the stream yourself for control
---------------------------------------

.. code-block:: python

   from contextlib import aclosing

   async with aclosing(client.stream("Inspect the project")) as events:
       async for event in events:
           await send_to_ui(event)

For a simple terminal, match only the fields associated with each event:

.. code-block:: python

   from zett_agent.events import AgentEventType

   async with aclosing(client.stream("What is 20 plus 22?")) as events:
       async for event in events:
           if event.type == AgentEventType.TEXT_DELTA:
               print(event.delta, end="", flush=True)
           elif event.type == AgentEventType.TOOL_STARTED:
               for call in event.tool_calls:
                   print(f"\nCalling {call.name} ({call.id})")
           elif event.type == AgentEventType.RUN_COMPLETED:
               final_message = event.message

``client`` must already have a model and any tools it needs. The final message
is the authoritative answer; deltas are for presentation. A provider may emit
an answer only in its final response, so a UI should reconcile the final content
instead of assuming text deltas always exist. If a dispatcher is bound, it has
already handled each event before you receive it here.

Use ``aclosing`` even when you break out of the loop early: it stops the request
and waits for cleanup. Callbacks are awaited in order, so slow callbacks apply
backpressure; hand work to a bounded application queue instead of spawning
unlimited tasks.

What you will see
-----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Event
     - Meaning
   * - ``TEXT_DELTA``
     - A fragment of the answer text.
   * - ``REASONING_DELTA``
     - A fragment of the model's reasoning, when the model exposes it.
   * - ``TOOL_STARTED`` / ``TOOL_COMPLETED`` / ``TOOL_FAILED``
     - A requested tool call and how it finished.
   * - ``TOOL_CALL_DELTA``
     - Partial tool arguments, for progress display only.
   * - ``COMPACTION_*``
     - Older context is being summarized before the next model step.
   * - ``MODEL_COMPLETED``
     - One complete model response, including usage.
   * - ``RUN_COMPLETED``
     - The request finished successfully.
   * - ``CUSTOM``
     - An event your own extension emitted.

Provider-hosted tools add ``SERVER_TOOL_*`` events; steering and internal
continuations add their own pairs. The full list is on
:class:`~zett_agent.events.AgentEventType`.

Track model usage
------------------

Read usage from ``event.response.usage`` on ``MODEL_COMPLETED``. It describes
one model step, not the whole user request. A tool round trip often produces
two or more such events, so sum their counters to measure the primary loop:

.. code-block:: python

   tokens = 0
   async with aclosing(client.stream("Explain the result.")) as events:
       async for event in events:
           if event.type == AgentEventType.MODEL_COMPLETED and event.response is not None:
               tokens += event.response.usage.total_tokens

``total_tokens`` is input plus output; reasoning and cache counters are already
included and must not be added again. Missing usage is not proof of zero cost.
Separate compaction or delegated models can incur additional usage outside
these primary model events. See :doc:`../examples/observability` for recorded
timings and :doc:`providers` for cache reporting.

Failures are exceptions
-----------------------

``RUN_COMPLETED`` means success. Provider errors, extension errors, and
cancellation propagate as exceptions — they are never converted into a synthetic
completion event. Wrap the stream when the UI needs to show a failure, and see
:doc:`../examples/cancellation` for cleanup and reuse.

Next: keep the conversation in :doc:`sessions`, or read
:doc:`../concepts/events` for the difference between UI events, internal
notifications, and external input.
