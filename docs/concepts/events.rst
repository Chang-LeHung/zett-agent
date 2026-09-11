Three event channels
========================

The framework separates UI output, internal notifications, and external input.
All three are called events, but their destinations and delivery rules differ.

.. list-table:: Choose the right channel
   :header-rows: 1
   :widths: 22 28 25 25

   * - Type
     - Producer
     - Consumer
     - Delivery
   * - AgentEvent
     - Runtime or event-producing hook
     - UI / AgentEventDispatcher
     - Async stream, in order
   * - ExtensionEvent
     - context.publish(event)
     - Every extension's on_event by default, or one named extension
     - Awaited sequential broadcast or targeted delivery
   * - ExternalEvent
     - Application / UI callback
     - Extension.accept(config, event)
     - Synchronous routed broadcast

UI output
-------------

A CUSTOM AgentEvent has a non-empty ``name`` and an application-defined ``payload``.
It retains the session ID. The dispatcher calls ``on_custom_event``; the custom
name is data, not a Python method to dynamically invoke.

Output segment boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~

Primary model calls expose ``REASONING_STARTED``, ``REASONING_COMPLETED``,
``CONTENT_STARTED``, and ``CONTENT_COMPLETED`` in both asynchronous and
synchronous streams. A typical sequence is::

    MODEL_STARTED
        REASONING_STARTED
        REASONING_DELTA ...
        REASONING_COMPLETED
        CONTENT_STARTED
        TEXT_DELTA ...
        CONTENT_COMPLETED
    MODEL_COMPLETED

The corresponding dispatcher methods are ``on_reasoning_started_event``,
``on_reasoning_completed_event``, ``on_content_started_event``, and
``on_content_completed_event``. Boundaries retain the current session ID and
the GENERATING request phase. Extension subscribers still receive the typed
output lifecycle notifications containing UTC and monotonic timestamps.

An empty delta does not open a segment. A response without streamed content
does not invent content boundaries. Tool arguments close any active reasoning
segment; reasoning cannot start again after content or tool arguments. Each
completion requires an open matching segment. Events after a final response
and output outside GENERATING raise AgentProtocolError.

Failure or cancellation leaves an interrupted segment without a successful
completion event. Consumers should handle the stream exception and close their
loading indicator in a finally block. Compaction retains its separate events.

Internal notifications
--------------------------

``MessageAppendedEvent`` contains original output plus timing/usage;
``PhaseTransitionEvent`` contains the previous/next phases and timestamps;
``RunCancelledEvent`` tells subscribers to release pending work. Publishing an
``InternalMessageEvent`` queues an AgentMessage for another model iteration.

An internal event is not retained or forwarded to the UI automatically. Pass a
unique extension name as ``target`` for one recipient; omit it to broadcast.
Subscriber exceptions stop delivery; previously processed subscribers are not
rolled back. Avoid recursive publishing of the same event from its own handler.

External replies
--------------------

The application supplies routing in AgentConfig and business data in the event
payload. ``emit_external_event`` returns the names of extensions that accepted it.
Acceptance means delivery was claimed, not that the pending tool succeeded.
Duplicate or stale approval replies are rejected by ExternalEventExtension.

Implement all three channels in :doc:`../extending/events`, or run the
:doc:`approval round trip <../examples/approval>`.
