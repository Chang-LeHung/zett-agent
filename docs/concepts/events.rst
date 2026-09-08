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
     - Every extension's on_event
     - Awaited sequential broadcast
   * - ExternalEvent
     - Application / UI callback
     - Extension.accept(config, event)
     - Synchronous routed broadcast

UI output
-------------

A CUSTOM AgentEvent has a non-empty ``name`` and an application-defined ``payload``.
It retains the session ID. The dispatcher calls ``on_custom_event``; the custom
name is data, not a Python method to dynamically invoke.

Internal notifications
--------------------------

``MessageAppendedEvent`` contains original output plus timing/usage;
``PhaseTransitionEvent`` contains the previous/next phases and timestamps;
``RunCancelledEvent`` tells subscribers to release pending work. Publishing an
``InternalMessageEvent`` queues an AgentMessage for another model iteration.

An internal event is not retained or forwarded to the UI automatically. Subscriber
exceptions stop the broadcast; previously processed subscribers are not rolled back.
Avoid recursive publishing of the same event from its own handler.

External replies
--------------------

The application supplies routing in AgentConfig and business data in the event
payload. ``emit_external_event`` returns the names of extensions that accepted it.
Acceptance means delivery was claimed, not that the pending tool succeeded.
Duplicate or stale approval replies are rejected by ExternalEventExtension.

Implement all three channels in :doc:`../extending/events`, or run the
:doc:`approval round trip <../examples/approval>`.
