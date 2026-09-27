Emit events and accept user input
=====================================

Send progress to a UI
-------------------------

Use ``context.emit`` from a lifecycle hook when work needs visible progress. Here the payload
contains a model-step counter from the first-extension tutorial:

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: NoteExtension.before_model

The application receives the exact event through ``on_custom_event``. Choose a
stable namespaced name and document payload keys as an application protocol.
Use JSON-compatible values if the UI forwards them over SSE or WebSocket.

Observe internal state without displaying it
------------------------------------------------

.. literalinclude:: ../_examples/observability.py
   :language: python
   :pyobject: UsageObserver

MessageAppendedEvent records message timing and model usage. The subscriber
does not need to infer timing by watching text arrive in the UI. Do not double
count cache reads as input tokens or reasoning as additional output tokens.
Run :doc:`../examples/observability` to inspect the complete example.

To publish a custom internal notification, define a small ExtensionEvent subclass
and ``await context.publish(event)``. Every extension's on_event receives it in
priority order. Use a different event type when reacting, rather than recursively
publishing the same event again. Internal events are not streamed automatically.

Pass ``target="ExtensionName"`` to deliver only to the extension with that
registered name. A missing or ambiguous target raises ``ValueError``; omitting
``target`` preserves ordered broadcast behavior.

Pause a tool for external approval
--------------------------------------

Subclass ExternalEventExtension for this behavior. It owns the lock, pending
Future, routing, duplicate rejection, loop-safe wake-up, and cleanup. Your subclass
owns the tool schema, outbound event, and validation of the returned answer.

.. literalinclude:: ../_examples/approval.py
   :language: python
   :pyobject: ApprovalExtension

The wait is registered **before** the event is emitted, so an immediate UI reply
is safe. The async-with exit waits for the response; it does not block the event
loop. Only after the reply is staged does the actual tool execute and consume it.

Return the answer through the Agent
---------------------------------------

The application routes the reply using config, separately from business payload:

.. literalinclude:: ../_examples/approval.py
   :language: python
   :pyobject: main

In this offline demonstration the answer is accepted automatically. In a real
application, render the question and wait for an actual user choice before emitting
the response. The tool does not execute the proposed action itself.

Acceptance and validation
-----------------------------

* The return value of emit_external_event is a list of accepting extension names.
* A duplicate, mismatched session, mismatched supplied request ID, stale route,
  or wrong response name is rejected by the base.
* Acceptance claims delivery; the tool still validates ``approved`` as a bool.
* RunCancelledEvent cancels pending waits; on_error wakes them with the error.
* Calling an accepted reply twice does not run the tool twice.

Use :doc:`../examples/approval` for run instructions and expected output.
Protected subclass helper contracts are included on the
:class:`~zett_agent.extensions.external.ExternalEventExtension` API page because extension authors
need them, despite their underscore names.

Internal continuation versus user steering
----------------------------------------------

Publish ``InternalMessageEvent(AgentMessage(content=...))`` when an extension
needs a bounded follow-on model step. GoalExtension uses this to request more work
after review. Do not mutate the loop queue directly. Urgent user steering is a
separate built-in channel and has priority over internal input. Neither mechanism
interrupts an HTTP request in the middle of receiving a model response.
