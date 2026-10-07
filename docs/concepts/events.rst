Three kinds of events
=====================

Zett Agent uses the word "event" for three different things. Choosing the right
one is the difference between a UI that updates and one that silently does
nothing.

.. list-table::
   :header-rows: 1
   :widths: 20 27 23 30

   * - Kind
     - Sent with
     - Received by
     - Typical use
   * - :class:`~zett_agent.events.AgentEvent`
     - ``await context.emit(...)``
     - ``Agent.stream`` and your dispatcher
     - Anything the user should see
   * - :class:`~zett_agent.extensions.events.ExtensionEvent`
     - ``await context.publish(...)``
     - Other extensions' ``on_event``
     - Coordination between extensions
   * - :class:`~zett_agent.extensions.external.ExternalEvent`
     - ``agent.emit_external_event(...)``
     - An extension's ``accept``
     - A reply from your UI back into a waiting request

UI output
---------

Emit an ``AgentEvent`` for anything a person should see. A custom event carries a
``name`` and a ``payload`` you define, and the dispatcher calls
``on_custom_event``; the name is data, not a Python method.

Streamed output arrives in a predictable order, so a UI can open and close
loading states:

.. code-block:: text

   MODEL_STARTED
     REASONING_STARTED → REASONING_DELTA … → REASONING_COMPLETED
     CONTENT_STARTED   → TEXT_DELTA …      → CONTENT_COMPLETED
   MODEL_COMPLETED

An empty delta does not open a segment, and a failure or cancellation leaves the
open segment without a completion — close your indicator in a ``finally`` block.

Coordination between extensions
-------------------------------

``context.publish`` notifies extensions, not the UI. Use it for things like
"history was compacted" (:class:`~zett_agent.extensions.events.CompactionEvent`)
or "a message was appended". Pass ``target="Name"`` to reach one extension, or
omit it to broadcast. An internal notification is never shown to the user
automatically.

Input from your UI
------------------

When the model asks a question or requests approval, the request pauses and your
application replies with ``emit_external_event``. The reply must match the
request's session ID and the expected event name and payload. If no extension
accepts it, nothing happens and the request keeps waiting.

See :doc:`../extending/events` to implement all three, or run the
:doc:`approval round trip <../examples/approval>`.
