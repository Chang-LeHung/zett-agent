State, concurrency, and cleanup
===================================

State has an owner and a lifetime
-------------------------------------

.. list-table:: Choose storage by lifetime
   :header-rows: 1
   :widths: 24 36 40

   * - Lifetime
     - Suitable location
     - Example
   * - One callback
     - Local variable
     - A computed payload
   * - One request
     - Dictionary keyed by AgentRunContext
     - Tool approval or model-step counter
   * - One conversation
     - Map keyed by session ID or persistent storage
     - Accumulated message history
   * - Application process
     - Immutable shared configuration or bounded sink
     - Model client pool or metrics exporter

One Agent can run different sessions concurrently. Same-session overlap is
rejected. The ``agent.state`` convenience attribute points to the most recently
prepared state, so it is not a safe way for a shared extension to select its
request. Always use ``context.state`` passed to that hook.

AgentRunContext uses identity semantics, making it suitable as a request-local map
key. A normal dictionary retains it strongly; remove entries on every terminal
path. Do not rely on garbage collection as your cleanup protocol.

Complete cleanup pattern
----------------------------

.. literalinclude:: ../_examples/cancellation.py
   :language: python
   :pyobject: WorkTracker

``discard``/``pop(..., None)`` make cleanup tolerant of partially initialized or
already-released state. Cancellation is not an Exception subclass handled by
on_error; it is delivered internally as RunCancelledEvent. Subclasses of the
external-response base must preserve its cleanup by calling super().

Check early stream closure
------------------------------

.. literalinclude:: ../_examples/cancellation.py
   :language: python
   :pyobject: main

After the context manager closes the stream, the phase is CANCELLED and the
extension's active set is empty. A later request succeeds on the same Agent with
a fresh state. Persisted prior messages are not automatically deleted or edited.

Testing an extension before shipping
----------------------------------------

At minimum, exercise:

1. Success and normal tool error.
2. Failure before your state is fully initialized.
3. Cancellation during model streaming and while waiting for external input.
4. Two sessions using the same extension instance.
5. Duplicate and late external replies.
6. A second request after success, failure, and cancellation.
7. Registration name/tool-name collisions and equal-priority ordering.

The documentation tests run these tutorial classes, not a separate copy. The
core runtime suite additionally exercises state-machine and external-event races.

Avoid blocking async hooks
------------------------------

Do not call synchronous terminal input or slow filesystem/network operations on
the event-loop thread. Use an async library or asyncio.to_thread where appropriate.
Do not add your own thread lock around an await. ExternalEventExtension already
handles cross-thread response delivery safely.
