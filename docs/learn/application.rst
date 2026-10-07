Integrate an application
========================

Zett Agent supplies the model/tool loop. Your application supplies the UI,
authentication, session routing, resource lifetime, and policies. This guide
shows the boundaries to handle when moving from a script to a service.

Choose a calling style
----------------------

.. list-table:: Three ways to consume a request
   :header-rows: 1
   :widths: 29 71

   * - API
     - Use it when
   * - ``await client.run(...)``
     - You need the final answer. A bound dispatcher still receives progress.
   * - ``client.stream(...)``
     - You forward events, stop early, or implement interactive controls.
   * - ``create_agent_sync(...)``
     - Your host is blocking Python code. See :doc:`synchronous`.

For an HTTP stream or WebSocket, translate events into your own wire format.
``AgentEvent`` may carry Python messages, exceptions, or provider payloads; it
is not an application-level JSON protocol. Select the fields your UI needs
and avoid forwarding secrets or raw exception objects.

Set a request deadline
----------------------

Use ``asyncio.wait_for`` on Python 3.10+ to bound the entire awaited run:

.. code-block:: python

   import asyncio

   try:
       reply = await asyncio.wait_for(client.run("Review the change."), timeout=60)
   except asyncio.TimeoutError:
       print("The request timed out.")
   else:
       print(reply.content)

Timeout cancels unfinished work and waits for cancellation cleanup. Cleanup
can make total elapsed time longer than the deadline. Provider retries and
tool waits consume that same deadline; ``max_iterations`` is not a timeout.

Cancellation is not rollback. A tool may already have changed an external
system, and a synchronous function running in a worker thread cannot be
forcibly stopped by cancelling its awaiter. Give mutating tools their own
timeouts, transaction rules, and idempotency controls. Never automatically
replay a whole request just because the connection to the UI closed.

Close streams on disconnect
---------------------------

.. code-block:: python

   from contextlib import aclosing

   async with aclosing(client.stream("Inspect the project.")) as events:
       async for event in events:
           await send_to_ui(event)

``send_to_ui`` is your application transport. If it raises on disconnect,
the context manager closes the unfinished request. The same cleanup happens
when the loop exits early. In a web framework, propagate request cancellation
instead of leaving detached generation tasks running accidentally.

A dispatcher attached to this client is called **before** the event is yielded.
Do not dispatch the event again yourself. Callbacks are awaited in order and
can slow generation; use a bounded queue for slow sinks. For a GUI, schedule
widget updates on the GUI thread rather than touching them from async or
background-loop callbacks.

Distinguish failure boundaries
------------------------------------------------------------

.. list-table:: What ends the request?
   :header-rows: 1
   :widths: 31 69

   * - Situation
     - Behavior
   * - A tool handler raises an ordinary exception
     - Emits ``TOOL_FAILED`` and gives the model an unsuccessful tool result. The model can continue.
   * - Provider, protocol, lifecycle hook, or callback fails
     - Propagates the exception. Do not assume a ``RUN_COMPLETED`` event will arrive.
   * - Request is cancelled or stream closes early
     - Cancels unfinished work and releases the session. No successful final answer is synthesized.
   * - Model-call budget is exhausted
     - Raises ``AgentIterationLimitError``. Use a new user request only after deciding how to handle partial work.

Close loading indicators in ``finally``. Preserve cancellation when handling
errors: do not convert ``CancelledError`` into an ordinary successful reply.
For user-visible errors, show a safe message and keep detailed diagnostics in
an access-controlled sink. Exceptions from tool handlers are visible to the
model as text; sanitize them before they leave sensitive integrations.

Route concurrent conversations
------------------------------

Different sessions can run concurrently on the same runtime:

.. code-block:: python

   from zett_agent.agent import AgentRunConfig

   replies = await asyncio.gather(
       client.run("First conversation", config=AgentRunConfig(session_id="chat-a")),
       client.run("Second conversation", config=AgentRunConfig(session_id="chat-b")),
   )

Overlapping requests for **one session** are rejected. Queue them in your
application or send supported external input into the active request; do not
start a second model loop for that conversation. ``client.agent.is_conflict(id)``
is useful for display, but the runtime's admission check is authoritative.
Use ``client.agent.get_state(session_id)`` for inspection rather than the
convenience ``state`` view, which reflects the most recently started request.

This guard belongs to one runtime instance. It is not a distributed lock across
processes or separate clients. In a multi-worker service, route a session to
one active worker or enforce serialization in your application's shared
coordination layer. Session IDs are identifiers, not authorization: validate
the user's access before running, listing, or deleting a conversation.

If callbacks serve several sessions, route by ``event.session_id``. Keep
extension state request-scoped and make shared tools safe for concurrency;
see :doc:`../extending/state` and :doc:`tools`.

Own resources explicitly
------------------------

Create providers and durable stores during application startup, keep them open
while requests use them, and close them during shutdown:

.. code-block:: python

   from zett_agent.client import create_agent
   from zett_agent.extensions.sqlite import SQLiteSessionExtension
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   history = SQLiteSessionExtension("sessions.sqlite")
   try:
       client = await create_agent(
           model,
           extensions=[history, ToolGuidelinesExtension()],
       )
       await serve_application(client)
   finally:
       await history.close()
       await model.aclose()

``model`` is your configured provider; ``serve_application`` is your server's
request lifecycle. Shutdown should stop admission and settle active requests
before closing their resources. The runtime closes the streams it creates,
but it does not close a provider or storage object you supplied. MCP transports
are an exception in scope: :doc:`mcp` opens and closes them per request.

An offline integration example
------------------------------

.. literalinclude:: ../_examples/application.py
   :language: python
   :pyobject: main

Run ``uv run python docs/_examples/application.py`` from a checkout:

.. code-block:: text

   Separate sessions: A1, B1
   Timeout cleaned up; session reused.

The :download:`complete program <../_examples/application.py>` verifies
independent history and cancellation cleanup without a network request.

Before deployment
-----------------

* Pin and test the model/endpoint combination your application relies on.
* Enforce tool access in code, not only in a prompt; isolate shell and file access.
* Set request deadlines, provider retry policies, and per-tool limits.
* Serialize each conversation and choose one history-restoring extension.
* Decide which prompts, images, results, and traces may be stored or logged.
* Test success, tool failure, disconnect, cancellation, and restart restoration.

Next: :doc:`testing` for fast regression tests, or :doc:`sessions` for history
and retention.
