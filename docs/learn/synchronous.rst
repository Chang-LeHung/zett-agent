Use it from synchronous code
============================

Not every program is async. ``create_agent_sync`` runs the same runtime on one
background event loop, so scripts, CLIs, and notebooks can call it from ordinary
``def`` code. Model behavior, tools, extensions, session history, retries, and
event types are identical to the async API.

Run and stream
--------------

.. code-block:: python

   from zett_agent.sync import create_agent_sync

   with create_agent_sync(model) as agent:
       reply = agent.run("Hello")
       print(reply.content)

       with agent.stream("Continue") as events:
           for event in events:
               print(event.type, event.delta)

Use the stream as a context manager if you might stop early; closing it cancels
the unfinished request and waits for cleanup. Later requests on the same session
run normally.

Blocking views of async objects
-------------------------------

Agents, clients, providers, tools, storage, and dispatchers expose ``.sync()``,
which mirrors their async methods with blocking calls:

.. code-block:: python

   with existing_agent.sync() as agent:
       agent.initialize(config=config)
       reply = agent.run("Hello")

   with read_file.sync() as read:
       result = read({"path": "README.md"})

   with storage.sync() as db:
       session = db.load("session-42")

For anything else — free functions, third-party clients — use
:class:`~zett_agent.sync_runtime.SyncRuntime` directly:

.. code-block:: python

   with SyncRuntime() as runtime:
       agent = runtime.call(Agent.create, model, config=config)
       result = runtime.call(agent.run, "Hello")
       with runtime.stream(agent.stream, "Continue") as events:
           for event in events:
               print(event)

Share one runtime for resources that belong together, and close what you own
before leaving it. A runtime you pass in explicitly stays open when a view exits.

Callbacks and custom models
---------------------------

Extension hooks and dispatcher callbacks stay ``async def``; they run on the
background loop with the real request context. GUI applications should schedule
visual updates through their own UI thread rather than touching widgets from a
callback.

If your model is a blocking ``def stream(request)`` generator, wrap it with
:class:`~zett_agent.sync.SyncModelAdapter`. Existing ``@tool`` functions already
support both styles.

Sending input into a running request
------------------------------------

While a synchronous stream is paused at an :doc:`Ask User <../extending/events>`
or Plan Mode event, another thread can deliver the reply with
``agent.emit_external_event(event, config=config)``. Different sessions can run
concurrently; overlapping requests for one session are rejected.

Complete offline example
------------------------

.. literalinclude:: ../_examples/synchronous.py
   :language: python

:download:`Download the example <../_examples/synchronous.py>`.

.. seealso::
   :doc:`streaming` for event handling, :doc:`sessions` for persistence, and
   :doc:`../extending/hooks` for the full hook list.
