Synchronous Python
==================

Use :func:`~zett_agent.create_agent_sync` or :class:`~zett_agent.SyncAgent` from
an ordinary ``def`` function. The synchronous API runs the existing Agent on one
persistent background event loop. Model responses, tools, extensions, session
history, retries, and event types have the same meaning as the asynchronous API.

Run and stream
--------------

.. code-block:: python

   from zett_agent import create_agent_sync

   with create_agent_sync(model) as agent:
       reply = agent.run("Hello")
       print(reply.content)

       with agent.stream("Continue") as events:
           for event in events:
               print(event.type, event.delta)

``SyncAgent`` is initialized during construction; its optional ``config`` selects
a session, or a new UUIDv7 session is generated. Use the stream as a context
manager if consumption might stop early. Each ``next()`` pulls one event. Closing
the iterator cancels unfinished work and waits for cleanup; later requests in
that session can run normally. The iterator retains one asynchronous owner task
throughout, including MCP resource acquisition and release.

All other asynchronous APIs
---------------------------

Agent, AgentClient, providers, tools, storage, and dispatchers offer ``.sync()``.
Its context-managed view exposes their original
method names with blocking behavior:

.. code-block:: python

   with existing_agent.sync() as agent:
       agent.initialize(config=config)
       reply = agent.run("Hello")

   with read_file.sync() as read:
       result = read({"path": "README.md"})

   with storage.sync() as db:
       context = db.load("session-42")

For free functions, third-party clients, or class factories, use
:class:`~zett_agent.SyncRuntime` directly. ``call`` preserves parameter types
and the return value; ``stream`` returns a closeable synchronous iterator;
``context`` adapts an asynchronous context manager with its exception-suppression
behavior and task ownership intact.

.. code-block:: python

   with SyncRuntime() as runtime:
       agent = runtime.call(Agent.create, model, config=config)
       result = runtime.call(agent.run, "Hello")
       with runtime.stream(agent.stream, "Continue") as events:
           for event in events:
               print(event)

Share one runtime for related resources
---------------------------------------

SDK transports stay attached to the loop on which they were used. Reuse a runtime
for subsequent calls, and close caller-owned resources before leaving it. The
Agent does not assume ownership of a supplied model or database.

.. code-block:: python

   with SyncRuntime() as runtime:
       provider = runtime.call(DeepSeekProvider, model="deepseek-chat")
       try:
           with SyncAgent(provider, runtime=runtime) as agent:
               print(agent.run("Hello").content)
           with provider.sync(runtime=runtime) as blocking_provider:
               with blocking_provider.stream(request) as events:
                   for event in events:
                       print(event)
       finally:
           runtime.call(provider.aclose)

An explicitly supplied runtime stays open when a view exits. ``SyncRuntime``
closes its streams, cancels unfinished tasks, shuts down its worker pool, and
joins its loop thread. Do not reuse already-active async resources from another
loop. A synchronous call can be used in a notebook with an existing loop, but
blocks that calling thread until it completes.

Extension hooks, callbacks, and models
--------------------------------------------

Extension lifecycle hooks use ordinary ``async def`` functions. They send UI
events through ``await context.emit(...)``. SyncAgent runs them on its background event loop with
the original :class:`~zett_agent.AgentRunContext`, without a proxy or automatic
worker-thread adaptation. Await ``context.publish(event)`` and
``context.append_message(...)`` as usual. Extensions do not expose ``.sync()``;
their external-event ``accept`` method remains synchronous.

The :class:`~zett_agent.MiddlewareHook` inherited by every extension uses the
same rule. Its model and tool wrappers run on the Agent event loop, including
when the extension is registered through ``SyncAgent`` or
``create_agent_sync``.

AgentRunContext, AgentEventQueue, and the internal ModelOutputTracker remain async-only;
they are runtime implementation objects, not independent synchronous entry points.

Application-side dispatcher callbacks also use ``async def``:

.. code-block:: python

   class Instructions(AgentExtension):
       async def on_message(self, context):
           context.input_message = UserMessage(content="Explain: " + context.input_message.text)

   class Console(AgentEventDispatcher):
       async def on_text_delta_event(self, event):
           print(event.delta, end="", flush=True)

Callbacks are completed in stream order. They execute on the Agent event loop, not on the UI
thread; GUI applications must schedule visual changes through their own UI
dispatcher. Existing ``@tool`` functions already accept both synchronous and
asynchronous implementations. Use :class:`~zett_agent.SyncModelAdapter` for a
custom model that returns a normal iterator from ``def stream(request)``.

Await asynchronous I/O inside callbacks; offload blocking work explicitly when
needed so it does not stall the Agent event loop.
Closing the owning runtime or stream inside its own callback is rejected; let
the callback return and close it from the consuming thread instead.

External events and concurrency
-------------------------------

Use ``agent.emit_external_event(event, config=config)`` while a synchronous stream
is paused at an Ask User or Plan Mode event. Another thread may also deliver the
reply or close the stream while ``next()`` is waiting. The same runtime can serve
different sessions concurrently; overlapping requests for one session are rejected.
Each iterator supports one consumer. Blocking from the runtime's own event loop
raises an error to prevent deadlock; async code on that loop must use ``await``.

Complete offline example
------------------------

This example includes synchronous model output, callbacks, two requests, and
SQLite persistence in a temporary directory:

.. literalinclude:: ../_examples/synchronous.py
   :language: python

:download:`Download the example <../_examples/synchronous.py>`.

.. seealso::
   :doc:`streaming` covers event handling, :doc:`sessions` explains persistence,
   and :doc:`../extending/hooks` lists all lifecycle hooks.
