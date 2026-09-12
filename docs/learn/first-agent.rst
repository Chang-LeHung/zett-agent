Your first agent
====================

**You will learn:** create a client, run two turns, and understand which component
keeps history. **Prerequisites:** :doc:`installation`; no API credentials.

1. Supply a model adapter
-----------------------------

An Agent does not know how to send an HTTP request to a specific model vendor.
It expects an adapter that accepts :class:`~zett_agent.ModelRequest` and streams
:class:`~zett_agent.ModelEvent`. For this first example, the adapter counts user
turns instead of calling an LLM.

.. literalinclude:: ../_examples/first_agent.py
   :language: python
   :pyobject: EchoModel

Each model step ends with one complete response. Text deltas are useful for
display, but do not replace that final response.

2. Create once, run more than once
--------------------------------------

.. literalinclude:: ../_examples/first_agent.py
   :language: python
   :pyobject: main

:func:`~zett_agent.create_agent` generates a UUIDv7 session when config is
omitted. The returned :class:`~zett_agent.AgentClient` wraps a normal Agent.
Its default extensions retain in-memory history, so the second request includes
the first exchange. The ``run`` method returns a complete AssistantMessage.

3. Run the complete program
-------------------------------

:download:`Download first_agent.py <../_examples/first_agent.py>`.

.. literalinclude:: ../_examples/first_agent.py
   :language: python
   :linenos:
   :caption: A complete offline program

From the repository root::

    uv run --directory backend/zett-agent python docs/_examples/first_agent.py

Expected output::

    Turn 1: Hello
    Turn 2: Remember this conversation

What changes with a real model?
-----------------------------------

Only the adapter construction needs to change. The same client, messages,
extensions, tools, and event callbacks work with the built-in providers.
Continue with :doc:`providers`, then :doc:`streaming`.

.. warning::

   Passing ``extensions=[]`` deliberately removes optional defaults, including
   in-memory history. A new request always starts with fresh AgentState; an
   extension must restore previous conversation messages.

Related API: :class:`~zett_agent.Agent`, :class:`~zett_agent.AgentRunConfig`,
:class:`~zett_agent.InMemoryMessageAccumulator`.
