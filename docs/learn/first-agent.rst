Your first agent
================

**You will learn:** how to create a client, run more than one turn, and what
keeps the conversation between them. **Prerequisites:** :doc:`installation`. No
API credentials are needed.

This tutorial uses a scripted model so you can check the result on any machine.
It is not a language model and does not learn; it lets you experience the
runtime before choosing a provider. If you already have credentials, jump to
the :doc:`complete tool-calling request <providers>`.

1. Give the loop a model
------------------------

An agent does not know how to reach a specific vendor's HTTP API. It expects a
model adapter: an object that accepts a :class:`~zett_agent.model.ModelRequest`
and streams :class:`~zett_agent.model.ModelEvent` objects back.

For the first example, the adapter counts user turns and echoes the latest one.
It is not an LLM, but it plugs into exactly the same loop a real provider does.

.. literalinclude:: ../_examples/first_agent.py
   :language: python
   :pyobject: EchoModel

Every model step ends with one complete :class:`~zett_agent.model.ModelResponse`.
The text deltas before it exist for display; the final response is what the
runtime uses.

2. Create one client, run many turns
------------------------------------

.. literalinclude:: ../_examples/first_agent.py
   :language: python
   :pyobject: main

:func:`~zett_agent.client.create_agent` builds the runtime and returns an
:class:`~zett_agent.client.AgentClient`. Because no session is supplied, it
generates one for you.

The second request remembers the first because the default extensions keep
in-memory history for the session. ``run()`` returns the complete
:class:`~zett_agent.messages.AssistantMessage`.

3. Run the whole program
------------------------

:download:`Download first_agent.py <../_examples/first_agent.py>`.

.. literalinclude:: ../_examples/first_agent.py
   :language: python
   :linenos:
   :caption: A complete offline program

After saving the download as ``first_agent.py`` in an environment where you
installed Zett Agent::

    python first_agent.py

Or, from a checkout::

    uv run python docs/_examples/first_agent.py

Expected output::

    Turn 1: Hello
    Turn 2: Remember this conversation

What happened?
---------------

``create_agent`` initialized a conversation with default history. Each
``run`` appended a user message, called the adapter, and returned its final
assistant message. On the second call, the adapter counted two user messages
because the previous turn was restored. You did not need to pass history
manually or write a conversation loop.

Real providers can request tools before answering. The runtime executes those
tools and calls the model again with their results; a single ``run`` can
therefore involve several model steps. Try
:doc:`the deterministic tool round trip <../examples/streaming-tools>` to see
that behavior before making a paid request.

Next: switch the scripted model for a real provider
---------------------------------------------------

Only the adapter construction changes. The same client, messages, extensions,
tools, and event callbacks work with every built-in provider. Continue with
:doc:`providers`, then :doc:`streaming`.

.. warning::

   Passing ``extensions=[]`` removes the optional defaults, including in-memory
   history. Each request starts with fresh context; add
   :class:`~zett_agent.extensions.memory.InMemoryMessageAccumulator` or a
   persistence extension when you want the conversation to carry over.

Related API: :class:`~zett_agent.agent.Agent`,
:class:`~zett_agent.agent.AgentRunConfig`,
:class:`~zett_agent.extensions.memory.InMemoryMessageAccumulator`.
