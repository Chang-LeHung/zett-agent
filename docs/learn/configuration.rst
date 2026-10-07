Configure your agent
====================

Start with ``create_agent(model)``. Configure the behavior shared by a
conversation when you create the client; override only what changes for one
request when you call ``run`` or ``stream``.

This guide assumes a configured ``model`` from :doc:`providers`. The examples
below belong inside an async function.

Choose defaults once
--------------------

.. code-block:: python

   from zett_agent.agent import AgentRunConfig
   from zett_agent.client import create_agent
   from zett_agent.model import ReasoningEffort

   client = await create_agent(
       model,
       config=AgentRunConfig(session_id="review-42"),
       system_prompt="Review changes carefully. Explain risks before suggesting edits.",
       reasoning_effort=ReasoningEffort.MEDIUM,
       parallel_tool_call=True,
       max_iterations=12,
       max_internal_messages=4,
   )
   reply = await client.run("Review this function.")

``create_agent`` returns an initialized :class:`~zett_agent.client.AgentClient`.
Use ``client.agent`` for advanced runtime operations such as external input.
The client and direct :class:`~zett_agent.agent.Agent` API use the same loop;
the client additionally dispatches events to your bound callbacks.

.. list-table:: Settings and their scope
   :header-rows: 1
   :widths: 27 19 54

   * - Setting
     - Default
     - Effect
   * - ``system_prompt``
     - Helpful assistant
     - Application instructions rebuilt for every request.
   * - ``tools``
     - Empty
     - Named operations the model may request. See :doc:`tools`.
   * - ``extensions``
     - ``None``
     - Keep default history and tool guidance, or replace them with an explicit list.
   * - ``reasoning_effort``
     - ``MEDIUM``
     - A provider-neutral preference; support depends on the endpoint and model.
   * - ``parallel_tool_call``
     - ``True``
     - Permit concurrent local calls to parallel tools and request provider support where available.
   * - ``max_iterations``
     - ``36``
     - Maximum model calls per user or internal input, including calls after tool results.
   * - ``max_internal_messages``
     - ``8``
     - Bound internal continuations queued during a request.
   * - ``event_dispatcher``
     - ``None``
     - Optional callbacks awaited as the client consumes events.

Use a positive ``max_iterations`` and a nonnegative integer for
``max_internal_messages`` (``0`` disallows internal continuations). Exhaustion raises
:class:`~zett_agent.exceptions.AgentIterationLimitError`; it does not return a
successful partial answer. Neither limit caps money, tokens, wall-clock time,
or the number of tools in one response. Add an application timeout and tool
limits separately; see :doc:`application`.

Override one request
--------------------

.. code-block:: python

   reply = await client.run(
       "Analyze the difficult edge cases.",
       reasoning_effort=ReasoningEffort.HIGH,
       parallel_tool_call=False,
       metadata={"source": "review-form", "trace_id": "trace-7"},
       tags={"task": "code-review"},
   )

These values apply only to this invocation. You can also pass ``model=other_model``
to use another adapter for one request, or create the client with ``model=None``
and supply a model on every call. Your application still owns both adapters.
Changing a model does not make provider-specific reasoning state portable;
see :doc:`providers`.

The same options are accepted by ``stream``. ``compact`` accepts identity,
model, metadata, and tags, but does not start a normal user request.

Identity is separate from behavior
----------------------------------

.. code-block:: python

   reply = await client.run(
       "Continue the review.",
       config=AgentRunConfig(session_id="review-42", request_id="request-7"),
   )

Reuse ``session_id`` to address the same conversation. Omit ``config`` to use
the session chosen at client creation; if you never supplied one, the runtime
generated a UUIDv7 ID, available as ``event.session_id`` in the event stream.
For an application that needs to store and reopen the ID, choose it explicitly.

``request_id`` is an application correlation ID, **not an idempotency key**.
Sending the same ID twice does not deduplicate work or tool side effects.
``parent_session_id`` identifies a delegated conversation, while ``cache_key``
is a provider cache-routing hint, not history storage or tenant isolation.
See :doc:`sessions` for durable conversations.

Metadata is not a prompt
------------------------

Request ``metadata`` and ``tags`` must contain JSON-compatible data. They reach
extensions through ``context.metadata`` and ``context.tags`` and are recorded
beside persisted messages. Built-in providers do not send them to the model.

For model-visible information, write it into the user message or add explicit
instructions through an extension. Do not assume that placing an account,
language, or permission in metadata changes the model's behavior. Enforcement
belongs in your application or tool middleware.

Compose extensions deliberately
------------------------------------------------------------

``extensions=None`` enables
:class:`~zett_agent.extensions.memory.InMemoryMessageAccumulator` and
:class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension`.
Any explicit sequence replaces both defaults, including ``extensions=[]``.

.. tab-set::

   .. tab-item:: Process-local history

      .. code-block:: python

         from zett_agent.extensions.agents_md import AgentsMdExtension
         from zett_agent.extensions.memory import InMemoryMessageAccumulator
         from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

         client = await create_agent(
             model,
             extensions=[
                 InMemoryMessageAccumulator(),
                 ToolGuidelinesExtension(),
                 AgentsMdExtension(directory="workspace", include_parents=False),
             ],
         )

   .. tab-item:: Durable history

      .. code-block:: python

         from zett_agent.extensions.sqlite import SQLiteSessionExtension
         from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

         history = SQLiteSessionExtension("sessions.sqlite")
         try:
             client = await create_agent(
                 model,
                 extensions=[history, ToolGuidelinesExtension()],
             )
             reply = await client.run("Remember this conversation.")
         finally:
             await history.close()

Use one history-restoring extension, not both memory and SQLite. Extension
instances may contain state: reusing one instance intentionally shares that
state, while constructing a new in-memory accumulator does not restore an old
conversation. Built-ins run by priority, lowest first, with list order breaking
ties. You usually do not need to tune priorities; custom extensions can consult
:doc:`../extending/hooks`.

Next: :doc:`application` for integrating this configuration into a service, or
:doc:`project-instructions` for project-specific guidance.
