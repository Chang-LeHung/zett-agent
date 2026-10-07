How the pieces fit
==================

One loop, several useful boundaries
------------------------------------------------------------

An agent alternates model calls and tool execution until it has a final answer.
Providers, tools, and extensions supply the capabilities around that loop.
Your application can adopt them separately without adopting a web server,
UI, or another application's data model.

.. code-block:: text

   your application / UI
          │
          ▼
     AgentClient ─────────▶ AgentEventDispatcher ──▶ your callbacks
          │
          ▼
        Agent ◀───────────▶ AgentModel ──────────▶ provider SDK
          │
          ├── AgentTool ───────▶ your typed function
          └── AgentExtension ──▶ context, sessions, approvals, compaction

Who owns what
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Responsibility
   * - :class:`~zett_agent.agent.Agent`
     - Admits requests, runs the model/tool loop, cleans up.
   * - :class:`~zett_agent.client.AgentClient`
     - Convenient creation and event handling; exposes the runtime as ``client.agent``.
   * - :class:`~zett_agent.model.AgentModel`
     - Talks to one provider and normalizes its stream.
   * - :class:`~zett_agent.agent.AgentRunContext`
     - Per-request state shared with extensions; use it instead of global state.
   * - :class:`~zett_agent.extensions.base.AgentExtension`
     - Optional behavior at documented hooks; defaults do nothing.
   * - :class:`~zett_agent.dispatcher.AgentEventDispatcher`
     - Your callbacks for a stream; it never injects context or runs tools.

Composition, not a second runtime
---------------------------------

``create_agent`` wraps an Agent; it does not create a different loop. You can use
:class:`~zett_agent.agent.Agent` directly with ``async for``, wrap an existing
Agent in a client, and add behavior through extensions without subclassing the
Agent.

Your application owns the resources it passes in. The runtime closes the streams
it opens, but it never closes a provider or database you supplied — release those
yourself when the application shuts down.

Which boundary should I use?
------------------------------------------------------------

* Use a provider adapter to connect an endpoint, not to manage conversation history.
* Use a tool for an operation the model may request, not a second model loop.
* Use an extension for context, permissions, persistence, or behavior during a request.
* Use a dispatcher for presentation and telemetry that only observe UI events.

For example, a review assistant can compose a provider, read-only tools,
project guidance, and session storage. Its UI displays the resulting events
without needing to know how those capabilities are implemented. See
:doc:`../learn/configuration` for composition and
:doc:`../extending/first-extension` when the built-ins do not cover your workflow.

Limits and safety
-----------------

* ``max_iterations`` caps model calls per user or internal input. Tool round
  trips consume that budget.
* ``max_internal_messages`` caps queued internal continuations in one request.
* Neither limit is a spending cap or a permission boundary.

Filesystem, shell, and MCP tools run with the permissions of your process or
server. Tool guidance is documentation, not enforcement. Guard untrusted input
with explicit allow-lists, approval rules, resource limits, and process
isolation.

Next: :doc:`lifecycle` for the exact order of hooks inside one request.
