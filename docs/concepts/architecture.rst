Architecture and ownership
==============================

The runtime is intentionally small
--------------------------------------

An Agent alternates model calls and tool execution until it has a final answer
and no higher-priority input remains. It does not know about HTTP endpoints,
web components, database tables, or a specific LLM vendor.

.. code-block:: text

   Application / UI
          |
          v
     AgentClient ----------> AgentEventDispatcher ---> on_*_event callbacks
          |
          v
        Agent <-----------> AgentModel -----> provider SDK
          |                     |
          |                     +-- ModelEvent deltas + ModelResponse
          +-- AgentTool ----------> typed function
          +-- AgentExtension -----> context / storage / approval / compaction

What each object owns
-------------------------

.. list-table:: Responsibilities
   :header-rows: 1
   :widths: 26 74

   * - Object
     - Responsibility
   * - Agent
     - Request admission, phase transitions, the model/tool loop, and cleanup.
   * - AgentClient
     - Convenience creation and event-aware stream collection; exposes the runtime as client.agent.
   * - AgentModel
     - Provider requests, event normalization, and pre-output retry policy.
   * - AgentContext
     - Per-request references shared with extensions. Use it instead of global Agent state.
   * - AgentExtension
     - Optional behavior at documented hooks; default hooks do nothing.
   * - AgentEventDispatcher
     - Application-side callbacks; it does not inject context or register tools.
   * - SessionStorage
     - Durable raw records, snapshot boundaries, and restoration consistency.

Composition rather than a second runtime
--------------------------------------------

``create_agent`` returns a wrapper, not a different loop. You can wrap an existing
Agent with AgentClient, or use Agent directly with ``async for``. Supplying an
extension does not require subclassing Agent.

The application owns provider and database lifetimes. The runtime closes nested
streams on completion/cancellation but does not close a shared provider after
every request. Do not create clients at module import time in libraries.

Limits and security boundaries
----------------------------------

``max_iterations`` limits model calls per user/internal input. Tool round trips
consume that budget. ``max_internal_messages`` bounds queued internal continuations
per request. Neither limit is a spending cap or an OS permission boundary.

Filesystem, shell, and MCP tools execute with the permissions of their host
process or server. Tool guidance is not enforcement. Use explicit allow-lists,
resource limits, approval rules, and process isolation at the application boundary.
