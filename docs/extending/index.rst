Extend the runtime
==================

Reach for an extension when behavior belongs **inside** the request lifecycle:
restoring history, reshaping context, adding tools, asking the user, or emitting
UI events. For display-only callbacks, use an
:class:`~zett_agent.dispatcher.AgentEventDispatcher` instead. You never subclass
:class:`~zett_agent.agent.Agent` or write a second model loop.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Write your first extension
      :link: first-extension
      :link-type: doc

      Register a tool, transform input, emit an event, and clean up.

   .. grid-item-card:: Built-in extensions
      :link: built-ins
      :link-type: doc

      Sessions, compaction, subagents, approvals, skills, MCP, and more.

   .. grid-item-card:: Lifecycle hooks
      :link: hooks
      :link-type: doc

      Every hook boundary and which one to override for a given job.

   .. grid-item-card:: Middleware
      :link: middleware
      :link-type: doc

      Wrap each provider request or tool call, in a documented order.

   .. grid-item-card:: Events and external input
      :link: events
      :link-type: doc

      Publish notifications and route responses back into a running request.

   .. grid-item-card:: State and cleanup
      :link: state
      :link-type: doc

      Keep per-request data safe when one instance serves many sessions.

   .. grid-item-card:: Model adapters
      :link: model-adapter
      :link-type: doc

      Add a provider or normalize a protocol the built-ins do not cover.

   .. grid-item-card:: Server-side tools
      :link: server-tools
      :link-type: doc

      Declare provider-hosted tools such as web search or code execution.

   .. grid-item-card:: Storage adapters
      :link: storage-adapter
      :link-type: doc

      Persist raw history and compaction checkpoints in your own store.

Suggested path: build your first extension, learn the event channels, then add
state and cancellation handling. Implement a new provider or storage adapter
only when an existing one cannot satisfy your application boundary.

.. toctree::
   :hidden:

   first-extension
   hooks
   middleware
   events
   state
   model-adapter
   server-tools
   storage-adapter
   built-ins
