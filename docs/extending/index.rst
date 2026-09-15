Extend Zett Agent
=====================

Use an extension when behavior belongs **inside** the request lifecycle. For
display-only behavior, use AgentEventDispatcher instead. You do not need to
subclass Agent or implement another model/tool loop.

.. toctree::
   :maxdepth: 1

   first-extension
   hooks
   middleware
   events
   state
   model-adapter
   server-tools
   storage-adapter
   built-ins

Suggested path: build your first extension, learn the event channels, then add
state and cancellation handling. Only implement a new provider or storage adapter
when an existing one cannot satisfy the application boundary.
