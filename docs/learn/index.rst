Get started
===========

Zett Agent runs one loop: send messages to a model, run the tools it asks for,
and repeat until it answers. These pages take you from a zero-key example to a
real provider with streaming, tools, and persistent sessions.

Build your first conversation
------------------------------------------------------------

**Suggested path:** :doc:`installation` → :doc:`first-agent` →
:doc:`providers` → :doc:`tools` → :doc:`streaming` → :doc:`sessions`.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Install
      :link: installation
      :link-type: doc

      Install from PyPI, or run every example from a checkout.

   .. grid-item-card:: Your first agent
      :link: first-agent
      :link-type: doc

      Run two turns without an API key and see what keeps the history.

   .. grid-item-card:: Connect a provider
      :link: providers
      :link-type: doc

      Point the loop at OpenAI, Anthropic, Google, DeepSeek, or Ollama.

   .. grid-item-card:: Define tools
      :link: tools
      :link-type: doc

      Turn a typed Python function into a model-callable tool.

   .. grid-item-card:: Stream to a UI
      :link: streaming
      :link-type: doc

      Show partial text, reasoning, tool calls, and results as they happen.

   .. grid-item-card:: Keep conversations
      :link: sessions
      :link-type: doc

      Restore history across requests, processes, and restarts.

Add the capabilities you need
------------------------------------------------------------

These features are optional. Choose them for your task rather than enabling
every extension from the start.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Project instructions
      :link: project-instructions
      :link-type: doc

      Load AGENTS.md guidance with an explicit workspace scope.

   .. grid-item-card:: Image inputs
      :link: images
      :link-type: doc

      Send text and images to a vision-capable model with typed content parts.

   .. grid-item-card:: MCP tools
      :link: mcp
      :link-type: doc

      Connect trusted HTTP or stdio servers and manage their tool lifecycle.

   .. grid-item-card:: Capabilities on demand
      :link: on-demand
      :link-type: doc

      Load skill instructions as needed and search deferred tools on supported endpoints.

Build an application
------------------------------------------------------------

For a service, follow :doc:`configuration` → :doc:`application` →
:doc:`testing`. Blocking scripts can start with :doc:`synchronous`.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Configure your agent
      :link: configuration
      :link-type: doc

      Choose defaults, override one request, and compose extensions safely.

   .. grid-item-card:: Synchronous calls
      :link: synchronous
      :link-type: doc

      Use the same runtime from ordinary blocking code and scripts.

   .. grid-item-card:: Integrate an application
      :link: application
      :link-type: doc

      Handle deadlines, concurrent sessions, disconnects, and shutdown.

   .. grid-item-card:: Test your agent
      :link: testing
      :link-type: doc

      Check tools and real provider adapters without network access.

Something unexpected? :doc:`troubleshooting` covers missing history, tool
registration, endpoint compatibility, project guidance, and stuck approvals.
If you need behavior the built-ins do not provide, follow
:doc:`the extension tutorial <../extending/first-extension>`.

.. toctree::
   :hidden:

   installation
   first-agent
   providers
   tools
   streaming
   sessions
   images
   project-instructions
   on-demand
   mcp
   configuration
   synchronous
   application
   testing
   troubleshooting
