Zett Agent documentation
============================

.. container:: zett-hero

   Build agents you can understand, extend, and embed.
   Zett Agent provides a small model/tool loop, typed streaming events, and
   extension hooks for tools, context, human input, and durable conversations.

.. container:: zett-paths

   .. container:: zett-path

      **New to Zett Agent?**

      :doc:`Run your first agent <learn/first-agent>` in five minutes, without
      an API key. Then connect a real provider and stream its output.

   .. container:: zett-path

      **Building an extension?**

      :doc:`Write a complete extension <extending/first-extension>` that
      registers a tool, transforms input, emits UI events, and cleans up state.

   .. container:: zett-path

      **Looking for working code?**

      :doc:`Browse runnable examples <examples/index>` for streaming,
      approval, SQLite, compaction, delegation, goals, and cancellation.

   .. container:: zett-path

      **Need a specific interface?**

      :doc:`Open the API reference <reference/index>` for model protocols,
      lifecycle hooks, storage contracts, schemas, and built-in tools.

Choose a learning path
--------------------------

* **Application developer:** first agent → real providers → streaming → sessions.
* **Extension author:** lifecycle → first extension → events → state and cleanup.
* **Integration author:** custom model → custom storage → API contracts.

All tutorial examples use the real Agent loop. Offline examples use scripted
model adapters to make results reproducible; they are not substitutes for an LLM.
Real-provider requests are opt-in and may incur provider charges.

Documentation contents
--------------------------

.. toctree::
   :maxdepth: 1

   learn/index
   concepts/index
   extending/index
   examples/index
   reference/index

Project status and documentation
------------------------------------

This is a development version; public APIs may change. API pages are generated
from the current checkout, not a separately installed release. Use
``make docs-serve`` at the repository root to build and preview this version.
See :doc:`the documentation guide <docstrings>` for contribution conventions.

The :ref:`genindex` and the sidebar search are useful when you know a symbol
name but not its module.
