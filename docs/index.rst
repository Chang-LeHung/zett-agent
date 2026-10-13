:html_theme.sidebar_secondary.remove:

.. meta::
   :description: Agents that forge the tools they need. A small Python runtime for tool-calling agents, with a JIT harness that lets a model write, reuse, and inspect its own tools in the middle of a run.

.. rst-class:: zett-landing

Zett Agent
==========

.. container:: zett-hero

   .. container:: zett-hero-copy

      .. container:: zett-hero-identity

         .. image:: _static/logo.svg
            :class: dark-light
            :width: 52
            :height: 52
            :alt: Zett Agent

         .. rst-class:: zett-hero-badge

         :doc:`Meet the JIT harness → <learn/jit-harness>`

      .. rst-class:: zett-hero-title

      Agents that forge **the tools they need.**

      Zett Agent is a small Python runtime for tool-calling agents. Its JIT
      harness lets a model write, reuse, and inspect its own tools in the
      middle of a run, so a long task stops paying for the same exploration
      twice.

      Bring your own model and ordinary Python functions. Add skills,
      persistence, and delegation when you need them.

      .. container:: zett-actions

         :doc:`Build your first agent → <learn/first-agent>`

         :doc:`Meet the JIT harness <learn/jit-harness>`

      .. rst-class:: zett-hero-meta

      * Python 3.10+
      * `MIT licensed <https://github.com/Chang-LeHung/zett-agent/blob/main/LICENSE>`__
      * `Available on PyPI ↗ <https://pypi.org/project/zett-agent/>`__

   .. container:: zett-hero-example

      .. rst-class:: zett-code-caption

      * Your application
      * Python

      .. code-block:: python

         from zett_agent.client import create_agent
         from zett_agent.extensions.memory import InMemoryMessageAccumulator
         from zett_agent.jit_harness import JITHarnessExtension

         client = await create_agent(
             model,
             tools=[add],
             extensions=[
                 InMemoryMessageAccumulator(),
                 JITHarnessExtension(),
             ],
         )
         await client.run("Build a tool that doubles 21, then use it.")

      .. rst-class:: zett-code-caption

      * Written by the model, mid-run
      * Python

      .. code-block:: python

         @tool(guidelines="Use to add an integer to itself.")
         async def twice(value: int) -> int:
             """Add the same integer twice."""
             return await harness.call_tool(
                 "add", left=value, right=value
             )

      .. container:: zett-code-note

         Use a configured model inside an async function. Start with
         :doc:`the complete, no-key example <learn/first-agent>`. Generated
         tools are trusted code, not a sandbox:
         :doc:`read the JIT harness guide <learn/jit-harness>`.

.. container:: zett-install

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: bash

            pip install zett-agent

      .. tab-item:: uv

         .. code-block:: bash

            uv add zett-agent

Let the model forge what it needs
---------------------------------

JIT here means building the tool environment when a task needs it, not a
Python just-in-time compiler. The harness works through ordinary function
calling, so it runs on Responses and Chat Completions alike.

.. container:: zett-steps

   .. container:: zett-step

      **Forge**

      ``create_tool`` registers a Python tool or a Markdown procedure the
      model just wrote. A Python tool can span several files that import each
      other.

   .. container:: zett-step

      **Reuse and compose**

      Tools call each other through ``harness.call_tool``. One outer call can
      drive a loop of nested calls whose results never enter the context.

   .. container:: zett-step

      **Inspect and unload**

      Source, profiles, and recent errors stay inspectable. Unload what the
      task no longer needs, and carry the toolbox to the next session in a
      snapshot.

:doc:`Read the JIT harness guide <learn/jit-harness>` for the control loop,
what the model sees, and the trust boundary.

Everything you need. Only when you need it.
------------------------------------------------

Keep the first agent simple. Add capabilities without changing how you run it.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Bring your model
      :link: learn/providers
      :link-type: doc

      OpenAI, Anthropic, Google, DeepSeek, or Ollama. Use the same client,
      tools, and event callbacks with every built-in adapter.

   .. grid-item-card:: Turn functions into tools
      :link: learn/tools
      :link-type: doc

      Write a typed Python function. Its signature and docstring become
      the tool definition; the runtime validates and executes calls.

   .. grid-item-card:: Stream the whole conversation
      :link: learn/streaming
      :link-type: doc

      Show incremental text, reasoning when available, tool progress, and
      results. Connect your own terminal, UI, or service.

   .. grid-item-card:: Load capabilities on demand
      :link: learn/on-demand
      :link-type: doc

      Keep skills in editable text files. Discover their instructions as
      needed, or search deferred tools on supported Responses endpoints.

Start here. Go further.
-----------------------

Follow a tutorial, find a complete example, or look up an exact interface.

.. container:: zett-paths

   .. container:: zett-path

      **01 / Get started**

      :doc:`Run your first agent <learn/first-agent>` without an API key.
      Then connect a provider, add tools, and stream a real answer.

   .. container:: zett-path

      **02 / Find working code**

      :doc:`Browse complete examples <examples/index>` for sessions,
      approvals, context compaction, delegation, and cancellation.

   .. container:: zett-path

      **03 / Make it yours**

      :doc:`Build an extension <extending/first-extension>` for your
      workflow. Learn how to add tools, instructions, and progress events.

   .. container:: zett-path

      **04 / Look it up**

      :doc:`Explore the API reference <reference/index>` for signatures,
      defaults, configuration fields, and examples.

Already building an application? Start with :doc:`learn/configuration` for
defaults and request overrides, :doc:`learn/application` for timeouts and
concurrent conversations, and :doc:`learn/testing` for deterministic tests.

A simple loop, from question to answer
--------------------------------------

The model can answer immediately or ask for tools. Zett Agent runs those
tools, returns their results, and calls the model again until it answers.

.. rst-class:: zett-flow

* Your message
* Model
* Tools, if needed
* Model
* Answer

History, approvals, and context management extend that same loop. You choose
what to enable; your application keeps control of its UI and resources.
See :doc:`how it works <concepts/architecture>` when you want the mental model.

.. toctree::
   :hidden:

   learn/index
   concepts/index
   extending/index
   examples/index
   reference/index
