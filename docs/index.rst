:html_theme.sidebar_secondary.remove:

.. meta::
   :description: Build Python agents with typed tools, streaming, persistent sessions, and on-demand skills. A small, provider-neutral runtime you can extend and embed.

.. rst-class:: zett-landing

Zett Agent
==========

.. container:: zett-hero

   .. container:: zett-hero-copy

      .. raw:: html

         <div class="zett-hero-identity">
           <img class="dark-light" src="_static/logo.svg" width="64" height="64" alt=""/>
           <span>YOUR AGENT. YOUR WAY.</span>
         </div>
         <p class="zett-hero-title">Small core.<br/><span>Yours to shape.</span></p>

      Build Python agents with your choice of model, ordinary functions as
      tools, and a stream your application can use. Add skills, persistence,
      and delegation when you need them.

      Your tools. Your instructions. Your choice of model.

      .. container:: zett-actions

         .. button-ref:: learn/first-agent
            :ref-type: doc
            :color: primary

            Build your first agent →

         .. button-ref:: examples/index
            :ref-type: doc

            Explore examples

      .. raw:: html

         <div class="zett-hero-meta">
           <span>Python 3.10+</span>
           <a href="https://github.com/Chang-LeHung/zett-agent/blob/main/LICENSE">MIT licensed</a>
           <a href="https://pypi.org/project/zett-agent/">Available on PyPI ↗</a>
         </div>

   .. container:: zett-hero-example

      .. raw:: html

         <div class="zett-code-caption"><span>YOUR MODEL. YOUR TOOLS.</span><span>Python</span></div>

      .. code-block:: python

         from zett_agent.client import create_agent
         from zett_agent.tools.base import tool

         @tool(guidelines="Use for exact integer addition.")
         def add(left: int, right: int) -> int:
             """Add two integers exactly."""
             return left + right

         client = await create_agent(model, tools=[add])
         reply = await client.run("What is 20 plus 22?")
         print(reply.content)

      .. container:: zett-code-note

         Use a configured model inside an async function. Start with
         :doc:`the complete, no-key example <learn/first-agent>`.

.. container:: zett-install

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: bash

            pip install zett-agent

      .. tab-item:: uv

         .. code-block:: bash

            uv add zett-agent

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

.. raw:: html

   <div class="zett-flow" role="img" aria-label="Your message goes to the model, which may call tools. Tool results return to the model, repeating until the model gives an answer.">
     <span class="zett-flow-step">Your message</span>
     <span class="zett-flow-arrow" aria-hidden="true">→</span>
     <span class="zett-flow-step">Model</span>
     <span class="zett-flow-arrow" aria-hidden="true">→</span>
     <span class="zett-flow-step zett-flow-tool">Tools, if needed</span>
     <span class="zett-flow-arrow" aria-hidden="true">↺</span>
     <span class="zett-flow-step">Model</span>
     <span class="zett-flow-arrow" aria-hidden="true">→</span>
     <span class="zett-flow-step">Answer</span>
   </div>

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
