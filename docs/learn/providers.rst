Connect a model provider
========================

Choose an adapter
-------------------

Every adapter implements the same protocol, so swapping one for another does not
touch the rest of your code.

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Adapter
     - Talks to
     - You supply
   * - :class:`~zett_agent.providers.openai.OpenAIProvider`
     - OpenAI Chat Completions or Responses
     - Model name and API key
   * - :class:`~zett_agent.providers.anthropic.AnthropicProvider`
     - Anthropic messages API
     - Model name and API key
   * - :class:`~zett_agent.providers.google.GoogleProvider`
     - Google GenAI
     - Model name and API key
   * - :class:`~zett_agent.providers.deepseek.DeepSeekProvider`
     - DeepSeek Chat Completions or Responses
     - Model name and API key
   * - :class:`~zett_agent.providers.ollama.OllamaProvider`
     - A running Ollama server
     - Installed model (and server URL if not localhost)

The adapter normalizes messages, tools, and events, not the capabilities of
every model. Pick an endpoint and model that support your workflow. In
particular, images, visible reasoning, hosted tools, and tool search are not
available on every endpoint. This runtime does not expose a general
``response_format`` or structured-final-output setting; for validated data,
use a typed tool or validate the final answer in your application.

Configure your chosen provider
------------------------------

Read credentials and model names from your environment. Every tab creates a
``model`` you can pass to ``create_agent``; close it with ``await model.aclose()``
when finished. These snippets configure real services and are not offline
examples.

.. tab-set::

   .. tab-item:: OpenAI

      .. code-block:: python

         import os
         from zett_agent.providers.openai import OpenAIProvider

         model = OpenAIProvider(
             model=os.environ["OPENAI_MODEL"],
             api_key=os.environ["OPENAI_API_KEY"],
             response=True,
         )

      ``response=True`` selects the Responses API. Omit it for Chat
      Completions (the default). Choose a protocol supported by your endpoint;
      enabling this flag does not add Responses support to a gateway.

   .. tab-item:: Anthropic

      .. code-block:: python

         import os
         from zett_agent.providers.anthropic import AnthropicProvider

         model = AnthropicProvider(
             model=os.environ["ANTHROPIC_MODEL"],
             api_key=os.environ["ANTHROPIC_API_KEY"],
         )

      Uses the Messages API. ``response=True`` is not supported.

   .. tab-item:: Google

      .. code-block:: python

         import os
         from zett_agent.providers.google import GoogleProvider

         model = GoogleProvider(
             model=os.environ["GOOGLE_MODEL"],
             api_key=os.environ["GOOGLE_API_KEY"],
         )

      Uses Google GenAI with an API key. The built-in constructor does not
      expose Vertex AI project/location configuration.

   .. tab-item:: DeepSeek

      .. code-block:: python

         import os
         from zett_agent.providers.deepseek import DeepSeekProvider

         model = DeepSeekProvider(
             model=os.environ["DEEPSEEK_MODEL"],
             api_key=os.environ["DEEPSEEK_API_KEY"],
         )

      Uses Chat Completions by default. ``response=True`` selects a
      Responses-compatible endpoint when your service supports it.

   .. tab-item:: Ollama

      .. code-block:: python

         import os
         from zett_agent.providers.ollama import OllamaProvider

         model = OllamaProvider(
             model=os.environ["OLLAMA_MODEL"],
             base_url="http://127.0.0.1:11434",
         )

      Start Ollama and install the selected model separately. Construction
      does not start a server or download a model. No API key is required
      for a local service; this adapter uses Ollama's chat protocol only.

A complete tool-calling request
------------------------------------------------------------

This program makes a real network request and may incur charges. Export the
variables in your shell rather than pasting keys into source or logs. The
program registers an addition tool, streams the response, and closes the
provider even if the request fails.

.. literalinclude:: ../_examples/real_provider.py
   :language: python
   :linenos:

:download:`Download real_provider.py <../_examples/real_provider.py>`.

.. code-block:: bash

   export OPENAI_MODEL="your-tool-capable-model"
   export OPENAI_API_KEY="your-api-key"
   uv run python docs/_examples/real_provider.py

From an installed package, download the file and run ``python real_provider.py``.
The exact wording and whether the model obeys the tool request depend on the
selected model. To verify the same tool round trip deterministically, run
:doc:`../examples/streaming-tools`.

Connect a compatible gateway
-----------------------------

Use ``base_url`` for a service implementing an existing protocol instead of
writing a new adapter:

.. code-block:: python

   model = OpenAIProvider(
       model=os.environ["GATEWAY_MODEL"],
       api_key=os.environ["GATEWAY_API_KEY"],
       base_url=os.environ["GATEWAY_BASE_URL"],
       send_prompt_cache_key=False,
   )

The URL is the API root, for example a service's ``.../v1``, not its
``.../chat/completions`` route. Match the protocol to the gateway. Some
gateways reject optional fields such as ``prompt_cache_key`` or particular
reasoning levels; configure only supported options and test your combination.
The adapter does not auto-detect model capabilities from an endpoint URL.

Reasoning effort
----------------

Set a provider-neutral level on the run instead of vendor-specific options:

.. code-block:: python

   from zett_agent.model import ReasoningEffort

   reply = await client.run("Plan the migration.", reasoning_effort=ReasoningEffort.HIGH)

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Level
     - Meaning
   * - ``MINIMAL``
     - Request a minimal reasoning budget when the model supports it.
   * - ``LOW`` / ``MEDIUM``
     - Lower or moderate effort; ``MEDIUM`` is the runtime default.
   * - ``HIGH`` / ``XHIGH``
     - Request more effort for difficult multi-step tasks.
   * - ``MAX`` / ``ULTRA``
     - Highest requested levels; an adapter may map them to the same supported setting.
   * - ``OFF``
     - Omit or disable explicit reasoning configuration according to the adapter.

The current Responses adapter maps ``XHIGH``, ``MAX``, and ``ULTRA`` to
``high``. Budget-based adapters also group the highest levels. OpenAI Chat
Completions forwards named levels, so a model that rejects one can still
raise an endpoint error. The runtime does not negotiate supported levels.

``OFF`` is not a universal guarantee of zero reasoning: OpenAI adapters omit
explicit effort configuration, while some other adapters disable thinking.
The endpoint's default may still reason. Raising the level never guarantees
visible reasoning text, a better answer, or a particular price.
Set a default for the whole agent with
``create_agent(model, reasoning_effort=...)`` and override it per request.

Retries and proxies
-------------------

``RetryOptions`` retries only *before* the first streamed event, so a failure
halfway through an answer never replays visible text or tool fragments.

.. code-block:: python

   OpenAIProvider(
       model=os.environ["OPENAI_MODEL"],
       api_key=os.environ["OPENAI_API_KEY"],
       retry=RetryOptions(base_delay=0.5, max_delay=8, max_retries=3),
   )

Default transports honor ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, and
``NO_PROXY``. Pass your own transport when you need custom routing.

``max_retries=3`` permits up to four transport attempts: the initial one plus
three retries. Authentication and validation failures are not blindly retried.
The policy belongs to the model adapter, not ``create_agent``. For a total
request deadline, see :doc:`application`; retries do not bound elapsed time.

Prompt caching
--------------

The runtime derives a prompt-cache key from the session, so every step of one
conversation can use the same routing hint on endpoints that support it. Cache
reuse and savings remain provider decisions, not runtime guarantees. Set
``cache_key`` on :class:`~zett_agent.agent.AgentRunConfig` to route several
conversations onto one prefix, or construct a provider with
``send_prompt_cache_key=False`` when an endpoint rejects the field.

OpenAI sends the key by default. DeepSeek omits it by default because its
first-party service caches automatically; enable it only for a gateway that
expects it. Other adapters may not expose cache routing.

When a provider reports cache hits, ``ModelUsage.cache_read_tokens`` and
``cache_hit_rate`` expose them. Requests through gateways that translate
protocols often omit the breakdown; ``ModelUsage.cache_reported`` is then
``False``, which distinguishes "unknown" from a measured 0% miss.

Always release the adapter
--------------------------

The client does not own a provider you passed in. Close it when the application
is done, even if several clients shared it:

.. code-block:: python

   await model.aclose()

Changing models mid-conversation
--------------------------------

Pass ``model=other_model`` to ``run`` or ``stream`` to override the default for
one request. Ordinary messages and local tools remain provider-neutral, but
vision support and hosted tools must still match the new endpoint. Signed
reasoning/replay blocks are retained only for the provider/model identity
that can consume them; do not treat them as portable instructions.

Next: :doc:`streaming`, or write your own adapter in
:doc:`../extending/model-adapter`.
