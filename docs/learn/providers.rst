Connect a model provider
============================

Choose an adapter
---------------------

.. list-table:: Built-in provider choices
   :header-rows: 1
   :widths: 26 36 38

   * - Adapter
     - Endpoint
     - Application responsibility
   * - :class:`~zett_agent.OpenAIProvider`
     - Chat Completions or Responses API
     - Model name and API key; optional base URL and ``response=True``
   * - :class:`~zett_agent.DeepSeekProvider`
     - DeepSeek Chat Completions or Responses API
     - Model name and API key; optional ``response=True``
   * - :class:`~zett_agent.AnthropicProvider`
     - Anthropic-style API
     - Model name, API key, optional base URL
   * - :class:`~zett_agent.GoogleProvider`
     - Google GenAI SDK
     - Model name and API key
   * - :class:`~zett_agent.OllamaProvider`
     - Running Ollama server
     - Installed model and reachable server URL

Capabilities such as vision, reasoning, and tool choice depend on the model and
endpoint. Selecting a reasoning enum does not guarantee reasoning text will be
returned. No credentials are inferred by the Agent; your application supplies them.

Complete network example
----------------------------

This program makes a real request and may incur charges. Set ``OPENAI_MODEL``
and ``OPENAI_API_KEY`` securely in your terminal before running it. Do not paste
keys into source, command output, or documentation.

.. literalinclude:: ../_examples/real_provider.py
   :language: python
   :linenos:

:download:`Download real_provider.py <../_examples/real_provider.py>`.

.. code-block:: bash

   uv run python docs/_examples/real_provider.py

Retries and proxies
-----------------------

``RetryOptions(base_delay=0.5, max_delay=8, max_retries=3)`` permits one initial
attempt plus three retries, with delays of 0.5, 1, and 2 seconds. A longer series
would stop growing at 8 seconds. The adapter retries only before any stream event
has been emitted; failures halfway through output propagate to avoid replaying
visible text or tool fragments.

Default transports honor ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, and
``NO_PROXY``. A supplied custom transport controls its own routing.

.. code-block:: bash

   export HTTP_PROXY=http://127.0.0.1:8899
   export HTTPS_PROXY=http://127.0.0.1:8899

Always release the provider with ``await model.aclose()``. The client's lifetime
does not imply ownership of an adapter shared with other clients.

Prompt cache keys
--------------------

Endpoints that cache prompt prefixes route on a caller-supplied key. The Agent
fills :attr:`~zett_agent.ModelRequest.cache_key` from the run's session identity,
so every step of one conversation reuses the same prefix, and the OpenAI adapter
declares it as ``prompt_cache_key`` on both Chat Completions and Responses.
Reported hits are normalized into :attr:`~zett_agent.ModelUsage.cache_read_tokens`.

Pass ``cache_key`` to :class:`~zett_agent.AgentRunConfig` to route several
conversations onto one prefix, or construct a provider with
``send_prompt_cache_key=False`` when an endpoint rejects the field.

Implement your own adapter: :doc:`../extending/model-adapter`.
