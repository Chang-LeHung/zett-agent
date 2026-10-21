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
   * - :class:`~zett_agent.providers.openrouter.OpenRouterProvider`
     - OpenRouter, one endpoint for many models
     - Model slug and API key
   * - :class:`~zett_agent.providers.kimi.KimiProvider`
     - Kimi (Moonshot AI) Chat Completions
     - Model name and API key
   * - :class:`~zett_agent.providers.glm.GLMProvider`
     - GLM (Z.ai) Chat Completions
     - Model name and API key
   * - :class:`~zett_agent.providers.xai.XAIProvider`
     - xAI (Grok) Chat Completions or Responses
     - Model name and API key
   * - :class:`~zett_agent.providers.minimax.MiniMaxProvider`
     - MiniMax Chat Completions
     - Model name and API key
   * - :class:`~zett_agent.providers.mistral.MistralProvider`
     - Mistral AI Chat Completions
     - Model name and API key
   * - :class:`~zett_agent.providers.groq.GroqProvider`, :class:`~zett_agent.providers.cerebras.CerebrasProvider`
     - Groq and Cerebras Chat Completions
     - Model name and API key
   * - :class:`~zett_agent.providers.hosts.TogetherProvider`, :class:`~zett_agent.providers.hosts.FireworksProvider`, :class:`~zett_agent.providers.hosts.SiliconFlowProvider`, :class:`~zett_agent.providers.hosts.DeepInfraProvider`
     - Together AI, Fireworks AI, SiliconFlow and DeepInfra Chat Completions
     - Model name and API key
   * - :class:`~zett_agent.providers.openai_compatible.OpenAICompatibleProvider`
     - Any service that speaks Chat Completions, with no vendor assumptions
     - Model name, API key, and ``base_url``
   * - :class:`~zett_agent.providers.deepseek.DeepSeekAnthropicProvider`, :class:`~zett_agent.providers.kimi.KimiAnthropicProvider`, :class:`~zett_agent.providers.glm.GLMAnthropicProvider`, :class:`~zett_agent.providers.minimax.MiniMaxAnthropicProvider`
     - The same four services over the Anthropic Messages protocol
     - Model name and API key
   * - :class:`~zett_agent.providers.anthropic_compatible.AnthropicCompatibleProvider`
     - Any service that speaks the Messages protocol, with no vendor assumptions
     - Model name, API key, and ``base_url``
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

      Uses Google GenAI with an API key. ``base_url`` points it at a proxy or
      a local test server. The built-in constructor does not expose Vertex AI
      project/location configuration.

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

   .. tab-item:: OpenRouter

      .. code-block:: python

         import os
         from zett_agent.providers.openrouter import OpenRouterProvider

         model = OpenRouterProvider(
             model=os.environ["OPENROUTER_MODEL"],
             api_key=os.environ["OPENROUTER_API_KEY"],
             site_url="https://example.com",
             app_name="My agent",
         )

      ``site_url`` and ``app_name`` are optional and identify your app to
      OpenRouter. Reasoning that a model returns in a form that cannot be shown
      as text is kept and sent back unchanged, which models that sign or encrypt
      their reasoning need across a tool call.

   .. tab-item:: Kimi

      .. code-block:: python

         import os
         from zett_agent.providers.kimi import KimiProvider

         model = KimiProvider(
             model=os.environ["KIMI_MODEL"],
             api_key=os.environ["MOONSHOT_API_KEY"],
         )

      The default endpoint is ``https://api.moonshot.ai/v1``. An account on the
      China platform passes ``base_url="https://api.moonshot.cn/v1"``; keys do
      not work across the two.

   .. tab-item:: GLM

      .. code-block:: python

         import os
         from zett_agent.providers.glm import GLMProvider

         model = GLMProvider(
             model=os.environ["GLM_MODEL"],
             api_key=os.environ["ZAI_API_KEY"],
         )

      The default endpoint is ``https://api.z.ai/api/paas/v4``. An account on the
      China platform passes
      ``base_url="https://open.bigmodel.cn/api/paas/v4"``. ``tool_stream=True``
      streams a tool call's arguments as they are written.

   .. tab-item:: xAI

      .. code-block:: python

         import os
         from zett_agent.providers.xai import XAIProvider

         model = XAIProvider(
             model=os.environ["XAI_MODEL"],
             api_key=os.environ["XAI_API_KEY"],
             response=True,
         )

      xAI calls Chat Completions its legacy endpoint, and its hosted web and X
      search tools are on Responses, which ``response=True`` selects. The level is
      sent only to ``grok-4.5`` and later.

   .. tab-item:: MiniMax

      .. code-block:: python

         import os
         from zett_agent.providers.minimax import MiniMaxProvider

         model = MiniMaxProvider(
             model=os.environ["MINIMAX_MODEL"],
             api_key=os.environ["MINIMAX_API_KEY"],
         )

      The default endpoint is ``https://api.minimax.io/v1``. An account on the
      China platform passes ``base_url="https://api.minimaxi.com/v1"``.
      ``MiniMaxAnthropicProvider`` is the same service over Messages.

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

:class:`~zett_agent.providers.openai.OpenAIProvider` speaks OpenAI's own dialect:
it sends ``parallel_tool_calls`` and a top-level ``reasoning_effort``. A gateway
that rejects either can use
:class:`~zett_agent.providers.openai_compatible.OpenAICompatibleProvider`
instead, which sends only the ``reasoning_effort`` and takes the same
``base_url``.

Services that need no adapter of their own
-------------------------------------------

These services speak a protocol the generic adapters already handle. Pass the
root as ``base_url``; give a model its reasoning options in ``extra_body``. The
roots below were checked to answer on the network, not exercised with a chat
request, so confirm the root, the model names, and the options in the service's
own documentation before relying on them.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Service
     - ``base_url``
     - Adapter
   * - Alibaba Cloud Model Studio (Qwen, international)
     - ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1``
     - ``OpenAICompatibleProvider``
   * - Alibaba Cloud Model Studio (Messages)
     - ``https://dashscope-intl.aliyuncs.com/apps/anthropic``
     - ``AnthropicCompatibleProvider``
   * - ByteDance Volcengine Ark
     - ``https://ark.cn-beijing.volces.com/api/v3``
     - ``OpenAICompatibleProvider``
   * - ByteDance Volcengine Ark (Messages)
     - ``https://ark.cn-beijing.volces.com/api/compatible``
     - ``AnthropicCompatibleProvider``
   * - Tencent Hunyuan
     - ``https://api.hunyuan.cloud.tencent.com/v1``
     - ``OpenAICompatibleProvider``
   * - Baidu Qianfan
     - ``https://qianfan.baidubce.com/v2``
     - ``OpenAICompatibleProvider``
   * - Xiaomi MiMo
     - ``https://api.xiaomimimo.com/v1``
     - ``OpenAICompatibleProvider``
   * - Xiaomi MiMo (Messages)
     - ``https://api.xiaomimimo.com/anthropic``
     - ``AnthropicCompatibleProvider``
   * - StepFun
     - ``https://api.stepfun.com/v1``
     - ``OpenAICompatibleProvider``
   * - Baichuan
     - ``https://api.baichuan-ai.com/v1``
     - ``OpenAICompatibleProvider``
   * - NVIDIA API Catalog
     - ``https://integrate.api.nvidia.com/v1``
     - ``OpenAICompatibleProvider``
   * - Perplexity
     - ``https://api.perplexity.ai/v1``
     - ``OpenAICompatibleProvider``
   * - Cohere (compatibility API)
     - ``https://api.cohere.ai/compatibility/v1``
     - ``OpenAICompatibleProvider``
   * - Hugging Face Inference Providers
     - ``https://router.huggingface.co/v1``
     - ``OpenAICompatibleProvider``
   * - Novita AI
     - ``https://api.novita.ai/openai``
     - ``OpenAICompatibleProvider``
   * - SambaNova
     - ``https://api.sambanova.ai/v1``
     - ``OpenAICompatibleProvider``
   * - Nebius AI Studio
     - ``https://api.studio.nebius.com/v1``
     - ``OpenAICompatibleProvider``
   * - Azure OpenAI
     - ``https://<resource>.openai.azure.com/openai/v1``
     - ``OpenAIProvider``
   * - LiteLLM proxy, Portkey, Helicone and other gateways
     - The gateway's own URL
     - ``OpenAICompatibleProvider`` or ``OpenAIProvider``

A service that has a hosted search tool of its own is a different matter:
:doc:`../extending/server-tools` lists the declarations the runtime ships, and a
string type plus a configuration mapping declares any other.

Write a provider for another compatible service
------------------------------------------------

Most services that speak Chat Completions bend it in a few places: the field
that carries reasoning, how thinking is switched on and off, an identifying
header. Each is a method that
:class:`~zett_agent.providers.openai_compatible.OpenAICompatibleProvider`
calls, so a provider for one more service is a subclass that overrides only the
places where the service differs:

.. code-block:: python

   from zett_agent.model import ModelRequest, ReasoningEffort
   from zett_agent.providers.openai_compatible import JsonObject, OpenAICompatibleProvider

   class AcmeProvider(OpenAICompatibleProvider):
       provider_name = "acme"
       reasoning_keys = ("thinking_text",)

       def __init__(self, model: str, api_key: str) -> None:
           super().__init__(model=model, api_key=api_key, base_url="https://api.acme.example/v1")

       def vendor_headers(self) -> dict[str, str]:
           return {"X-Acme-Client": "zett"}

       def request_fields(self, request: ModelRequest) -> JsonObject:
           on = request.reasoning_effort != ReasoningEffort.OFF
           return {"thinking": {"type": "enabled" if on else "disabled"}}

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Override it when
   * - ``vendor_headers``
     - The service wants headers on every request.
   * - ``request_fields``
     - The service spells the reasoning level, or takes other options, in the request body.
   * - ``replay_reasoning``
     - An earlier assistant turn must be sent back in a form of the service's own.
   * - ``read_delta`` and ``reasoning_keys``
     - A streamed delta carries reasoning, or other data, under names the plain protocol does not use.
   * - ``merge_replay``
     - Opaque blocks arrive in pieces that must be put together before they are sent back.
   * - ``read_usage``
     - The token counts use names that the default reader does not know.
   * - ``chat_stop``
     - The service has finish reasons of its own.
   * - ``status_error``
     - The service explains an HTTP failure in a body of its own.

A method that is not overridden does what the plain protocol does, so a provider
stays as small as the ways its service differs. The built-in OpenRouter, Kimi and
GLM providers are written this way.

The Messages protocol works the same way through
:class:`~zett_agent.providers.anthropic_compatible.AnthropicCompatibleProvider`,
which :class:`~zett_agent.providers.anthropic.AnthropicProvider` and the
Messages providers for DeepSeek, Kimi and GLM are built on. Its hooks are the
places where services differ most, and the plain Anthropic behavior is what a
hook does when it is not overridden:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Override it when
   * - ``vendor_headers``
     - The service wants headers on every request.
   * - ``request_fields``
     - The service takes the reasoning level in a field of its own, or rejects the adaptive thinking Anthropic uses.
   * - ``mark_cache``
     - The service ignores the cache markers Anthropic needs, or reads them somewhere else.
   * - ``start_usage`` and ``finish_usage``
     - The token counts carry more than Anthropic's, such as reasoning tokens.
   * - ``message_stop``
     - The service has stop reasons of its own.
   * - ``status_error``
     - The service explains an HTTP failure in a body of its own.

Two class attributes cover the rest: ``bearer_auth = True`` also sends the key as
``Authorization: Bearer``, for a service that takes only that, and
``native_deferred_tools`` stays ``False`` unless the service itself loads a
tool when a ``tool_reference`` names it, which only Anthropic does.

Send extra headers, query parameters, or body fields
---------------------------------------------------------

Gateways often route, bill, or version on something the adapters do not model.
Every adapter accepts ``extra_headers``; the OpenAI-style and Anthropic
adapters also take ``extra_query`` and ``extra_body``, Google takes
``extra_body`` and ``base_url``, and Ollama takes ``extra_query``.

.. code-block:: python

   model = OpenAIProvider(
       model=os.environ["GATEWAY_MODEL"],
       api_key=os.environ["GATEWAY_API_KEY"],
       base_url=os.environ["GATEWAY_BASE_URL"],
       extra_headers={"X-Tenant": "acme"},
       extra_query={"api-version": "2025-01-01"},
       extra_body={"service_tier": "flex"},
   )

Headers and query parameters go on every request after the adapter's own, so
one given here replaces the SDK default of the same name (use it for
``User-Agent`` or an ``anthropic-beta`` flag). ``extra_body`` is merged into
the request body last: it adds options the adapter does not model and may also
override ones it does, such as DeepSeek's ``thinking``. The values are copied
and validated when the provider is built, and ``extra_body`` must be JSON.

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
``high``. The Anthropic adapter sends adaptive thinking with the level as
``output_config.effort`` (``low``, ``medium``, ``high``, ``xhigh``, ``max``;
``MINIMAL`` shares ``low`` and ``ULTRA`` shares ``max``) and sets no token
budget. Google still turns the level into a ``thinking_budget``, which groups
the highest levels. OpenAI Chat
Completions forwards named levels, so a model that rejects one can still
raise an endpoint error. OpenRouter sends the level as ``reasoning.effort``;
Kimi and GLM send the three levels they know (``low``, ``high``, ``max``) and
only to the models that take them. xAI, MiniMax, Groq and Cerebras do the same
per model family, and Mistral and the plain hosting services send no level
because their models disagree on the names: pass one in ``extra_body``. xAI
cannot switch reasoning off, so ``OFF`` is sent as ``low``. The GLM Messages
adapter still sends the token budget Z.ai documents. The runtime does not
negotiate supported levels.

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
request deadline, see :doc:`application`. A silent provider is bounded by the
timeouts in the next section, and a timeout is not retried.

.. _provider-timeouts:

When the provider goes quiet
----------------------------

A provider that takes the request and never answers used to be waited on for ten
minutes, and then asked three more times; the Google and Ollama adapters waited
forever. Every adapter now gives up, and says which wait ran out.
:class:`~zett_agent.model.TimeoutOptions` has one limit for each stretch where
nothing may arrive:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Limit
     - Default
     - What it bounds
   * - ``connect``
     - 5 s
     - Opening the connection. A reachable host answers in milliseconds.
   * - ``first_byte``
     - 60 s
     - From sending the request to receiving the response *headers*. Streaming
       APIs answer the headers before the model has produced anything, so a long
       wait here means a gateway that took the request and did nothing.
   * - ``idle``
     - 600 s
     - The longest gap between two chunks once the response has started. It is
       long because a reasoning model may think silently.

.. code-block:: python

   from zett_agent.model import TimeoutOptions

   OpenAIProvider(
       model=os.environ["OPENAI_MODEL"],
       api_key=os.environ["OPENAI_API_KEY"],
       timeout=TimeoutOptions(first_byte=30, idle=900),
   )

There is deliberately no limit on the *total* time: a long answer that keeps
streaming is healthy. Set a limit to ``None`` to switch it off. A limit that
runs out raises :class:`~zett_agent.providers.base.ProviderTimeoutError`, whose
``phase`` is ``"connect"``, ``"first_byte"`` or ``"idle"``.

A timeout is **not retried**, except for a failed connection. Retrying a silent
provider waits the whole limit a second time, and the first request may still be
running, and billed, on the other side. If the response had already started when
it went quiet, what arrived was delivered, so a UI has shown it; the request then
fails rather than restarting.

.. note::
   The Google and Ollama clients send their request when the first chunk is
   asked for, so for them ``first_byte`` covers connecting, the response headers
   and the first chunk, and the connection has no limit of its own. Ollama's
   default allows ten minutes for that first chunk, because a local model may
   need to be loaded from disk first.

When the provider answers with a web page
-----------------------------------------

Between you and a provider there is often a reverse proxy, a gateway, a firewall
or a captive portal. When that fails it answers with an HTML page, a bare line of
text, or nothing, and a failure used to be reduced to "Provider stream request
failed". The status, the provider's own explanation, and the fact that it was a web
page are now in the message:

.. code-block:: text

   openai returned HTTP 400: model [gpt-6-Astra] service info not found
   openai answered HTTP 502 with a web page ("502 Bad Gateway") instead of an API
   response. A proxy, gateway, firewall or login page in front of the API usually
   does this; check the base_url and the network path to it.

Failures arrive as :class:`~zett_agent.providers.base.ProviderHTTPError`, which
has ``status_code``, ``reason`` (the provider's own words, when the body was a
JSON error), ``preview`` and ``is_html``. A rejected credential is still a
:class:`~zett_agent.providers.base.ProviderAuthError`, with the same detail added.
The SDK's own error stays on ``__cause__``.

A streaming request that is answered with a web page or with JSON is an error too,
whatever the status. A ``200`` with a login page, or with JSON because the gateway
does not stream, used to be read as a successful reply with no text; it is now
raised, naming the content type and what the body said. Any other content type is
read as a stream, because some gateways send one as ``text/plain``, and a stream
that ends without a single event is raised as an error.

Streams that end early or carry broken tool calls
-------------------------------------------------

Providers keep a request alive through three failures that used to end it.

* **Malformed or truncated tool arguments.** If a tool call's JSON is invalid,
  for example because the response hit its output limit or a gateway cut the
  stream, the call is not run. It becomes a failed tool result that says what
  happened and how to retry, so the model can resend a smaller call. Slightly
  malformed but complete JSON is still repaired silently. A cut-off payload is
  never completed by guesswork, because that could write truncated content.
  ``AnthropicProvider(max_tokens=...)`` sets the per-response output budget
  (default 16384); raise it for large writes, or lower it for a model with a
  smaller limit.
* **Responses ``error`` events.** A mid-stream ``error`` event (rate limit,
  overload) is raised with its own code and message instead of surfacing as a
  missing terminal response.
* **Responses streams without a terminal event.** If a gateway drops
  ``response.completed`` after every output item finished, the finished items
  are used and ``finish_reason`` is ``"incomplete"``. If an item never
  finished, or nothing arrived, the stream raised ``ProviderResponseError``;
  an empty stream is retried because nothing had been shown yet.

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
