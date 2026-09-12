Register provider-hosted tools
==================================

A server tool runs inside a model provider, unlike an ordinary ``AgentTool``
executed by the Agent loop. Install one provider-specific extension for each
hosted capability you want to expose::

    from zett_agent import GoogleServerToolExtension, create_agent
    from zett_agent import ServerToolDefinition

    client = await create_agent(
        model,
        extensions=[
            GoogleServerToolExtension([
                "google_search",
                "url_context",
            ]),
        ],
    )

Every extension calls :meth:`~zett_agent.AgentRunContext.register_server_tool`
during ``on_tool``. The resulting definitions are request-scoped and reach
every subsequent before-model hook and the final ``ModelRequest``. They never
enter the Agent's local tool executor.

Choose the extension that matches the endpoint
---------------------------------------------------

.. list-table:: Built-in declarations
   :header-rows: 1
   :widths: 30 28 42

   * - Extension
     - Default type
     - Transport requirement
   * - :class:`~zett_agent.GoogleServerToolExtension`
     - ``google_search`` + ``url_context``
     - Gemini GenerateContent; supported by GoogleProvider.
   * - :class:`~zett_agent.AnthropicServerToolExtension`
     - ``web_search_20260318`` + ``web_fetch_20260318``
     - Anthropic Messages; supported by AnthropicProvider.
   * - :class:`~zett_agent.OpenRouterServerToolExtension`
     - ``openrouter:web_search`` + ``openrouter:web_fetch``
     - OpenRouter-compatible Chat Completions endpoint.
   * - :class:`~zett_agent.OpenAIServerToolExtension`
     - ``web_search``
     - OpenAI Responses API; not OpenAI Chat Completions.
   * - :class:`~zett_agent.DeepSeekServerToolExtension`
     - ``web_search``
     - DeepSeek Responses API; not legacy Chat Completions.

OpenAIProvider and DeepSeekProvider currently implement Chat Completions. Their
extensions are provider-neutral declarations for a Responses-capable adapter;
they do not silently switch transport protocols. This keeps endpoint selection
explicit and avoids changing response semantics when a provider changes.

OpenAI and DeepSeek currently publish no separate ``web_fetch`` type. OpenAI's
hosted web-search workflow retrieves relevant pages itself; DeepSeek currently
documents search only. The defaults intentionally use provider-supported types
instead of sending an invented fetch discriminator.

Allow protocol evolution
-----------------------------

Tool types are open strings and configurations are opaque mappings. New provider
versions therefore do not require a zett-agent enum update::

    extension = AnthropicServerToolExtension(
        [
            ServerToolDefinition(
                "web_fetch_20260318",
                {
                    "name": "web_fetch",
                    "max_uses": 3,
                    "allowed_domains": ["docs.python.org"],
                },
            ),
            ServerToolDefinition("code_execution_20260521"),
        ],
    )

The provider remains responsible for model compatibility and field validation.
Duplicate types in one request are rejected before network I/O because their
ordering would otherwise make the active configuration ambiguous.

When none of the tools needs configuration, pass their type strings directly::

    extension = GoogleServerToolExtension([
        "google_search",
        "url_context",
        "code_execution",
    ])
