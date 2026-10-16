Test your agent
===============

Use deterministic models to test runtime behavior and a small set of opt-in
live evaluations to test model quality. They answer different questions:
"Did my tool, routing, and cleanup work?" does not require a network request;
"Does this model reliably solve the task?" does.

Test tools as Python operations
-------------------------------

Decoration returns an ``AgentTool``, so call it with an argument mapping.
This exercises validation and the handler without asking a model to select it:

.. code-block:: python

   import pytest
   from pydantic import ValidationError
   from zett_agent.tools.base import tool

   @tool(guidelines="Use for exact integer addition.")
   def add(left: int, right: int) -> int:
       """Add two integers exactly."""
       return left + right

   @pytest.mark.asyncio
   async def test_add() -> None:
       assert await add({"left": 20, "right": 22}) == 42
       with pytest.raises(ValidationError):
           await add({"left": "not-an-integer", "right": 22})

Install ``pytest`` and ``pytest-asyncio`` in your test environment. Direct tool
calls propagate exceptions; inside an agent run, ordinary tool exceptions
become unsuccessful results. Test both boundaries for important integrations.

Script the model's decisions
------------------------------------------------------------

Implement the public :class:`~zett_agent.model.AgentModel` protocol to choose
exactly what the model returns in a test. There is no special mock-agent API:
the real runtime still restores history, validates tools, dispatches events,
and cleans up.

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: MathModel

This adapter first asks for ``add``, then answers using the returned tool
result. Its imports, tool, dispatcher, and assertions are in
:download:`streaming_tools.py <../_examples/streaming_tools.py>`.
Successful adapters must return a terminal ``ModelResponse`` containing the
complete message. Text deltas alone do not constitute a final answer.
See :doc:`../extending/model-adapter` for stream invariants.

Assert behavior, not incidental timing
------------------------------------------------------------

For a tool-calling workflow, verify the request includes the expected tool,
the handler receives the right arguments, the model receives its result, and
the client returns the expected answer. For concurrent tools, correlate by
tool-call ID rather than assuming completion order.

Extend the same test to cover:

* Invalid arguments and a handler error the model can recover from.
* Two isolated sessions and restored history after reopening storage.
* Early stream closure, request timeout, and request-local cleanup.
* A rejected or duplicate approval response, not only approval success.
* Budget exhaustion and a provider stream that fails after partial output.

Use events, queues, or explicit test barriers to synchronize concurrency.
Avoid timing-dependent sleeps. The examples in :doc:`../examples/index`
include assertions for many of these boundaries.

Test a real adapter without a real endpoint
------------------------------------------------------------

Inject ``httpx.MockTransport`` to test a built-in provider's request encoding
and streamed response handling:

.. literalinclude:: ../_examples/provider_mock.py
   :language: python

This program uses the actual OpenAI adapter with a local canned SSE response.
It asserts the request path and user message, so no credential or network
access is required. Run ``uv run python docs/_examples/provider_mock.py`` from
a checkout:

.. code-block:: text

   Mock provider reply: Hello from a mock endpoint.

:download:`Download provider_mock.py <../_examples/provider_mock.py>`.
Other providers expose the same ``transport`` injection point, but their wire
formats differ. A passing mock test proves your protocol fixture works, not
that a live endpoint supports every requested model feature.

Opt into checks against a real endpoint
------------------------------------------------------------

``tests/test_live_gateway.py`` runs the built-in adapters against a service
that speaks both protocols: the OpenAI Responses API at ``/responses`` and the
Anthropic Messages API at ``/v1/messages``. It covers what a fixture cannot:
that a real service accepts what an adapter sends, and that the adapter reads
what comes back, including provider-hosted tools, tool search on both the
client and the server, and computer use.

The checks skip unless two variables are set, so the default run stays
offline. Name the model for each protocol you want to exercise:

.. code-block:: text

   ZETT_AGENT_LIVE_TESTS=1 \
   ZETT_AGENT_GATEWAY_URL=http://127.0.0.1:8787 \
   ZETT_AGENT_GATEWAY_RESPONSES_MODEL=gpt-5.6-sol \
   ZETT_AGENT_GATEWAY_MESSAGES_MODEL='claude-sonnet-5[1m]' \
   ZETT_AGENT_GATEWAY_SEARCH_MODEL=kimi-k3 \
   uv run pytest tests/test_live_gateway.py

``ZETT_AGENT_GATEWAY_TOKEN`` supplies the credential (``test`` by default) and
``ZETT_AGENT_GATEWAY_TIMEOUT`` the per-request limit in seconds (180 by
default). A service that refuses a hosted tool type is expected, so that check
skips rather than fails; a stream that ends without a terminal response never
skips, because that is the adapter's bug to catch.

Keep tests safe and reproducible
------------------------------------------------------------

Use temporary directories for files and SQLite databases. Never let a test
construct a default user-data store or discover personal MCP/skill files.
Pass explicit paths, explicit server lists, and explicit skill roots.
Keep credentials absent from offline tests and opt into live calls separately.

The downloadable documentation programs are exercised by
``make docs-examples`` with network connections disabled and temporary working
directories. ``make check`` additionally checks formatting, the runtime
coverage gate, and a warning-free isolated documentation build.

Next: :doc:`application` for production boundaries, or
:doc:`../examples/cancellation` for a complete cleanup test.
