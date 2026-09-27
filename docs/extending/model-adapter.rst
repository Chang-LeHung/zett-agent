Implement a model adapter
=============================

Use a custom adapter for a new protocol or a deterministic test double. For an
OpenAI-compatible HTTP service, first consider OpenAIProvider with a custom
base_url; a new adapter may be unnecessary.

The protocol
----------------

:class:`~zett_agent.model.AgentModel` accepts one fully assembled ModelRequest and
returns an async iterator of ModelEvent. It has a model-owned RetryOptions policy.
The protocol method is declared as a normal ``def`` returning AsyncIterator;
an implementation may use ``async def`` with ``yield`` to satisfy that contract.

.. code-block:: python

   from collections.abc import AsyncIterator
   from zett_agent.messages import AssistantMessage
   from zett_agent.model import (
       ModelEvent,
       ModelRequest,
       ModelResponse,
       RetryOptions,
   )

   class FixedModel:
       retry = RetryOptions(max_retries=0)

       async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
           yield ModelEvent.text("Hello")
           yield ModelEvent.completed(
               ModelResponse(AssistantMessage(content="Hello"))
           )

Required stream invariants
------------------------------

* Yield zero or more text, reasoning, or tool-call fragments.
* End a successful stream with exactly one completed ModelResponse.
* Include the complete answer and complete ToolCalls in that response, even when
  they were also streamed as deltas. The runtime does not reconstruct the final
  message from display text.
* Do not emit anything after the final response.
* Raise on transport/protocol failure; do not disguise failure as a successful empty answer.

ModelRequest contains the complete context for this step. Support the applicable
message roles and typed image parts explicitly. AgentMessage is internal input,
not assistant output; built-in adapters map it to user input for vendor APIs.
Declare :attr:`~zett_agent.model.ModelRequest.cache_key` to the endpoint when its
protocol caches prompt prefixes; the OpenAI adapter sends it as
``prompt_cache_key`` on both of its protocols.

Tool calls and usage
------------------------

Use ToolCallDelta for partial ID/name/argument fragments and ToolCall for final
parsed arguments. Correlate ToolMessage results by call ID, not list position.
Report ModelUsage on the final response; input includes cache reads/writes and
output includes reasoning when reported. Missing counters remain zero.

Replay signed vendor blocks only for the same provider/model identity. Preserve
them in AssistantMessage.replay_blocks instead of concatenating them into text.

Retry policy belongs here
-----------------------------

Retry transient failures only before an event escapes to the caller. Restarting
after partial text or tool deltas would duplicate visible output and could confuse
tool-call assembly. Validation/authentication failures should not be blindly retried.
Close SDK streams in finally/context managers when the consumer stops early.

Run a test adapter: :doc:`../learn/first-agent`. Inspect built-in implementations
under :doc:`../_generated/group-providers` for concrete vendor mappings.
