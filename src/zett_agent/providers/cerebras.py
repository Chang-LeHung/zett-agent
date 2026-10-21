"""Cerebras Inference.

:class:`CerebrasProvider` speaks Chat Completions. It differs from the plain protocol in these ways:

* The reasoning level is a top-level ``reasoning_effort`` whose accepted values depend on the model:
  ``gpt-oss`` takes ``low``, ``medium`` and ``high`` and cannot switch reasoning off; the ``qwen`` and
  ``gemma`` models also take ``none``. Other models, ``kimi-k2.7-code`` among them, get no field.
* Reasoning comes back as ``reasoning`` and is sent back under the same name in the assistant message
  that produced it. Cerebras preserves it unless the request sets ``clear_thinking``.

Source: https://inference-docs.cerebras.ai/capabilities/reasoning
"""

from __future__ import annotations

from typing import ClassVar

from ..messages import AssistantMessage
from ..model import ModelRequest, ReasoningEffort
from .openai_compatible import ChatMessagePayload, OpenAICompatibleProvider, low_medium_high
from .wire import JsonObject


class CerebrasProvider(OpenAICompatibleProvider):
    """Stream a model served by Cerebras Inference.

    Args:
        model: Cerebras model identifier, such as ``gpt-oss-120b``, or the endpoint ID of a dedicated
            deployment.
        api_key: Cerebras API key.
        base_url: API root; None uses ``https://api.cerebras.ai/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. Cerebras uses it
            to route related requests together.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, so they add an option the
            adapter does not model, such as ``reasoning_format``, and may override one it does.

    Note:
        A dedicated deployment is named by an endpoint ID, so the model name cannot say which level
        names it takes. Give such an endpoint its level in ``extra_body``.

    Examples:
        Stream a model on Cerebras::

            model = CerebrasProvider(model="gpt-oss-120b", api_key=os.environ["CEREBRAS_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "cerebras"
    default_base_url: ClassVar[str | None] = "https://api.cerebras.ai/v1"

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Give ``gpt-oss``, ``qwen`` and ``gemma`` models a level they accept."""
        effort = request.reasoning_effort
        if self.model.startswith("gpt-oss"):
            return {"reasoning_effort": low_medium_high(effort)}
        if self.model.startswith(("qwen", "gemma")):
            return {"reasoning_effort": "none" if effort == ReasoningEffort.OFF else low_medium_high(effort)}
        return {}

    def replay_reasoning(self, message: AssistantMessage, payload: ChatMessagePayload) -> None:
        """Send the reasoning back as ``reasoning``, the name Cerebras returns it under."""
        if message.reasoning is not None and self._same_provider(message):
            payload["reasoning"] = message.reasoning
