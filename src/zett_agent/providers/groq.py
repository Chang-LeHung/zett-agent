"""Groq's inference service.

:class:`GroqProvider` speaks Chat Completions. It differs from the plain protocol in these ways:

* ``gpt-oss`` models take a top-level ``reasoning_effort`` of ``low``, ``medium`` or ``high`` and
  cannot switch reasoning off. Other models get no field, because the accepted values differ per model.
* Reasoning comes back as ``reasoning`` and is sent back under the same name in the assistant message
  that produced it.

Source: https://console.groq.com/docs/reasoning
"""

from __future__ import annotations

from typing import ClassVar

from ..messages import AssistantMessage
from ..model import ModelRequest
from .openai_compatible import ChatMessagePayload, OpenAICompatibleProvider, low_medium_high
from .wire import JsonObject


class GroqProvider(OpenAICompatibleProvider):
    """Stream a model served by Groq.

    Args:
        model: Groq model identifier, such as ``openai/gpt-oss-120b``.
        api_key: Groq API key.
        base_url: API root; None uses ``https://api.groq.com/openai/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. Off by default,
            because Groq does not document the field.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, so they add an option the
            adapter does not model, such as ``reasoning_format``, and may override one it does.

    Examples:
        Stream a model on Groq::

            model = GroqProvider(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "groq"
    default_base_url: ClassVar[str | None] = "https://api.groq.com/openai/v1"
    sends_prompt_cache_key: ClassVar[bool] = False

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Give a ``gpt-oss`` model a level it accepts."""
        if "gpt-oss" in self.model:
            return {"reasoning_effort": low_medium_high(request.reasoning_effort)}
        return {}

    def replay_reasoning(self, message: AssistantMessage, payload: ChatMessagePayload) -> None:
        """Send the reasoning back as ``reasoning``, the name Groq returns it under."""
        if message.reasoning is not None and self._same_provider(message):
            payload["reasoning"] = message.reasoning
