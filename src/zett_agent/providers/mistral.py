"""Mistral AI.

:class:`MistralProvider` speaks Chat Completions. Mistral validates a request against its schema and
refuses a field it does not list, so this adapter does not send ``stream_options``: the usage arrives
in the last chunk of a stream anyway.

Source: https://docs.mistral.ai/api/
"""

from __future__ import annotations

from typing import ClassVar

from ..model import ModelRequest
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject


class MistralProvider(OpenAICompatibleProvider):
    """Stream a Mistral model.

    Args:
        model: Mistral model identifier, such as ``mistral-medium-latest``.
        api_key: Mistral API key.
        base_url: API root; None uses ``https://api.mistral.ai/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last. Mistral's ``reasoning_effort``
            (``none``, ``minimal``, ``low``, ``medium`` or ``high``) and ``prompt_mode`` go here.

    Note:
        No reasoning field is sent, because Mistral models differ in whether they accept one and
        refuse a request that names a field they do not take.

    Examples:
        Stream a Mistral model with a reasoning level::

            model = MistralProvider(
                model="mistral-medium-latest",
                api_key=os.environ["MISTRAL_API_KEY"],
                extra_body={"reasoning_effort": "high"},
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "mistral"
    default_base_url: ClassVar[str | None] = "https://api.mistral.ai/v1"
    sends_stream_options: ClassVar[bool] = False

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Send no reasoning field; see the note on the class."""
        return {}
