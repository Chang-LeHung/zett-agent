"""Services that host open models behind a plain Chat Completions endpoint.

Each class here fixes the API root and the provider name, and sends no reasoning field. The services
host many models that disagree on whether they take one and on how they spell it, so a request that
names a field the model does not know is a 400. Give a model its level in ``extra_body``.
"""

from __future__ import annotations

from typing import ClassVar

from ..model import ModelRequest
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject


class _PlainHost(OpenAICompatibleProvider):
    """A hosting service that takes the Chat Completions protocol as it is."""

    sends_prompt_cache_key: ClassVar[bool] = False

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Send no reasoning field; see the module note."""
        return {}


class TogetherProvider(_PlainHost):
    """Stream a model hosted by Together AI.

    Args:
        model: Together model identifier, such as ``openai/gpt-oss-120b``.
        api_key: Together API key.
        base_url: API root; None uses ``https://api.together.xyz/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. Off by default,
            because Together does not document the field.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, such as a model's
            ``reasoning_effort``.

    Examples:
        Stream a model on Together::

            model = TogetherProvider(model="openai/gpt-oss-120b", api_key=os.environ["TOGETHER_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "together"
    default_base_url: ClassVar[str | None] = "https://api.together.xyz/v1"


class FireworksProvider(_PlainHost):
    """Stream a model hosted by Fireworks AI.

    Args:
        model: Fireworks model identifier, such as ``accounts/fireworks/models/gpt-oss-120b``.
        api_key: Fireworks API key.
        base_url: API root; None uses ``https://api.fireworks.ai/inference/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, such as a model's
            ``reasoning_effort``.

    Examples:
        Stream a model on Fireworks::

            model = FireworksProvider(
                model="accounts/fireworks/models/gpt-oss-120b",
                api_key=os.environ["FIREWORKS_API_KEY"],
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "fireworks"
    sends_prompt_cache_key: ClassVar[bool] = True
    default_base_url: ClassVar[str | None] = "https://api.fireworks.ai/inference/v1"


class SiliconFlowProvider(_PlainHost):
    """Stream a model hosted by SiliconFlow.

    Args:
        model: SiliconFlow model identifier, such as ``deepseek-ai/DeepSeek-V3.2``.
        api_key: SiliconFlow API key.
        base_url: API root; None uses ``https://api.siliconflow.cn/v1``. Accounts on the
            international platform use ``https://api.siliconflow.com/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. Off by default,
            because SiliconFlow does not document the field.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, such as ``enable_thinking``.

    Examples:
        Stream a model on SiliconFlow::

            model = SiliconFlowProvider(
                model="deepseek-ai/DeepSeek-V3.2",
                api_key=os.environ["SILICONFLOW_API_KEY"],
                extra_body={"enable_thinking": True},
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "siliconflow"
    default_base_url: ClassVar[str | None] = "https://api.siliconflow.cn/v1"


class DeepInfraProvider(_PlainHost):
    """Stream a model hosted by DeepInfra.

    Args:
        model: DeepInfra model identifier, such as ``deepseek-ai/DeepSeek-V3.2``.
        api_key: DeepInfra API key.
        base_url: API root; None uses ``https://api.deepinfra.com/v1/openai``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. Off by default,
            because DeepInfra does not document the field.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, such as a model's
            ``reasoning_effort``.

    Examples:
        Stream a model on DeepInfra::

            model = DeepInfraProvider(
                model="deepseek-ai/DeepSeek-V3.2",
                api_key=os.environ["DEEPINFRA_API_KEY"],
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "deepinfra"
    default_base_url: ClassVar[str | None] = "https://api.deepinfra.com/v1/openai"
