from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..json_types import JsonValue
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    ModelRequest,
    ReasoningEffort,
    RetryOptions,
    TimeoutOptions,
)
from .anthropic_compatible import DEFAULT_MAX_TOKENS, AnthropicCompatibleProvider
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject

if TYPE_CHECKING:
    import httpx


class DeepSeekProvider(OpenAICompatibleProvider):
    """Stream DeepSeek through its OpenAI-compatible protocol.

    Args:
        model: DeepSeek model identifier accepted by the endpoint.
        api_key: API credential supplied by the application.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses the default DeepSeek v1 endpoint.
        temperature: Optional provider sampling temperature.
        response: Use DeepSeek's Responses-compatible endpoint instead of Chat Completions.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on:
            connecting, waiting for the response headers, and between chunks of a response
            that has started. See :class:`~zett_agent.model.TimeoutOptions`; a timeout raises
            :class:`~zett_agent.providers.base.ProviderTimeoutError` and is not retried.
        send_prompt_cache_key: Declare the run's prompt cache key on every
            request. Off by default: DeepSeek caches context automatically and
            its compatibility endpoint does not document the field. Enable it
            for gateways that route on the key.
        extra_headers: HTTP headers sent on every request after the adapter's
            own, for gateways that route or bill on a header. A header given
            here replaces the SDK default of the same name.
        extra_query: URL query parameters sent on every request, such as the
            ``api-version`` an Azure-style endpoint requires.
        extra_body: JSON fields merged into every request body last, so they
            add a vendor option the adapter does not model and may override one
            it does, such as ``thinking``.

    Note:
        Cache-hit counters are normalized into ModelUsage. Reasoning deltas
        are emitted only when returned by the provider. Returned reasoning is
        retained in AssistantMessage and replayed as reasoning_content in later
        requests, which is required by DeepSeek thinking mode. Close with
        aclose().

    Examples:
        Select the model explicitly rather than inferring it from the key::

            model = DeepSeekProvider(
                model="deepseek-chat",
                api_key=os.environ["DEEPSEEK_API_KEY"],
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()

    .. note::
        Reasoning effort is mapped only for models whose endpoint supports the
        corresponding request fields. A selected effort does not guarantee that
        the server will return reasoning deltas.

    .. seealso::
        :class:`~zett_agent.model.ReasoningEffort` lists provider-neutral levels, and
        :doc:`/learn/providers` covers shared provider lifecycle rules.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        send_prompt_cache_key: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or ("https://api.deepseek.com" if response else "https://api.deepseek.com/v1"),
            transport=transport,
            temperature=temperature,
            response=response,
            retry=retry,
            send_prompt_cache_key=send_prompt_cache_key,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        self.provider_name = "deepseek"

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Switch thinking on or off, and give the effort to the models that take one."""
        if request.reasoning_effort == ReasoningEffort.OFF:
            return {"thinking": {"type": "disabled"}}
        fields: JsonObject = {"thinking": {"type": "enabled"}}
        if self.model.startswith("deepseek-v4"):
            fields["reasoning_effort"] = _v4_effort(request.reasoning_effort)
        return fields

    def responses_fields(self, request: ModelRequest) -> JsonObject:
        """Spell DeepSeek's Responses reasoning object, whose ``none`` is the only off switch.

        Thinking is on by default on this endpoint, so leaving ``reasoning`` out does not disable
        it the way the Chat Completions ``thinking`` field does.
        """
        if request.reasoning_effort == ReasoningEffort.OFF:
            return {"reasoning": {"effort": "none"}}
        return {"reasoning": {"effort": _v4_effort(request.reasoning_effort)}}


def _v4_effort(effort: ReasoningEffort) -> str:
    """Map a provider-neutral level onto the three the V4 models know."""
    match effort:
        case ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
            return "low"
        case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return "max"
        case _:
            return "high"


class DeepSeekAnthropicProvider(AnthropicCompatibleProvider):
    """Stream DeepSeek through its Anthropic Messages protocol.

    Use it for a tool that speaks Messages, or to keep a conversation on the protocol the rest of the
    application uses. :class:`DeepSeekProvider` is the same service over Chat Completions.

    Args:
        model: DeepSeek model identifier, such as ``deepseek-v4-pro``. A name that starts with
            ``claude`` is accepted by the service and mapped to a DeepSeek model; any other name that is
            not DeepSeek's is a 400.
        api_key: API credential supplied by the application.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses ``https://api.deepseek.com/anthropic``.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        max_tokens: Output budget for one response.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last.

    Note:
        DeepSeek ignores ``budget_tokens`` and ``cache_control``, so neither is sent. Thinking is
        switched on or off with ``thinking``, and the level goes in ``output_config.effort`` for the V4
        models. Thinking blocks are sent back unchanged. Redacted thinking, documents and search
        results are not supported by the service.

    Examples:
        Stream a DeepSeek model over Messages::

            model = DeepSeekAnthropicProvider(
                model="deepseek-v4-pro",
                api_key=os.environ["DEEPSEEK_API_KEY"],
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "deepseek-anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str | None = None,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com/anthropic",
            transport=transport,
            retry=retry,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Switch thinking on or off, without a token budget, and give the V4 models their level."""
        if request.reasoning_effort == ReasoningEffort.OFF:
            return {"thinking": {"type": "disabled"}}
        fields: JsonObject = {"thinking": {"type": "enabled"}}
        if self.model.startswith("deepseek-v4"):
            fields["output_config"] = {"effort": _v4_effort(request.reasoning_effort)}
        return fields
