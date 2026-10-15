from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    ModelRequest,
    ReasoningEffort,
    RetryOptions,
    TimeoutOptions,
)
from .base import _OpenAIStyleProvider

if TYPE_CHECKING:
    import httpx


class DeepSeekProvider(_OpenAIStyleProvider):
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
        extra_body: Mapping[str, Any] | None = None,
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

    def _provider_specific_request_fields(self, request: ModelRequest) -> dict[str, Any]:
        if request.reasoning_effort == ReasoningEffort.OFF or not self.model.startswith("deepseek-v4"):
            return {}
        match request.reasoning_effort:
            case ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
                effort = "low"
            case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
                effort = "max"
            case _:
                effort = "high"
        return {"reasoning_effort": effort}

    def _provider_specific_request_extra_fields(self, request: ModelRequest) -> dict[str, Any]:
        thinking = {"type": "enabled"} if request.reasoning_effort != ReasoningEffort.OFF else {"type": "disabled"}
        return {"thinking": thinking}
