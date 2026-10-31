from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..json_types import JsonValue
from ..model import DEFAULT_RETRY_OPTIONS, DEFAULT_TIMEOUT_OPTIONS, RetryOptions, TimeoutOptions, validate_response
from .anthropic_compatible import DEFAULT_MAX_TOKENS, AnthropicCompatibleProvider

if TYPE_CHECKING:
    import httpx


class AnthropicProvider(AnthropicCompatibleProvider):
    """Map Anthropic SDK streams, including signed thinking and tool blocks.

    Args:
        model: Model identifier accepted by the Anthropic-compatible endpoint.
        api_key: API credential supplied by the application.
        base_url: API root, defaulting to the official Anthropic endpoint.
        transport: Optional HTTPX transport for isolated tests or custom routing.
        response: Must remain false; this adapter uses Anthropic Messages API.
        retry: Exponential backoff for transient failures before output starts.
        timeout: How long the provider may stay silent before the request is given up on:
            connecting, waiting for the response headers, and between chunks of a response
            that has started. See :class:`~zett_agent.model.TimeoutOptions`; a timeout raises
            :class:`~zett_agent.providers.base.ProviderTimeoutError` and is not retried.
        prompt_cache: Mark cache breakpoints on the tool definitions, the system
            prompt, and the conversation so far. Anthropic caches only what a
            marker asks for, so disabling this trades prompt-cache reads for a
            payload without ``cache_control``. A request's ``cache_mode`` can
            narrow this to the stable tool and system prefix.
        max_tokens: Output budget for one response, thinking included. A tool
            call that writes a large file must fit in it: a response cut off by
            this limit leaves the call's JSON incomplete, and the model is then
            told to retry with a smaller call. Lower it for models with a
            smaller output limit.
        extra_headers: HTTP headers sent on every request, for gateways that
            route or bill on a header or for ``anthropic-beta`` flags. A header
            given here replaces the SDK default of the same name.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, so they
            add an option the adapter does not model and may override one it
            does.

    Note:
        Client-side tool search works here too: ``ToolSearchExtension`` ranks the
        deferred tools locally, and this adapter declares them with
        ``defer_loading`` and answers the search with ``tool_reference`` blocks.

    Note:
        Signed replay blocks belong to the originating provider and model.
        Persist them unchanged for tool round trips; do not render them as
        user-facing reasoning. Await aclose() to release client resources.

    Examples:
        Stream an Anthropic model through the provider-neutral client::

            model = AnthropicProvider(
                model=os.environ["ANTHROPIC_MODEL"],
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            try:
                client = await create_agent(model)
                reply = await client.run("Review the proposed API")
            finally:
                await model.aclose()

    .. warning::
        Signed thinking and tool replay blocks must be returned unchanged only
        to the same provider and model. They are not interchangeable with plain
        reasoning text.

    .. seealso::
        :class:`~zett_agent.messages.AssistantMessage` documents ``replay_blocks``;
        :doc:`/concepts/context` explains replay during tool round trips.
    """

    #: See :attr:`zett_agent.model.NativeDeferredToolsModel.native_deferred_tools`. The Messages
    #: API accepts ``defer_loading`` tools and loads one when a ``tool_reference``
    #: names it, so every deferred definition travels with the request.
    native_deferred_tools: bool = True

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        response: bool = False,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        prompt_cache: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
    ) -> None:
        validate_response(response)
        if response:
            raise ValueError("AnthropicProvider does not support the Responses API")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry=retry,
            timeout=timeout,
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )
