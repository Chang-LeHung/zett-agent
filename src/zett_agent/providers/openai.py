from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..model import DEFAULT_RETRY_OPTIONS, ModelRequest, RetryOptions
from .base import _OpenAIStyleProvider

if TYPE_CHECKING:
    import httpx


class OpenAIProvider(_OpenAIStyleProvider):
    """Map OpenAI-compatible chat streams to provider-neutral model events.

    Args:
        model: Model identifier understood by the configured endpoint.
        api_key: API credential supplied by the application.
        transport: Optional HTTPX transport, useful for isolated tests.
        base_url: Compatible API root; None uses the official OpenAI v1 endpoint.
        temperature: Optional sampling temperature passed to the provider.
        response: Use the Responses API instead of Chat Completions.
        retry: Exponential backoff applied before the first emitted event only.
        send_prompt_cache_key: Declare the run's prompt cache key on every
            request. Disable it when a compatible endpoint rejects the field.

    Note:
        Await aclose() when finished. Model capabilities determine support for
        images, tool choice, and reasoning effort; this adapter does not infer
        unsupported features from the model name alone.

    Examples:
        Read credentials from the environment and close the client explicitly::

            import os

            model = OpenAIProvider(
                model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
                api_key=os.environ["OPENAI_API_KEY"],
                retry=RetryOptions(max_retries=3),
            )
            try:
                client = await create_agent(model)
                reply = await client.run("Summarize this module")
            finally:
                await model.aclose()

    .. note::
        The default HTTP client honors ``HTTP_PROXY``, ``HTTPS_PROXY``,
        ``ALL_PROXY``, and ``NO_PROXY`` from the process environment.

    .. note::
        Both protocols declare ``prompt_cache_key`` from
        :attr:`~zett_agent.model.ModelRequest.cache_key`, so each conversation keeps a
        stable prefix for provider-side prompt caching. Reported cache hits are
        normalized into :attr:`~zett_agent.model.ModelUsage.cache_read_tokens`.

    .. seealso::
        :doc:`/learn/providers` covers credentials and transport ownership;
        :class:`~zett_agent.model.RetryOptions` controls retries before output starts.
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
        send_prompt_cache_key: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            transport=transport,
            temperature=temperature,
            response=response,
            retry=retry,
            send_prompt_cache_key=send_prompt_cache_key,
        )
        self.provider_name = "openai"

    def _provider_specific_request_fields(self, request: ModelRequest) -> dict[str, Any]:
        """Map the provider-neutral parallel preference to OpenAI's API."""
        return {
            **super()._provider_specific_request_fields(request),
            "parallel_tool_calls": request.parallel_tool_call,
        }
