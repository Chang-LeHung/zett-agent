import httpx

from ..model import DEFAULT_RETRY_OPTIONS, RetryOptions
from .base import _OpenAIStyleProvider


class OpenAIProvider(_OpenAIStyleProvider):
    """Map OpenAI-compatible chat streams to provider-neutral model events.

    Args:
        model: Model identifier understood by the configured endpoint.
        api_key: API credential supplied by the application.
        transport: Optional HTTPX transport, useful for isolated tests.
        base_url: Compatible API root; None uses the official OpenAI v1 endpoint.
        temperature: Optional sampling temperature passed to the provider.
        retry: Exponential backoff applied before the first emitted event only.

    Note:
        Await aclose() when finished. Model capabilities determine support for
        images, tool choice, and reasoning effort; this adapter does not infer
        unsupported features from the model name alone.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            transport=transport,
            temperature=temperature,
            retry=retry,
        )
        self.provider_name = "openai"
