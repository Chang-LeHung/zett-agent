import httpx

from ..model import DEFAULT_RETRY_OPTIONS, RetryOptions
from .base import _OpenAIStyleProvider


class OpenAIProvider(_OpenAIStyleProvider):
    """OpenAI-compatible streaming adapter with event mapping to model-neutral objects."""

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
