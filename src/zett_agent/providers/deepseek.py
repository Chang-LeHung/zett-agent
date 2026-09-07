from typing import Any

import httpx

from ..model import DEFAULT_RETRY_OPTIONS, ModelRequest, ReasoningEffort, RetryOptions
from .base import _OpenAIStyleProvider


class DeepSeekProvider(_OpenAIStyleProvider):
    """DeepSeek streaming adapter implemented through OpenAI-compatible protocol."""

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
            base_url=base_url or "https://api.deepseek.com/v1",
            transport=transport,
            temperature=temperature,
            retry=retry,
        )
        self.provider_name = "deepseek"

    def _provider_specific_request_fields(self, request: ModelRequest) -> dict[str, Any]:
        if request.reasoning_effort == ReasoningEffort.OFF or not self.model.startswith("deepseek-v4"):
            return {}
        match request.reasoning_effort:
            case ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
                effort = "low"
            case ReasoningEffort.XHIGH:
                effort = "max"
            case _:
                effort = "high"
        return {"reasoning_effort": effort}

    def _provider_specific_request_extra_fields(self, request: ModelRequest) -> dict[str, Any]:
        thinking = {"type": "enabled"} if request.reasoning_effort != ReasoningEffort.OFF else {"type": "disabled"}
        return {"thinking": thinking}
