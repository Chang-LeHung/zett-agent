from typing import Any

import httpx

from ..model import DEFAULT_RETRY_OPTIONS, ModelRequest, ReasoningEffort, RetryOptions
from .base import _OpenAIStyleProvider


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
        :class:`~zett_agent.ReasoningEffort` lists provider-neutral levels, and
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
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or ("https://api.deepseek.com" if response else "https://api.deepseek.com/v1"),
            transport=transport,
            temperature=temperature,
            response=response,
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
