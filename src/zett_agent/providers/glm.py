"""GLM, Zhipu AI's models (Z.ai).

GLM speaks Chat Completions (:class:`GLMProvider`) and the Anthropic Messages protocol
(:class:`GLMAnthropicProvider`). On Chat Completions it differs from the plain protocol in these ways:

* Reasoning comes back as ``reasoning_content``.
* It is controlled by a ``thinking`` object (models from GLM-4.6 on) whose ``clear_thinking`` says
  whether the server drops the reasoning of earlier turns. Turning that off ("Preserved Thinking")
  needs the earlier reasoning to be sent back complete and in order, which this provider does.
* GLM-5.2 and later also take a top-level ``reasoning_effort``.
* ``tool_stream`` streams the arguments of a tool call as they are written instead of all at once.

On Messages it authenticates with a bearer token (``ANTHROPIC_AUTH_TOKEN`` in Claude Code's terms).

Source: https://docs.z.ai/api-reference/llm/chat-completion
"""

from __future__ import annotations

import re
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
from .base import reasoning_effort_to_budget
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject

if TYPE_CHECKING:
    import httpx

_VERSION = re.compile(r"^glm-(\d+)(?:\.(\d+))?", re.IGNORECASE)


def glm_version(model: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` of a GLM model name such as ``glm-4.6`` or ``glm-5``, else None."""
    found = _VERSION.match(model)
    return (int(found.group(1)), int(found.group(2) or 0)) if found else None


class GLMProvider(OpenAICompatibleProvider):
    """Stream GLM through its OpenAI-compatible protocol.

    Args:
        model: GLM model identifier, such as ``glm-5.1`` or ``glm-4.6``.
        api_key: Z.ai API key.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses ``https://api.z.ai/api/paas/v4``. Accounts on the China
            platform use ``https://open.bigmodel.cn/api/paas/v4``.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        clear_thinking: Let the server drop the reasoning of earlier turns. Off by default: an agent
            that calls tools is continuing one line of thought, and the server is sent the earlier
            reasoning in full to use it. Turn it on to spend fewer tokens on a plain chat.
        tool_stream: Stream the arguments of a tool call as they are generated. Only the GLM-4.6 series
            and later support it.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. Off by default:
            the GLM reference does not document the field.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, so they add an option the
            adapter does not model and may override one it does, such as ``thinking``.

    Note:
        ``thinking`` is sent only for GLM-4.6 and later, ``reasoning_effort`` only for GLM-5.2 and
        later; an older model gets neither. GLM-5.3 always thinks, so an ``OFF`` effort is not sent
        to it. The levels GLM takes are ``low``, ``high`` and ``max``; the others map to the nearest.

    Examples:
        Stream a GLM model::

            model = GLMProvider(model="glm-5.1", api_key=os.environ["ZAI_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "glm"

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
        retry: RetryOptions = DEFAULT_RETRY_OPTIONS,
        timeout: TimeoutOptions = DEFAULT_TIMEOUT_OPTIONS,
        clear_thinking: bool = False,
        tool_stream: bool = False,
        send_prompt_cache_key: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self.clear_thinking = clear_thinking
        self.tool_stream = tool_stream
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.z.ai/api/paas/v4",
            transport=transport,
            temperature=temperature,
            retry=retry,
            timeout=timeout,
            send_prompt_cache_key=send_prompt_cache_key,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Switch thinking on or off, give the effort to the models that take one, and stream tools."""
        fields: JsonObject = {}
        version = glm_version(self.model)
        if version is None or version < (4, 6):
            return fields
        if self.tool_stream:
            fields["tool_stream"] = True
        effort = request.reasoning_effort
        if effort == ReasoningEffort.OFF:
            # GLM-5.3 can only think; for it "off" has nowhere to go and is left out.
            if version < (5, 3):
                fields["thinking"] = {"type": "disabled"}
            return fields
        fields["thinking"] = {"type": "enabled", "clear_thinking": self.clear_thinking}
        if version >= (5, 2):
            fields["reasoning_effort"] = _effort(effort)
        return fields


def _effort(effort: ReasoningEffort) -> str:
    """Map a provider-neutral level onto the three GLM knows."""
    match effort:
        case ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
            return "low"
        case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return "max"
        case _:
            return "high"


class GLMAnthropicProvider(AnthropicCompatibleProvider):
    """Stream GLM through its Anthropic Messages protocol.

    Use it for a tool that speaks Messages, or to keep a conversation on the protocol the rest of the
    application uses. :class:`GLMProvider` is the same service over Chat Completions.

    Args:
        model: GLM model identifier, such as ``glm-5.1``.
        api_key: Z.ai API key, sent as a bearer token.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses ``https://api.z.ai/api/anthropic``. Accounts on the China
            platform use ``https://open.bigmodel.cn/api/anthropic``.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        max_tokens: Output budget for one response, excluding thinking.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last.

    Note:
        Thinking is requested with Anthropic's standard ``thinking`` field, for GLM-4.6 and later; an
        older model gets none. GLM-5.3 always thinks, so an ``OFF`` effort sends nothing to it. Z.ai
        documents only how Claude Code reaches this endpoint, not which Messages fields it reads, so
        use ``extra_body`` if your model needs another.

    Examples:
        Stream a GLM model over Messages::

            model = GLMAnthropicProvider(model="glm-5.1", api_key=os.environ["ZAI_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "glm-anthropic"
    bearer_auth = True

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
            base_url=base_url or "https://api.z.ai/api/anthropic",
            transport=transport,
            retry=retry,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Request thinking from the models that have it, and nothing from the ones that do not."""
        version = glm_version(self.model)
        if version is None or version < (4, 6):
            return {}
        if request.reasoning_effort == ReasoningEffort.OFF and version >= (5, 3):
            return {}  # only ever thinks
        if request.reasoning_effort == ReasoningEffort.OFF or request.tool_choice:
            return {"thinking": {"type": "disabled"}}
        # Unchanged from before the Anthropic default moved to adaptive thinking: Z.ai is still sent a budget.
        budget = reasoning_effort_to_budget(request.reasoning_effort)
        return {"max_tokens": budget + self.max_tokens, "thinking": {"type": "enabled", "budget_tokens": budget}}
