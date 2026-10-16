"""Kimi, Moonshot AI's models.

Kimi speaks Chat Completions (:class:`KimiProvider`) and the Anthropic Messages protocol
(:class:`KimiAnthropicProvider`). On Chat Completions it differs from the plain protocol in these ways:

* Reasoning comes back as ``reasoning_content`` and must be sent back with the assistant message that
  produced it, in every later request of a multi-step conversation.
* How reasoning is controlled depends on the model: ``kimi-k3`` always reasons and takes a top-level
  ``reasoning_effort`` of ``low``, ``high`` or ``max``; ``kimi-k2.6`` and ``kimi-k2.7-code`` take a
  ``thinking`` object, whose ``keep: "all"`` makes the server use the reasoning that is sent back.
* A request refused by content moderation is an HTTP 400 with ``content_filter``; the runtime reports
  it as a refusal.

On Messages it authenticates with a bearer token, takes the reasoning level in ``output_config.effort``
and no ``thinking`` field, and honors only a top-level ``cache_control``.

Sources: https://platform.moonshot.ai/docs/api/chat and https://platform.moonshot.ai/docs/api/messages
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from ..json_types import JsonValue
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    ModelRequest,
    ModelUsage,
    ReasoningEffort,
    RetryOptions,
    TimeoutOptions,
)
from .anthropic_compatible import DEFAULT_MAX_TOKENS, AnthropicCompatibleProvider
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject, read_int, read_object

if TYPE_CHECKING:
    import httpx


def _k3_effort(effort: ReasoningEffort) -> str:
    """Map a provider-neutral level onto the three ``kimi-k3`` knows."""
    match effort:
        case ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
            return "low"
        case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return "max"
        case _:
            return "high"


class KimiProvider(OpenAICompatibleProvider):
    """Stream Kimi through its OpenAI-compatible protocol.

    Args:
        model: Kimi model identifier, such as ``kimi-k3`` or ``kimi-k2.6``.
        api_key: Moonshot API key.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses ``https://api.moonshot.ai/v1``. Accounts on the China
            platform use ``https://api.moonshot.cn/v1``.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, so they add an option the
            adapter does not model and may override one it does, such as ``thinking``.

    Note:
        Reasoning fields are sent only to the models that document them: ``reasoning_effort`` for
        ``kimi-k3``, ``thinking`` for ``kimi-k2.6`` and ``kimi-k2.7-code``. Any other model gets
        none, and ``extra_body`` is the way to give it one. ``kimi-k3`` always reasons and
        ``kimi-k2.7-code`` cannot switch thinking off, so an ``OFF`` effort is not sent to them.

    Examples:
        Stream a Kimi model::

            model = KimiProvider(model="kimi-k3", api_key=os.environ["MOONSHOT_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "kimi"

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
        send_prompt_cache_key: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.moonshot.ai/v1",
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
        """Spell the reasoning level the way this model takes it."""
        effort = request.reasoning_effort
        if self.model.startswith("kimi-k3"):
            # Always reasoning; the three levels it knows are the only choice, so there is no "off".
            if effort == ReasoningEffort.OFF:
                return {}
            return {"reasoning_effort": _k3_effort(effort)}
        if self.model.startswith(("kimi-k2.6", "kimi-k2.7")):
            if effort == ReasoningEffort.OFF:
                # kimi-k2.7-code is always on and rejects a switch that turns it off.
                return {} if self.model.startswith("kimi-k2.7-code") else {"thinking": {"type": "disabled"}}
            # Reasoning is sent back with every assistant turn, so ask the server to use it.
            return {"thinking": {"type": "enabled", "keep": "all"}}
        return {}


class KimiAnthropicProvider(AnthropicCompatibleProvider):
    """Stream Kimi through its Anthropic Messages protocol.

    Use it for a tool that speaks Messages, or to keep a conversation on the protocol the rest of the
    application uses. :class:`KimiProvider` is the same service over Chat Completions.

    Args:
        model: Kimi model identifier, such as ``kimi-k3``.
        api_key: Moonshot API key, sent as a bearer token.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses ``https://api.moonshot.ai/anthropic``.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_cache_key: Send the run's prompt cache key as ``metadata.user_id``, which Kimi uses to
            improve its cache hit rate. Use an opaque id, not a personal one.
        max_tokens: Output budget for one response, excluding thinking.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last.

    Note:
        ``kimi-k3`` always reasons, so an ``OFF`` effort sends nothing, and its level goes in
        ``output_config.effort``. No reasoning field is sent to any other model; ``extra_body`` is the
        way to give it one. Kimi reads only a top-level ``cache_control`` and caches by default, so no
        cache marker is put in the messages. Thinking blocks are kept with their signature and sent back
        unchanged.

    Examples:
        Stream a Kimi model over Messages::

            model = KimiAnthropicProvider(model="kimi-k3", api_key=os.environ["MOONSHOT_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "kimi-anthropic"
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
        send_cache_key: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self.send_cache_key = send_cache_key
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.moonshot.ai/anthropic",
            transport=transport,
            retry=retry,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Give ``kimi-k3`` its reasoning level, and the cache key as ``metadata.user_id``."""
        fields: JsonObject = {}
        if self.model.startswith("kimi-k3") and request.reasoning_effort != ReasoningEffort.OFF:
            fields["output_config"] = {"effort": _k3_effort(request.reasoning_effort)}
        if self.send_cache_key and request.cache_key:
            fields["metadata"] = {"user_id": request.cache_key}
        return fields

    def start_usage(self, usage: Mapping[str, JsonValue]) -> ModelUsage:
        """Read the usage counts, with the reasoning tokens Kimi reports under ``output_tokens_details``."""
        return self._with_reasoning(super().start_usage(usage), usage)

    def finish_usage(self, usage: ModelUsage, update: Mapping[str, JsonValue]) -> ModelUsage:
        """Fold in the final counts, keeping the reasoning tokens already read if the update has none."""
        return self._with_reasoning(super().finish_usage(usage, update), update)

    @staticmethod
    def _with_reasoning(usage: ModelUsage, source: Mapping[str, JsonValue]) -> ModelUsage:
        reasoning = read_int(read_object(source.get("output_tokens_details")).get("thinking_tokens"))
        if not reasoning:
            return usage
        # Reasoning is a subset of the output, and the output count is still 0 when the response starts.
        return replace(usage, reasoning_tokens=min(reasoning, usage.output_tokens))
