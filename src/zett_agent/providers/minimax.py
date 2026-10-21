"""MiniMax's models.

MiniMax speaks Chat Completions (:class:`MiniMaxProvider`) and the Anthropic Messages protocol
(:class:`MiniMaxAnthropicProvider`). How a model reasons depends on its generation:

* ``MiniMax-M3.1`` and later always reason, take the level as ``reasoning_effort`` (Chat
  Completions) or ``output_config.effort`` (Messages), and answer ``400`` to ``thinking: disabled``.
  Without a level they use ``max``.
* ``MiniMax-M3`` reasons when ``thinking`` is ``adaptive``. It takes no level.
* The ``MiniMax-M2`` series always reasons and takes neither.

On Chat Completions the reasoning text comes back inside ``content`` between tags unless the request
sets ``reasoning_split``, which this adapter always does so that it arrives as ``reasoning_content``
and can be sent back with the assistant message that produced it.

Sources: https://platform.minimax.io/docs/api-reference/text-openai-api and
https://platform.minimax.io/docs/api-reference/text-anthropic-api
"""

from __future__ import annotations

import re
from typing import ClassVar

from ..model import ModelRequest, ReasoningEffort
from .anthropic_compatible import AnthropicCompatibleProvider, messages_effort
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject

_VERSION = re.compile(r"minimax-m(\d+)(?:\.(\d+))?", re.IGNORECASE)


def _minimax_version(model: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` of a MiniMax model name such as ``MiniMax-M3.1-Flash``, else None."""
    found = _VERSION.match(model)
    return (int(found.group(1)), int(found.group(2) or 0)) if found else None


def _takes_effort(model: str) -> bool:
    version = _minimax_version(model)
    return version is not None and version >= (3, 1)


def _can_adapt(model: str) -> bool:
    """Whether ``thinking: adaptive`` and ``thinking: disabled`` are both something the model answers to."""
    return _minimax_version(model) == (3, 0)


class MiniMaxProvider(OpenAICompatibleProvider):
    """Stream MiniMax through its OpenAI-compatible protocol.

    Args:
        model: MiniMax model identifier, such as ``MiniMax-M3.1-Flash``.
        api_key: MiniMax API key.
        base_url: API root; None uses ``https://api.minimax.io/v1``. Accounts on the China platform
            use ``https://api.minimaxi.com/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last, so they add an option the
            adapter does not model and may override one it does.

    Note:
        ``reasoning_effort`` is sent to ``MiniMax-M3.1`` and later, and ``thinking`` to
        ``MiniMax-M3``. An ``OFF`` level disables thinking only on ``MiniMax-M3``, the one model that
        can switch it off.

    Examples:
        Stream a MiniMax model::

            model = MiniMaxProvider(model="MiniMax-M3.1-Flash", api_key=os.environ["MINIMAX_API_KEY"])
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "minimax"
    default_base_url: ClassVar[str | None] = "https://api.minimax.io/v1"

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Ask for the reasoning as its own field, and give the model its level or its thinking switch."""
        fields: JsonObject = {"reasoning_split": True}
        if request.reasoning_effort == ReasoningEffort.OFF:
            if _can_adapt(self.model):
                fields["thinking"] = {"type": "disabled"}
        elif _takes_effort(self.model):
            fields["reasoning_effort"] = messages_effort(request.reasoning_effort)
        elif _can_adapt(self.model):
            fields["thinking"] = {"type": "adaptive"}
        return fields


class MiniMaxAnthropicProvider(AnthropicCompatibleProvider):
    """Stream MiniMax through its Anthropic Messages protocol.

    Use it for a tool that speaks Messages, or to keep a conversation on the protocol the rest of the
    application uses. :class:`MiniMaxProvider` is the same service over Chat Completions.

    Args:
        model: MiniMax model identifier, such as ``MiniMax-M3.1-Flash``.
        api_key: MiniMax API key.
        base_url: API root; None uses ``https://api.minimax.io/anthropic``. Accounts on the China
            platform use ``https://api.minimaxi.com/anthropic``.
        transport: Optional HTTPX transport for custom routing or tests.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        prompt_cache: Mark cache breakpoints. Off by default.
        max_tokens: Output budget for one response. Thinking is paid out of it, so a value that is too
            small ends the response with no text.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last.

    Note:
        The level goes in ``output_config.effort`` for ``MiniMax-M3.1`` and later, and
        ``thinking: adaptive`` switches ``MiniMax-M3`` on. No token budget is sent. Thinking blocks
        are sent back unchanged, as MiniMax requires.

    Examples:
        Stream a MiniMax model over Messages::

            model = MiniMaxAnthropicProvider(
                model="MiniMax-M3.1-Flash",
                api_key=os.environ["MINIMAX_API_KEY"],
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "minimax-anthropic"
    default_base_url: ClassVar[str | None] = "https://api.minimax.io/anthropic"

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Give the model its level or its thinking switch, and nothing it would answer with a 400."""
        if request.reasoning_effort == ReasoningEffort.OFF or request.tool_choice:
            return {"thinking": {"type": "disabled"}} if _can_adapt(self.model) else {}
        if _takes_effort(self.model):
            return {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": messages_effort(request.reasoning_effort)},
            }
        return {"thinking": {"type": "adaptive"}} if _can_adapt(self.model) else {}
