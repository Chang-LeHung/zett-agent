"""OpenRouter: one endpoint in front of many models.

OpenRouter speaks Chat Completions, with these differences from the plain protocol:

* Reasoning is requested with a ``reasoning`` object (``{"effort": "high"}``) instead of the
  top-level ``reasoning_effort``, and comes back as ``reasoning`` text and a ``reasoning_details``
  array rather than ``reasoning_content``.
* ``reasoning_details`` must be sent back unchanged with the assistant message that produced it,
  or a model that signs or encrypts its reasoning (Claude, the OpenAI reasoning models) loses the
  thread across a tool call. They are kept as the message's ``replay_blocks``.
* An app can identify itself with ``HTTP-Referer`` and ``X-OpenRouter-Title`` headers.
* ``prompt_cache_key`` is used as the routing key that keeps a conversation on one upstream
  provider, so its cache stays warm.

Source: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, cast

from ..json_types import JsonValue
from ..messages import AssistantMessage
from ..model import (
    DEFAULT_RETRY_OPTIONS,
    DEFAULT_TIMEOUT_OPTIONS,
    ModelRequest,
    ReasoningEffort,
    RetryOptions,
    TimeoutOptions,
)
from .openai_compatible import ChatDelta, ChatMessagePayload, OpenAICompatibleProvider
from .wire import JsonObject, read_objects

if TYPE_CHECKING:
    import httpx

#: OpenRouter accepts the same level names as :class:`~zett_agent.model.ReasoningEffort`, spelled
#: the same, except that its ceiling is ``max`` and that turning reasoning off is ``none``.
_EFFORTS: dict[ReasoningEffort, str] = {
    ReasoningEffort.OFF: "none",
    ReasoningEffort.MINIMAL: "minimal",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
    ReasoningEffort.XHIGH: "xhigh",
    ReasoningEffort.MAX: "max",
    ReasoningEffort.ULTRA: "max",
}


class OpenRouterProvider(OpenAICompatibleProvider):
    """Stream any model OpenRouter serves.

    Args:
        model: OpenRouter model slug, such as ``anthropic/claude-sonnet-4.5``.
        api_key: OpenRouter API key.
        transport: Optional HTTPX transport for custom routing or tests.
        base_url: API root; None uses ``https://openrouter.ai/api/v1``.
        temperature: Optional sampling temperature.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        site_url: Your app's URL, sent as ``HTTP-Referer`` so OpenRouter can attribute traffic.
        app_name: Your app's name, sent as ``X-OpenRouter-Title``.
        send_prompt_cache_key: Declare the run's prompt cache key on every request. OpenRouter uses
            it to keep a conversation on one upstream provider.
        extra_headers: HTTP headers sent on every request after the adapter's own.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last. OpenRouter-only fields such as
            ``provider`` (routing preferences) or ``reasoning`` options go here; a ``reasoning``
            given here replaces the one built from the request's effort.

    Note:
        A reasoning level is always sent as ``reasoning.effort``; OpenRouter passes it on in the form
        the upstream model takes and ignores it for a model that cannot reason. Reasoning that cannot
        be shown as text (encrypted or summarized) is kept on the message and sent back unchanged.

    Examples:
        Route one request through OpenRouter::

            model = OpenRouterProvider(
                model="anthropic/claude-sonnet-4.5",
                api_key=os.environ["OPENROUTER_API_KEY"],
                site_url="https://example.com",
                app_name="My agent",
            )
            try:
                client = await create_agent(model)
                print((await client.run("Explain this function")).content)
            finally:
                await model.aclose()
    """

    provider_name = "openrouter"
    reasoning_keys: ClassVar[tuple[str, ...]] = ("reasoning", "reasoning_content")

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
        site_url: str | None = None,
        app_name: str | None = None,
        send_prompt_cache_key: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, str] | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self.site_url = site_url
        self.app_name = app_name
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            transport=transport,
            temperature=temperature,
            retry=retry,
            timeout=timeout,
            send_prompt_cache_key=send_prompt_cache_key,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )

    def vendor_headers(self) -> dict[str, str]:
        """Identify the app to OpenRouter."""
        headers: dict[str, str] = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-OpenRouter-Title"] = self.app_name
        return headers

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Send the reasoning level as a ``reasoning`` object, and the parallel-tools preference."""
        return {
            "reasoning": {"effort": _EFFORTS[request.reasoning_effort]},
            "parallel_tool_calls": request.parallel_tool_call,
        }

    def replay_reasoning(self, message: AssistantMessage, payload: ChatMessagePayload) -> None:
        """Send back ``reasoning_details`` unchanged, or the plain reasoning text if there are none."""
        if not self._same_provider(message):
            return
        if message.replay_blocks:
            payload["reasoning_details"] = cast("list[JsonValue]", [dict(block) for block in message.replay_blocks])
        elif message.reasoning is not None:
            payload["reasoning"] = message.reasoning

    def read_delta(self, delta: Mapping[str, JsonValue]) -> ChatDelta:
        """Read the ``reasoning`` text, and keep every ``reasoning_details`` entry for replay."""
        read = super().read_delta(delta)
        details = tuple(read_objects(delta.get("reasoning_details")))
        return replace(read, replay=details) if details else read

    def merge_replay(self, blocks: list[JsonObject], fragment: JsonObject) -> None:
        """Join the pieces of one reasoning detail into one entry.

        A detail can arrive in several chunks that share a ``type`` and an integer ``index``. Their
        ``text`` and ``summary`` are concatenated; any other field takes the latest value that is
        not null, which is how a signature that closes a text detail is kept. A detail without an
        ``index`` cannot be told apart from its neighbour, so it is never merged.
        """
        previous = blocks[-1] if blocks else None
        index = fragment.get("index")
        if (
            previous is None
            or not isinstance(index, int)
            or previous.get("index") != index
            or previous.get("type") != fragment.get("type")
        ):
            blocks.append(dict(fragment))
            return
        for key, value in fragment.items():
            current = previous.get(key)
            if key in ("text", "summary") and isinstance(value, str) and isinstance(current, str):
                previous[key] = current + value
            elif value is not None:
                previous[key] = value
