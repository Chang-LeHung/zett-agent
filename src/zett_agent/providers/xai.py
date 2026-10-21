"""xAI, the maker of Grok.

:class:`XAIProvider` speaks Chat Completions, or Responses with ``response=True``. xAI describes Chat
Completions as the legacy endpoint: new features, including its hosted search tools, arrive on
Responses first. The service differs from the plain protocol in these ways:

* ``grok-4.5`` and later take a ``reasoning_effort`` of ``low``, ``medium``, ``high`` or ``xhigh``,
  on Chat Completions as a top-level field and on Responses as ``reasoning.effort``. Reasoning
  cannot be switched off, and earlier models reject the field.
* On Responses, ``grok-4.7`` returns its encrypted reasoning on every response, and expects the
  reasoning items back unchanged.
* ``presence_penalty``, ``frequency_penalty`` and ``stop`` are errors on reasoning models.

Sources: https://docs.x.ai/developers/model-capabilities/text/reasoning and
https://docs.x.ai/developers/tools/web-search
"""

from __future__ import annotations

import re
from typing import ClassVar

from ..model import ModelRequest, ReasoningEffort
from .openai_compatible import OpenAICompatibleProvider
from .wire import JsonObject

_VERSION = re.compile(r"grok-(\d+)(?:\.(\d+))?")


def _grok_version(model: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` of a Grok model name such as ``grok-4.7`` or ``grok-4``, else None."""
    found = _VERSION.match(model)
    return (int(found.group(1)), int(found.group(2) or 0)) if found else None


def _grok_effort(effort: ReasoningEffort) -> str:
    """Map a provider-neutral level onto the four Grok knows; reasoning cannot be turned off."""
    match effort:
        case ReasoningEffort.OFF | ReasoningEffort.MINIMAL | ReasoningEffort.LOW:
            return "low"
        case ReasoningEffort.MEDIUM:
            return "medium"
        case ReasoningEffort.HIGH:
            return "high"
        case ReasoningEffort.XHIGH | ReasoningEffort.MAX | ReasoningEffort.ULTRA:
            return "xhigh"


class XAIProvider(OpenAICompatibleProvider):
    """Stream Grok through Chat Completions, or through Responses with ``response=True``.

    Args:
        model: Grok model identifier, such as ``grok-4.7``.
        api_key: xAI API key.
        base_url: API root; None uses ``https://api.x.ai/v1``.
        transport: Optional HTTPX transport for custom routing or tests.
        temperature: Optional sampling temperature.
        response: Use the Responses API instead of Chat Completions. Hosted tools such as
            :class:`~zett_agent.extensions.server_tools.xai.XAIServerToolExtension` need it.
        retry: Model-owned retry/backoff policy; never restarts an emitted stream.
        timeout: How long the provider may stay silent before the request is given up on. See
            :class:`~zett_agent.model.TimeoutOptions`.
        send_prompt_cache_key: Declare the run's prompt cache key on every request.
        extra_headers: HTTP headers sent on every request after the adapter's own. xAI documents an
            ``x-grok-conv-id`` header that keeps a conversation on one server for better cache hits.
        extra_query: URL query parameters sent on every request.
        extra_body: JSON fields merged into every request body last.

    Note:
        The level is sent only to ``grok-4.5`` and later, as ``low``, ``medium``, ``high`` or
        ``xhigh``. ``OFF`` is sent as ``low``, because xAI cannot switch reasoning off and leaving
        the field out means ``high``. A model before ``grok-4.5`` gets no field at all.

    Examples:
        Stream Grok with live web search::

            model = XAIProvider(model="grok-4.7", api_key=os.environ["XAI_API_KEY"], response=True)
            try:
                client = await create_agent(model, extensions=[XAIServerToolExtension()])
                print((await client.run("What happened in the news today?")).content)
            finally:
                await model.aclose()
    """

    provider_name = "xai"
    default_base_url: ClassVar[str | None] = "https://api.x.ai/v1"

    def _takes_effort(self) -> bool:
        version = _grok_version(self.model)
        return version is not None and version >= (4, 5)

    def request_fields(self, request: ModelRequest) -> JsonObject:
        """Give ``grok-4.5`` and later their level as a top-level ``reasoning_effort``."""
        if not self._takes_effort():
            return {}
        return {"reasoning_effort": _grok_effort(request.reasoning_effort)}

    def responses_fields(self, request: ModelRequest) -> JsonObject:
        """Spell the level as ``reasoning.effort`` for ``grok-4.5`` and later, and send none to older models.

        The object replaces the standard one, which also asks for a reasoning ``summary`` that xAI does
        not document. An older model gets ``reasoning: null`` so it does not see the standard one.
        """
        if not self._takes_effort():
            return {"reasoning": None}
        return {"reasoning": {"effort": _grok_effort(request.reasoning_effort)}}
