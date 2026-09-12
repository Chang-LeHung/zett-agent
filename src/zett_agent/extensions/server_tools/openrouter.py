"""OpenRouter Chat Completions server-tool declarations."""

from ...model import ServerToolDefinition
from .base import ServerToolExtension


class OpenRouterServerToolExtension(ServerToolExtension):
    """Declare one tool hosted by an OpenRouter-compatible endpoint.

    ``openrouter:web_search`` and ``openrouter:web_fetch`` are enabled by
    default, allowing discovery followed by full URL or PDF retrieval. Both are
    supported through OpenRouter's Chat Completions endpoint.

    Examples:
        Allow up to four searches from selected domains::

            OpenRouterServerToolExtension([
                ServerToolDefinition(
                    "openrouter:web_search",
                    {
                        "parameters": {
                            "max_uses": 4,
                            "allowed_domains": ["python.org"],
                        }
                    },
                ),
            ])
    """

    default_server_tools = (
        ServerToolDefinition("openrouter:web_search"),
        ServerToolDefinition("openrouter:web_fetch"),
    )
