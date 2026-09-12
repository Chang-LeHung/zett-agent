"""OpenAI hosted-tool declarations."""

from ...model import ServerToolDefinition
from .base import ServerToolExtension


class OpenAIServerToolExtension(ServerToolExtension):
    """Declare one OpenAI Responses API hosted tool.

    ``web_search`` is the default and lets the hosted search workflow retrieve
    relevant pages; OpenAI does not expose a separate ``web_fetch`` tool type.
    Other Responses API types include
    ``file_search``, ``code_interpreter``, ``image_generation``, and ``mcp``;
    pass their current provider configuration unchanged.

    .. note::
        OpenAI hosts these tools on the Responses API. An OpenAI-compatible
        Chat Completions gateway may use different names or reject them.

    Examples:
        Enable web search::

            OpenAIServerToolExtension([
                ServerToolDefinition(
                    "web_search",
                    {"search_context_size": "medium"},
                ),
            ])

        Search an existing vector store::

            OpenAIServerToolExtension([
                ServerToolDefinition(
                    "file_search",
                    {"vector_store_ids": ["vs_123"]},
                ),
            ])
    """

    default_server_tools = (ServerToolDefinition("web_search"),)
