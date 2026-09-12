"""Google Gemini built-in-tool declarations."""

from ...model import ServerToolDefinition
from .base import ServerToolExtension


class GoogleServerToolExtension(ServerToolExtension):
    """Declare one server-executed Gemini tool.

    ``google_search`` and ``url_context`` are enabled by default, providing web
    discovery and URL retrieval respectively. Gemini also exposes
    ``code_execution``, ``file_search``, and ``google_maps`` on compatible
    models. Configuration keys map directly to ``google.genai.types.Tool``.

    Examples:
        Ground answers with Google Search::

            GoogleServerToolExtension()

        Search uploaded corpora::

            GoogleServerToolExtension([
                ServerToolDefinition(
                    "file_search",
                    {"file_search_store_names": ["fileSearchStores/manuals"]},
                ),
            ])

        Combine URL reading and code execution by registering two extensions::

            extensions = [
                GoogleServerToolExtension([
                    "url_context",
                    "code_execution",
                ]),
            ]
    """

    default_server_tools = (
        ServerToolDefinition("google_search"),
        ServerToolDefinition("url_context"),
    )
