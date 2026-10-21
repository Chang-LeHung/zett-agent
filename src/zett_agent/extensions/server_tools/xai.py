"""xAI hosted-tool declarations."""

from zett_agent.model import ServerToolDefinition

from .base import ServerToolExtension


class XAIServerToolExtension(ServerToolExtension):
    """Declare xAI Responses API hosted tools.

    ``web_search`` searches the web and browses pages in one tool, so xAI has no separate fetch
    type. ``x_search`` searches posts on X. Both are the defaults. Other Responses API types are
    ``code_execution``, ``collections_search``, and ``mcp``; pass their current configuration
    unchanged.

    .. note::
        xAI serves these tools on its Responses API, so build the provider with
        ``XAIProvider(response=True)``. They are billed per call in addition to tokens.

    Examples:
        Enable both searches::

            XAIServerToolExtension()

        Search only some sites and some accounts::

            XAIServerToolExtension([
                ServerToolDefinition("web_search", {"allowed_domains": ["example.com"]}),
                ServerToolDefinition("x_search", {"allowed_x_handles": ["xai"]}),
            ])
    """

    default_server_tools = (ServerToolDefinition("web_search"), ServerToolDefinition("x_search"))
