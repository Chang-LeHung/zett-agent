"""DeepSeek hosted-tool declarations."""

from zett_agent.model import ServerToolDefinition

from .anthropic import AnthropicServerToolExtension
from .base import ServerToolExtension


class DeepSeekServerToolExtension(ServerToolExtension):
    """Declare one DeepSeek Responses API hosted tool.

    DeepSeek currently documents ``web_search`` and the versioned
    ``web_search_2025_08_26`` type. The unversioned spelling is the default;
    its Responses API does not currently publish a separate web-fetch type.

    .. note::
        This schema belongs to DeepSeek's Responses API, not its legacy
        Chat Completions endpoint.

    Examples:
        Enable the stable alias::

            DeepSeekServerToolExtension()

        Pin the documented version::

            DeepSeekServerToolExtension([
                ServerToolDefinition("web_search_2025_08_26"),
            ])
    """

    default_server_tools = (ServerToolDefinition("web_search"),)


class DeepSeekMessagesServerToolExtension(AnthropicServerToolExtension):
    """Declare a web search that DeepSeek's Anthropic Messages endpoint runs on the server.

    The default is ``web_search_20250305``, with ``name`` filled in. The model searches, the
    endpoint returns ``server_tool_use`` and ``web_search_tool_result`` blocks, and the model
    answers from them. Those blocks are sent back unchanged on later turns.

    .. warning::
        This is verified only against a gateway in front of ``deepseek-flash``. DeepSeek's own
        documentation lists the two block kinds as supported but does not document this tool
        type in ``tools``; its search and fetch are separate HTTP endpoints. Test it on your
        endpoint first. A type the endpoint does not know as a server tool, such as the
        ``web_search_20260318`` that :class:`AnthropicServerToolExtension` declares, is treated as
        an ordinary client tool: the model calls it, nothing runs it, and the model reports an
        "Unknown tool" error.

    Examples:
        Search with a cap on the number of searches::

            DeepSeekMessagesServerToolExtension([
                ServerToolDefinition("web_search_20250305", {"max_uses": 3}),
            ])
    """

    default_server_tools = (ServerToolDefinition("web_search_20250305"),)
