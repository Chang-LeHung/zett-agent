"""DeepSeek hosted-tool declarations."""

from ...model import ServerToolDefinition
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
