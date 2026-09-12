"""Shared registration logic for provider-hosted tools."""

from __future__ import annotations

from collections.abc import Sequence

from ...agent import AgentContext
from ...model import ServerToolDefinition
from ..base import AgentExtension


class ServerToolExtension(AgentExtension):
    """Register one or more provider-hosted tools for every request.

    Provider-specific subclasses document the wire protocol and supply a useful
    default tool type. The type remains a string because hosted tool names and
    versions evolve independently of zett-agent releases.

    Args:
        server_tools: Ordered hosted-tool types or complete definitions to
            register together. A string becomes a definition with no
            configuration. ``None`` selects the provider subclass defaults;
            an explicit empty sequence disables defaults and is rejected because
            the extension would have no behavior.
        name: Optional extension identity. It defaults to the concrete class
            name because one instance can hold the provider's complete tool set.

    .. warning::
        This extension declares a capability; it does not emulate it locally.
        The selected model endpoint must support the resulting wire schema.

    Examples:
        Define several gateway-specific hosted tools without changing the core model::

            extension = ServerToolExtension(
                [
                    ServerToolDefinition("gateway:search", {"max_uses": 4}),
                    "gateway:fetch",
                ],
                name="gateway-search",
            )
    """

    default_server_tools: tuple[ServerToolDefinition, ...] = ()

    def __init__(
        self,
        server_tools: Sequence[str | ServerToolDefinition] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        supplied = tuple(server_tools) if server_tools is not None else self.default_server_tools
        resolved = tuple(
            ServerToolDefinition(server_tool) if isinstance(server_tool, str) else server_tool
            for server_tool in supplied
        )
        if not resolved:
            raise ValueError("server_tools must contain at least one definition")
        types = [tool.type for tool in resolved]
        if len(types) != len(set(types)):
            raise ValueError("server_tools cannot contain duplicate types")
        self.server_tools = resolved
        self.name = name or type(self).__name__

    async def on_tool(self, context: AgentContext) -> None:
        """Add all hosted tools to the current request's isolated registry."""
        for server_tool in self.server_tools:
            context.register_server_tool(server_tool)
