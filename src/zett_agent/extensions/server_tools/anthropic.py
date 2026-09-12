"""Anthropic server-tool declarations."""

from __future__ import annotations

from collections.abc import Sequence

from ...model import ServerToolDefinition
from .base import ServerToolExtension


class AnthropicServerToolExtension(ServerToolExtension):
    """Declare one versioned tool executed by Anthropic.

    The defaults are ``web_search_20260318`` and ``web_fetch_20260318`` so the
    model can discover pages and then read selected URLs. Anthropic also
    documents versioned ``code_execution``, ``tool_search``, advisor, and MCP
    connector types. Their version suffixes are deliberately not hidden because
    the suffix is part of Anthropic's compatibility contract.

    ``name`` is required by Anthropic's wire schema. This extension infers the
    common names for web search, web fetch, and code execution; pass it explicitly
    in ``configuration`` for another type.

    Examples:
        Restrict web search to selected domains::

            AnthropicServerToolExtension([
                ServerToolDefinition(
                    "web_search_20260318",
                    {
                        "max_uses": 5,
                        "allowed_domains": ["docs.python.org"],
                    },
                ),
            ])

        Enable server-side code execution::

            AnthropicServerToolExtension([
                ServerToolDefinition("code_execution_20260521"),
            ])
    """

    default_server_tools = (
        ServerToolDefinition("web_search_20260318"),
        ServerToolDefinition("web_fetch_20260318"),
    )

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
        normalized: list[ServerToolDefinition] = []
        for server_tool in resolved:
            configuration = dict(server_tool.configuration)
            inferred_name = next(
                (
                    tool_name
                    for prefix, tool_name in (
                        ("web_search_", "web_search"),
                        ("web_fetch_", "web_fetch"),
                        ("code_execution_", "code_execution"),
                    )
                    if server_tool.type.startswith(prefix)
                ),
                None,
            )
            if inferred_name is not None:
                configuration.setdefault("name", inferred_name)
            normalized.append(ServerToolDefinition(server_tool.type, configuration))
        super().__init__(normalized, name=name)
