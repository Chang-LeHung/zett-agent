"""Adapt tools from Model Context Protocol servers to Zett Agent tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mcp import Client, StdioServerParameters
from mcp_types import CallToolResult, ListToolsResult, Tool

from ..agent import AgentContext
from ..messages import AssistantMessage
from ..tools import AgentTool
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent


@dataclass(frozen=True, slots=True)
class McpHttpServer:
    """One MCP server reached through Streamable HTTP."""

    name: str
    url: str

    def __post_init__(self) -> None:
        _validate_server_name(self.name)
        if not self.url.strip():
            raise ValueError("MCP server URL cannot be empty")


@dataclass(frozen=True, slots=True)
class McpStdioServer:
    """One local MCP server launched as a child process."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | Path | None = None

    def __post_init__(self) -> None:
        _validate_server_name(self.name)
        if not self.command.strip():
            raise ValueError("MCP server command cannot be empty")


type McpServer = McpHttpServer | McpStdioServer


class McpClient(Protocol):
    """Small client surface consumed by McpExtension."""

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult: ...


type McpClientFactory = Callable[[McpServer], AbstractAsyncContextManager[McpClient]]


@dataclass(slots=True)
class _McpRequest:
    """Connections owned by one request and closed at its terminal hook."""

    stack: AsyncExitStack = field(default_factory=AsyncExitStack)


class McpExtension(AgentExtension):
    """Discover and register namespaced tools from configured MCP servers.

    Connections are request-scoped: they open during ``on_tool`` and remain
    alive while the model/tool loop runs. Success, failure, and cancellation
    all close every transport. Remote tool ``search`` from server ``docs`` is
    exposed as ``docs__search`` by default, preventing ambiguous registrations
    when several servers provide the same tool name.

    Example:
        extension = McpExtension(
            [
                McpHttpServer(name="docs", url="http://127.0.0.1:8000/mcp"),
                McpStdioServer(name="local", command="python", args=("server.py",)),
            ]
        )
    """

    priority = 70

    def __init__(
        self,
        servers: Sequence[McpServer],
        *,
        namespace_tools: bool = True,
        client_factory: McpClientFactory | None = None,
    ) -> None:
        names = [server.name for server in servers]
        if len(set(names)) != len(names):
            raise ValueError("MCP server names must be unique")
        self.servers = tuple(servers)
        self.namespace_tools = namespace_tools
        self._client_factory = client_factory or _default_client_factory
        self._requests: dict[AgentContext, _McpRequest] = {}

    async def on_tool(self, context: AgentContext) -> None:
        """Connect all servers and register their complete paginated tool lists."""
        request = _McpRequest()
        await request.stack.__aenter__()
        self._requests[context] = request
        try:
            for server in self.servers:
                client = await request.stack.enter_async_context(self._client_factory(server))
                async for remote_tool in _list_all_tools(client):
                    context.register_tool(self._adapt_tool(server, client, remote_tool))
        except BaseException:
            self._requests.pop(context, None)
            await request.stack.aclose()
            raise

    async def on_success(self, context: AgentContext, result: AssistantMessage) -> None:
        """Close request transports after a successful final answer."""
        await self._close(context)

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Close request transports after a failed request."""
        await self._close(context)

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Close request transports when cancellation bypasses on_error."""
        if isinstance(event, RunCancelledEvent):
            await self._close(context)

    def _adapt_tool(self, server: McpServer, client: McpClient, remote_tool: Tool) -> AgentTool:
        exposed_name = f"{server.name}__{remote_tool.name}" if self.namespace_tools else remote_tool.name

        async def invoke(**arguments: Any) -> Any:
            result = await client.call_tool(remote_tool.name, arguments)
            if result.is_error:
                raise RuntimeError(_mcp_result_text(result) or f"MCP tool {remote_tool.name!r} failed")
            if result.structured_content is not None:
                return result.structured_content
            return [part.model_dump(mode="json", by_alias=True, exclude_none=True) for part in result.content]

        return AgentTool(
            name=exposed_name,
            description=remote_tool.description or f"Call {remote_tool.name} on the {server.name} MCP server.",
            parameters=dict(remote_tool.input_schema),
            handler=invoke,
            guidelines=(f"Use this tool for capabilities provided by the {server.name} MCP server.",),
        )

    async def _close(self, context: AgentContext) -> None:
        request = self._requests.pop(context, None)
        if request is not None:
            await request.stack.aclose()


async def _list_all_tools(client: McpClient) -> AsyncIterator[Tool]:
    cursor: str | None = None
    while True:
        result = await client.list_tools(cursor=cursor)
        for remote_tool in result.tools:
            yield remote_tool
        cursor = result.next_cursor
        if cursor is None:
            return


def _default_client_factory(server: McpServer) -> AbstractAsyncContextManager[McpClient]:
    match server:
        case McpHttpServer(url=url):
            return Client(url)
        case McpStdioServer(command=command, args=args, env=env, cwd=cwd):
            parameters = StdioServerParameters(
                command=command,
                args=list(args),
                env=dict(env) if env is not None else None,
                cwd=cwd,
            )
            return Client(parameters)


def _mcp_result_text(result: CallToolResult) -> str:
    return "\n".join(part.text for part in result.content if getattr(part, "type", None) == "text")


def _validate_server_name(name: str) -> None:
    if not name.strip():
        raise ValueError("MCP server name cannot be empty")
    if "__" in name:
        raise ValueError("MCP server name cannot contain '__'")
