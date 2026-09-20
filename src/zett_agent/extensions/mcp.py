"""Adapt tools from Model Context Protocol servers to Zett Agent tools."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp_types import CallToolResult, ListToolsResult, Tool

from ..agent import AgentRunContext
from ..messages import AssistantMessage, SystemMessage
from ..tools import AgentTool
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent

DEFAULT_MCP_CONFIG_PATH = Path("~/.zett/mcp.json")
DEFAULT_MCP_SERVER_KEYS = ("servers", "mcpServers")
_STREAMABLE_HTTP_TYPES = frozenset({"streamable", "streamable-http", "streamable_http", "http"})
_STDIO_TYPES = frozenset({"stdio"})


@dataclass(frozen=True, slots=True)
class McpHttpServer:
    """Configure one MCP server reached through Streamable HTTP.

    Attributes:
        name: Unique server namespace used when registering its tools.
        url: Non-empty Streamable HTTP endpoint; credentials/routing are the
            application's responsibility.
        headers: HTTP headers sent during handshake, tool calls, streaming,
            and session termination. The mapping is excluded from ``repr`` so
            authorization values are not accidentally written to logs.
    """

    name: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _validate_server_name(self.name)
        if not self.url.strip():
            raise ValueError("MCP server URL cannot be empty")
        for key, value in self.headers.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("MCP HTTP header names must be non-empty strings")
            if not isinstance(value, str):
                raise ValueError(f"MCP HTTP header {key!r} must have a string value")


@dataclass(frozen=True, slots=True)
class McpStdioServer:
    """Configure one local MCP server launched as a child process.

    Attributes:
        name: Unique server namespace used when registering its tools.
        command: Executable used to start the server.
        args: Command arguments passed without shell-string interpolation.
        env: Optional environment mapping supplied to the MCP process adapter.
        cwd: Optional working directory for the server process.

    Warning:
        Only launch trusted servers with appropriate process permissions.
    """

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


@dataclass(frozen=True, slots=True)
class McpConfiguration:
    """Validated MCP servers loaded from a JSON configuration document.

    MCP clients use both ``servers`` and ``mcpServers`` as their root key.
    Keeping that variation here prevents transport registration code from
    depending on a particular client's file convention.
    """

    servers: tuple[McpServer, ...] = ()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        server_keys: Sequence[str] = DEFAULT_MCP_SERVER_KEYS,
    ) -> McpConfiguration:
        """Read and validate one UTF-8 JSON configuration file.

        Args:
            path: File containing the MCP server mapping.
            server_keys: Accepted root keys in precedence-independent order.
                Exactly one of these keys may occur in the document.
        """
        resolved_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid MCP configuration JSON in {resolved_path}: {error}") from error
        return cls.from_mapping(payload, server_keys=server_keys)

    @classmethod
    def from_mapping(
        cls,
        payload: object,
        *,
        server_keys: Sequence[str] = DEFAULT_MCP_SERVER_KEYS,
    ) -> McpConfiguration:
        """Parse a server mapping under one of the configured root keys."""
        if not isinstance(payload, dict):
            raise ValueError("MCP configuration root must be a JSON object")
        accepted_keys = _validate_server_keys(server_keys)
        keys = [key for key in accepted_keys if key in payload]
        if len(keys) > 1:
            rendered = ", ".join(repr(key) for key in accepted_keys)
            raise ValueError(f"MCP configuration must use exactly one of {rendered}")
        if not keys:
            rendered = ", ".join(repr(key) for key in accepted_keys)
            raise ValueError(f"MCP configuration must contain one of {rendered}")
        root_key = keys[0]
        entries = payload[root_key]
        if not isinstance(entries, dict):
            raise ValueError(f"MCP configuration '{root_key}' must be a JSON object")

        servers: list[McpServer] = []
        for name, entry in entries.items():
            servers.append(cls._parse_server(name, entry))
        return cls(tuple(servers))

    @staticmethod
    def _parse_server(name: object, entry: object) -> McpServer:
        """Validate one named HTTP or stdio server entry."""
        if not isinstance(name, str):
            raise ValueError("MCP server names must be strings")
        if not isinstance(entry, dict):
            raise ValueError(f"MCP server {name!r} must be a JSON object")
        transport = entry.get("type", entry.get("transport"))
        # Most editor-generated stdio configurations omit ``type`` because a
        # command unambiguously identifies the transport.
        if transport is None and "command" in entry:
            transport = "stdio"
        if not isinstance(transport, str):
            raise ValueError(f"MCP server {name!r} must define a string transport type")
        match transport.lower():
            case value if value in _STREAMABLE_HTTP_TYPES:
                return McpConfiguration._parse_http_server(name, entry)
            case value if value in _STDIO_TYPES:
                return McpConfiguration._parse_stdio_server(name, entry)
            case _:
                raise ValueError(f"MCP server {name!r} uses unsupported transport {transport!r}")

    @staticmethod
    def _parse_http_server(name: str, entry: dict[str, object]) -> McpHttpServer:
        """Validate one named Streamable HTTP server entry."""
        url = entry.get("url")
        if not isinstance(url, str):
            raise ValueError(f"MCP server {name!r} must define a string URL")
        headers = entry.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError(f"MCP server {name!r} headers must be a JSON object")
        return McpHttpServer(name=name, url=url, headers=headers)

    @staticmethod
    def _parse_stdio_server(name: str, entry: dict[str, object]) -> McpStdioServer:
        """Validate one named child-process server without invoking a shell."""
        command = entry.get("command")
        if not isinstance(command, str):
            raise ValueError(f"MCP stdio server {name!r} must define a string command")
        raw_args = entry.get("args", [])
        if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
            raise ValueError(f"MCP stdio server {name!r} args must be an array of strings")
        raw_env = entry.get("env")
        if raw_env is not None and (
            not isinstance(raw_env, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in raw_env.items())
        ):
            raise ValueError(f"MCP stdio server {name!r} env must be an object with string values")
        cwd = entry.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError(f"MCP stdio server {name!r} cwd must be a string")
        return McpStdioServer(
            name=name,
            command=command,
            args=tuple(raw_args),
            env=raw_env,
            cwd=cwd,
        )


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

    Examples:
        Usage::

            # With no arguments, load Streamable HTTP servers from
            # ~/.zett/mcp.json when that file exists. A configured path that
            # does not exist yet is treated as no servers and is still named in
            # the request's system message.
            extension = McpExtension()

            # A custom path and explicit server definitions may be used
            # independently or merged.
            extension = McpExtension(
                servers=[
                    McpHttpServer(name="docs", url="http://127.0.0.1:8000/mcp"),
                    McpStdioServer(name="local", command="python", args=("server.py",)),
                ],
                config_path="./mcp.json",
                server_keys=("servers", "mcpServers"),
            )

        ``mcp.json`` accepts either ``servers`` or the ecosystem-compatible
        ``mcpServers`` key. Streamable HTTP and local stdio transports may be
        declared together::

            {
              "servers": {
                "docs": {
                  "type": "streamable-http",
                  "url": "https://docs.example.com/mcp",
                  "headers": {
                    "Authorization": "Bearer token",
                    "X-Tenant-ID": "tenant-1"
                  }
                },
                "local": {
                  "command": "uvx",
                  "args": ["some-mcp-server"],
                  "env": {"TOKEN": "secret"},
                  "cwd": "/path/to/project"
                }
              }
            }
    """

    priority = 70

    def __init__(
        self,
        servers: Sequence[McpServer] | None = None,
        *,
        config_path: str | Path | None = None,
        server_keys: Sequence[str] = DEFAULT_MCP_SERVER_KEYS,
        namespace_tools: bool = True,
        client_factory: McpClientFactory | None = None,
    ) -> None:
        self.server_keys = _validate_server_keys(server_keys)
        configured_servers: tuple[McpServer, ...] = ()
        resolved_config_path: Path | None = None
        if config_path is not None:
            resolved_config_path = Path(config_path).expanduser().resolve()
            # A configured location is reported in the system prompt, so a file
            # that does not exist yet simply means no servers are loaded.
            if resolved_config_path.is_file():
                configured_servers = McpConfiguration.from_file(
                    resolved_config_path,
                    server_keys=self.server_keys,
                ).servers
        elif servers is None:
            resolved_config_path = DEFAULT_MCP_CONFIG_PATH.expanduser().resolve()
            if resolved_config_path.is_file():
                configured_servers = McpConfiguration.from_file(
                    resolved_config_path,
                    server_keys=self.server_keys,
                ).servers

        resolved_servers = (*configured_servers, *(servers or ()))
        names = [server.name for server in resolved_servers]
        if len(set(names)) != len(names):
            raise ValueError("MCP server names must be unique")
        self.servers = resolved_servers
        self.config_path = resolved_config_path
        self.namespace_tools = namespace_tools
        self._client_factory = client_factory or _default_client_factory
        self._requests: dict[AgentRunContext, _McpRequest] = {}

    async def on_tool(self, context: AgentRunContext) -> None:
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

    async def on_state(self, context: AgentRunContext) -> None:
        """Describe where MCP servers are configured and how their tools are named."""
        if self.config_path is None and not self.servers:
            return
        message = SystemMessage(content=self._instructions())
        instructions = [item for item in context.state.messages if isinstance(item, SystemMessage)]
        context.add_message(message, index=len(instructions))

    def _instructions(self) -> str:
        """Describe the MCP configuration file and the namespaces it declares."""
        keys = " or ".join(f"`{key}`" for key in self.server_keys)
        lines = ["# MCP servers"]
        if self.config_path is not None:
            lines.append(
                f"MCP servers are configured in `{self.config_path}`: a JSON file whose {keys} object maps a "
                "server name to its transport. Servers changed there apply to the next request."
            )
        else:
            lines.append("MCP servers are supplied by the application; this request loads no configuration file.")
        if not self.servers:
            lines.append("No MCP servers are configured, so no MCP tools are available.")
            return "\n".join(lines)
        if self.namespace_tools:
            lines.append("Remote tools are exposed as `<server>__<tool>`, so each name names its server.")
        entries = "\n".join(f"- {server.name}: {_transport_label(server)}" for server in self.servers)
        return f"{'\n'.join(lines)}\n\nConfigured servers:\n{entries}"

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Close request transports after a successful final answer."""
        await self._close(context)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Close request transports after a failed request."""
        await self._close(context)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
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

    async def _close(self, context: AgentRunContext) -> None:
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


def _transport_label(server: McpServer) -> str:
    """Name one configured transport without exposing credentials or arguments."""
    match server:
        case McpHttpServer():
            return "streamable HTTP"
        case McpStdioServer():
            return "stdio child process"
    raise ValueError(f"Unsupported MCP server: {type(server).__name__}")


def _default_client_factory(server: McpServer) -> AbstractAsyncContextManager[McpClient]:
    match server:
        case McpHttpServer(headers=headers) if headers:
            return _authenticated_http_client(server)
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


@asynccontextmanager
async def _authenticated_http_client(server: McpHttpServer) -> AsyncGenerator[McpClient, None]:
    """Connect Streamable HTTP with SDK-owned timeouts and caller headers."""
    async with create_mcp_http_client(headers=dict(server.headers)) as http_client:
        transport = streamable_http_client(server.url, http_client=http_client)
        async with Client(transport) as client:
            yield client


def _mcp_result_text(result: CallToolResult) -> str:
    return "\n".join(part.text for part in result.content if getattr(part, "type", None) == "text")


def _validate_server_name(name: str) -> None:
    if not name.strip():
        raise ValueError("MCP server name cannot be empty")
    if "__" in name:
        raise ValueError("MCP server name cannot contain '__'")


def _validate_server_keys(server_keys: Sequence[str]) -> tuple[str, ...]:
    """Normalize configurable JSON root keys and reject ambiguous input."""
    keys = tuple(server_keys)
    if not keys:
        raise ValueError("MCP server keys cannot be empty")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("MCP server keys must be non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("MCP server keys must be unique")
    return keys
