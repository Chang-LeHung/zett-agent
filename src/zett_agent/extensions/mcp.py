"""Adapt tools from Model Context Protocol servers to Zett Agent tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .._compat import TypeAliasType
from ..agent import AgentRunContext
from ..messages import AssistantMessage, SystemMessage
from ..tools.base import AgentTool
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent

if TYPE_CHECKING:
    from mcp_types import CallToolResult, ListToolsResult, Tool

DEFAULT_MCP_CONFIG_PATH = Path("~/.zett/mcp.json")
DEFAULT_MCP_SERVER_KEYS = ("servers", "mcpServers")
_STREAMABLE_HTTP_TYPES = frozenset({"streamable", "streamable-http", "streamable_http", "http"})
_STDIO_TYPES = frozenset({"stdio"})
_MAX_TOOL_PAGES = 100
_MAX_TOOL_NAME_LENGTH = 64
_CLOSE_TIMEOUT = 5.0
_HTTP_TIMEOUT = 30.0
_HTTP_SSE_READ_TIMEOUT = 300.0
_UNSAFE_TOOL_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]")
_UNREAD = object()

logger = logging.getLogger(__name__)


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


McpServer = TypeAliasType("McpServer", McpHttpServer | McpStdioServer)


@dataclass(frozen=True, slots=True)
class McpConfiguration:
    """Validated MCP servers loaded from a JSON configuration document.

    MCP clients use both ``servers`` and ``mcpServers`` as their root key.
    Keeping that variation here prevents transport registration code from
    depending on a particular client's file convention.
    """

    servers: tuple[McpServer, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict)
    """Entries dropped by a non-strict load, mapping server name to the reason."""

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        server_keys: Sequence[str] = DEFAULT_MCP_SERVER_KEYS,
        strict: bool = True,
    ) -> McpConfiguration:
        """Read and validate one UTF-8 JSON configuration file.

        Args:
            path: File containing the MCP server mapping.
            server_keys: Accepted root keys in precedence-independent order.
                Exactly one of these keys may occur in the document.
            strict: When true (the default) one invalid server entry fails the
                whole file. When false invalid entries are skipped, logged, and
                listed in :attr:`skipped` so the valid servers stay usable.
                A document that is not valid JSON or lacks a server mapping
                still raises: there is nothing to salvage.
        """
        resolved_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid MCP configuration JSON in {resolved_path}: {error}") from error
        return cls.from_mapping(payload, server_keys=server_keys, strict=strict)

    @classmethod
    def from_mapping(
        cls,
        payload: object,
        *,
        server_keys: Sequence[str] = DEFAULT_MCP_SERVER_KEYS,
        strict: bool = True,
    ) -> McpConfiguration:
        """Parse a server mapping under one of the configured root keys.

        ``strict`` has the meaning described in :meth:`from_file`.
        """
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
        skipped: dict[str, str] = {}
        for name, entry in entries.items():
            try:
                servers.append(cls._parse_server(name, entry))
            except ValueError as error:
                if strict:
                    raise
                logger.warning("Skipping invalid MCP server %r: %s", name, error)
                skipped[str(name)] = str(error)
        return cls(tuple(servers), skipped)

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


McpClientFactory = TypeAliasType("McpClientFactory", Callable[[McpServer], AbstractAsyncContextManager[McpClient]])


class _Connection:
    """Lazily open one server and keep it alive in a dedicated task.

    The MCP SDK enters transports through anyio cancel scopes, which must be
    exited by the task that entered them. A task that owns the whole
    ``async with`` block therefore lets many servers connect concurrently, lets
    calls come from any task, and lets the connection be released from a
    synchronous hook.
    """

    def __init__(self, server: McpServer, factory: McpClientFactory, connect_timeout: float | None) -> None:
        self.server = server
        self._factory = factory
        self._connect_timeout = connect_timeout
        self._lock = asyncio.Lock()
        self._client: McpClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()
        self._closed = False

    async def client(self) -> McpClient:
        """Return the connected client, opening the transport on first use."""
        async with self._lock:
            if self._client is not None:
                return self._client
            if self._closed:
                raise RuntimeError(f"MCP server {self.server.name!r} connection is already closed")
            ready: asyncio.Future[McpClient] = asyncio.get_running_loop().create_future()
            self._closing = asyncio.Event()
            self._task = asyncio.create_task(self._hold(ready, self._closing))
            try:
                self._client = await asyncio.wait_for(ready, self._connect_timeout)
            except BaseException:
                await self._stop(cancel=True)
                raise
            return self._client

    async def _hold(self, ready: asyncio.Future[McpClient], closing: asyncio.Event) -> None:
        try:
            async with self._factory(self.server) as client:
                if ready.done():  # the caller timed out or was cancelled while connecting
                    return
                ready.set_result(client)
                await closing.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except Exception as error:
            # Before the handshake the caller receives the error; afterwards the
            # owning task only has to unwind, so the failure is logged on close.
            if ready.done():
                logger.warning("MCP server %r failed while closing: %r", self.server.name, error)
            else:
                ready.set_exception(error)

    def release(self) -> None:
        """Ask the owning task to unwind; safe to call from synchronous code."""
        self._closed = True
        self._closing.set()

    async def close(self) -> None:
        """Close the transport and wait for the owning task; never raises for transport errors."""
        self._closed = True
        await self._stop()

    async def _stop(self, *, cancel: bool = False) -> None:
        task, self._task = self._task, None
        self._closing.set()
        if task is None:
            return
        if cancel:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=_CLOSE_TIMEOUT)
        if not done:
            task.cancel()
            await asyncio.wait({task}, timeout=_CLOSE_TIMEOUT)
        self._client = None


@dataclass(slots=True)
class _CachedTools:
    """Tool definitions remembered between requests for one server configuration."""

    server: McpServer
    tools: tuple[Tool, ...]
    expires_at: float


@dataclass(slots=True)
class _McpRequest:
    """Servers, lazy connections, and discovery failures owned by one request."""

    servers: tuple[McpServer, ...]
    connections: dict[str, _Connection] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    def release(self) -> None:
        for connection in self.connections.values():
            connection.release()

    async def close(self) -> None:
        # _Connection.close() reports transport errors itself; one stuck server
        # must not keep the others open.
        await asyncio.gather(*(item.close() for item in self.connections.values()), return_exceptions=True)


class McpExtension(AgentExtension):
    """Discover and register namespaced tools from configured MCP servers.

    Connections are request-scoped: they open during ``on_tool`` and remain
    alive while the model/tool loop runs. Success, failure, and cancellation
    all close every transport. Remote tool ``search`` from server ``docs`` is
    exposed as ``docs__search`` by default, preventing ambiguous registrations
    when several servers provide the same tool name.

    Args:
        servers: Explicit server definitions, merged after the file's servers.
        config_path: JSON file to load; re-read whenever it changes.
        server_keys: Accepted root keys of the JSON file.
        namespace_tools: Expose tools as ``<server>__<tool>``.
        client_factory: Replaces the SDK client, mainly for tests.
        strict_config: When true (default) one invalid server entry makes the
            whole configuration invalid; when false it is skipped and logged
            so the remaining servers still work.
        connect_timeout: Seconds allowed to connect and list one server.
        call_timeout: Seconds allowed for one tool call.
        tools_cache_ttl: Seconds a server's tool list is reused; 0 disables.

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
        strict_config: bool = True,
        connect_timeout: float | None = 30.0,
        call_timeout: float | None = 300.0,
        tools_cache_ttl: float = 300.0,
    ) -> None:
        self.server_keys = _validate_server_keys(server_keys)
        self.namespace_tools = namespace_tools
        self.strict_config = strict_config
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.tools_cache_ttl = tools_cache_ttl
        self._client_factory = client_factory or _default_client_factory
        self._requests: dict[AgentRunContext, _McpRequest] = {}
        self._tool_cache: dict[str, _CachedTools] = {}
        self._explicit_servers: tuple[McpServer, ...] = tuple(servers or ())
        self._file_servers: tuple[McpServer, ...] = ()
        self._file_stamp: tuple[int, int] | None | object = _UNREAD
        self.config_path: Path | None = None
        if config_path is not None:
            # A configured location is reported in the system prompt, so a file
            # that does not exist yet simply means no servers are loaded.
            self.config_path = Path(config_path).expanduser().resolve()
        elif servers is None:
            self.config_path = DEFAULT_MCP_CONFIG_PATH.expanduser().resolve()
        # Construction always reports a broken file to the developer.
        self._refresh_file_servers(raise_errors=True)
        _require_unique(self.servers)

    @property
    def servers(self) -> tuple[McpServer, ...]:
        """Servers declared in the configuration file followed by explicit ones."""
        return (*self._file_servers, *self._explicit_servers)

    def _refresh_file_servers(self, *, raise_errors: bool) -> None:
        """Re-read the configuration file when it changed since the last read.

        Requests are lenient: a file that is half-written or invalid keeps the
        last good server list instead of failing every request. ``strict_config``
        decides whether one bad *entry* invalidates the whole file.
        """
        path = self.config_path
        if path is None:
            return
        stamp: tuple[int, int] | None = None
        try:
            if path.is_file():
                status = path.stat()
                stamp = (status.st_mtime_ns, status.st_size)
        except OSError:
            stamp = None
        if stamp == self._file_stamp:
            return
        if stamp is None:
            self._file_servers = ()
            self._file_stamp = stamp
            return
        try:
            loaded = McpConfiguration.from_file(path, server_keys=self.server_keys, strict=self.strict_config).servers
            _require_unique((*loaded, *self._explicit_servers))
        except (OSError, ValueError) as error:
            if raise_errors:
                raise
            logger.warning("Ignoring invalid MCP configuration %s: %s", path, error)
            self._file_stamp = stamp
            return
        self._file_servers = loaded
        self._file_stamp = stamp

    async def on_tool(self, context: AgentRunContext) -> None:
        """Discover every server's tools concurrently; connections open lazily.

        Tool lists are cached per server configuration, so a warm cache starts
        no process and opens no socket until the model actually calls a tool.
        A server that fails or times out is reported to the model and skipped
        instead of failing the request.
        """
        self._refresh_file_servers(raise_errors=False)
        previous = self._requests.pop(context, None)
        if previous is not None:
            await previous.close()
        request = _McpRequest(self.servers)
        self._requests[context] = request
        try:
            discovered = await asyncio.gather(*(self._discover(request, server) for server in request.servers))
            for server, tools in zip(request.servers, discovered, strict=True):
                for remote_tool in tools or ():
                    tool = self._adapt_tool(server, request.connections[server.name], remote_tool)
                    if tool.name in context.tools:
                        logger.warning("Skipping MCP tool %r from %r: name already registered", tool.name, server.name)
                        continue
                    context.register_tool(tool)
        except BaseException:
            self._requests.pop(context, None)
            await request.close()
            raise

    async def _discover(self, request: _McpRequest, server: McpServer) -> tuple[Tool, ...] | None:
        """List one server's tools from the cache or the network; None means unavailable."""
        connection = _Connection(server, self._client_factory, self.connect_timeout)
        request.connections[server.name] = connection
        cached = self._tool_cache.get(server.name)
        if cached is not None and cached.server == server and cached.expires_at > time.monotonic():
            return cached.tools
        try:
            client = await connection.client()
            tools = await asyncio.wait_for(_collect_tools(client), self.connect_timeout)
        except Exception as error:
            logger.warning("MCP server %r is unavailable: %r", server.name, error)
            request.unavailable[server.name] = type(error).__name__
            await connection.close()
            return None
        if self.tools_cache_ttl > 0:
            self._tool_cache[server.name] = _CachedTools(server, tools, time.monotonic() + self.tools_cache_ttl)
        return tools

    async def on_state(self, context: AgentRunContext) -> None:
        """Describe where MCP servers are configured and how their tools are named."""
        request = self._requests.get(context)
        servers = request.servers if request is not None else self.servers
        if self.config_path is None and not servers:
            return
        unavailable = request.unavailable if request is not None else {}
        message = SystemMessage(content=self._instructions(servers, unavailable))
        instructions = [item for item in context.state.messages if isinstance(item, SystemMessage)]
        context.add_message(message, index=len(instructions))

    def _instructions(self, servers: Sequence[McpServer], unavailable: Mapping[str, str]) -> str:
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
        if not servers:
            lines.append("No MCP servers are configured, so no MCP tools are available.")
            return "\n".join(lines)
        if self.namespace_tools:
            lines.append("Remote tools are exposed as `<server>__<tool>`, so each name names its server.")
        entries = "\n".join(
            f"- {server.name}: {_transport_label(server)}"
            + (f" (unavailable in this request: {unavailable[server.name]})" if server.name in unavailable else "")
            for server in servers
        )
        configured = "\n".join(lines)
        return f"{configured}\n\nConfigured servers:\n{entries}"

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Close request transports after a successful final answer."""
        await self._close(context)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Close request transports after a failed request."""
        await self._close(context)

    async def on_compact(self, context: AgentRunContext) -> None:
        """Close transports of a maintenance pass, which never reaches on_success."""
        await self._close(context)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Close request transports when cancellation bypasses on_error."""
        if isinstance(event, RunCancelledEvent):
            await self._close(context)

    def _request_finished(self, context: AgentRunContext) -> None:
        """Safety net: signal any connection a terminal path forgot to close."""
        super()._request_finished(context)
        request = self._requests.pop(context, None)
        if request is not None:
            request.release()

    def _adapt_tool(self, server: McpServer, connection: _Connection, remote_tool: Tool) -> AgentTool:
        raw_name = f"{server.name}__{remote_tool.name}" if self.namespace_tools else remote_tool.name

        async def invoke(**arguments: Any) -> Any:
            try:
                client = await connection.client()
                result = await asyncio.wait_for(client.call_tool(remote_tool.name, arguments), self.call_timeout)
            except asyncio.TimeoutError as error:
                raise TimeoutError(f"MCP tool {remote_tool.name!r} on server {server.name!r} timed out") from error
            if result.is_error:
                raise RuntimeError(_mcp_result_text(result) or f"MCP tool {remote_tool.name!r} failed")
            if result.structured_content is not None:
                return result.structured_content
            return [part.model_dump(mode="json", by_alias=True, exclude_none=True) for part in result.content]

        return AgentTool(
            name=_provider_safe_name(raw_name),
            description=remote_tool.description or f"Call {remote_tool.name} on the {server.name} MCP server.",
            parameters=_compact_schema(remote_tool.input_schema),
            handler=invoke,
            # The system message already says every `<server>__<tool>` name
            # belongs to its server; repeating that sentence per tool only
            # spends prompt tokens.
            guidelines=(),
        )

    async def _close(self, context: AgentRunContext) -> None:
        request = self._requests.pop(context, None)
        if request is not None:
            await request.close()


async def _collect_tools(client: McpClient) -> tuple[Tool, ...]:
    """Follow ``next_cursor`` to the end, refusing cursor loops and runaway paging."""
    tools: list[Tool] = []
    seen: set[str] = set()
    cursor: str | None = None
    for _ in range(_MAX_TOOL_PAGES):
        result = await client.list_tools(cursor=cursor)
        tools.extend(result.tools)
        cursor = result.next_cursor
        if cursor is None:
            return tuple(tools)
        if cursor in seen:
            raise RuntimeError("MCP server returned a repeating tools/list cursor")
        seen.add(cursor)
    raise RuntimeError(f"MCP server returned more than {_MAX_TOOL_PAGES} tools/list pages")


# JSON Schema keywords whose value is one sub-schema, a list of them, or a map of them.
_SCHEMA_VALUE = frozenset({"items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"})
_SCHEMA_LIST = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_SCHEMA_MAP = frozenset({"properties", "patternProperties", "$defs", "definitions"})
# Annotations that cost prompt tokens without helping a model fill arguments:
# `title` merely repeats the property name in prose.
_NOISE_KEYWORDS = frozenset({"$schema", "title"})


def _compact_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Drop schema keywords that only repeat what the model already sees.

    MCP servers generated from typed code (pydantic, zod) attach a ``title`` to
    every property and a ``$schema`` URL to every tool. Across dozens of tools
    that is a large block of identical-looking text with no validation value.
    Property *names* are never touched, only the annotation keywords.
    """

    def clean(node: Any) -> Any:
        if not isinstance(node, Mapping):
            return node
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in _NOISE_KEYWORDS:
                continue
            if key in _SCHEMA_MAP and isinstance(value, Mapping):
                result[key] = {name: clean(child) for name, child in value.items()}
            elif key in _SCHEMA_LIST and isinstance(value, list):
                result[key] = [clean(child) for child in value]
            elif key in _SCHEMA_VALUE:
                result[key] = clean(value)
            else:
                result[key] = value
        return result

    return clean(schema)


def _provider_safe_name(name: str) -> str:
    """Fit a tool name into the character set and length model providers accept."""
    cleaned = _UNSAFE_TOOL_CHARACTERS.sub("_", name)
    if len(cleaned) <= _MAX_TOOL_NAME_LENGTH:
        return cleaned
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: _MAX_TOOL_NAME_LENGTH - 9]}_{digest}"


def _require_unique(servers: Sequence[McpServer]) -> None:
    names = [server.name for server in servers]
    if len(set(names)) != len(names):
        raise ValueError("MCP server names must be unique")


def _transport_label(server: McpServer) -> str:
    """Name one configured transport without exposing credentials or arguments."""
    match server:
        case McpHttpServer():
            return "streamable HTTP"
        case McpStdioServer():
            return "stdio child process"
    raise ValueError(f"Unsupported MCP server: {type(server).__name__}")


def _default_client_factory(server: McpServer) -> AbstractAsyncContextManager[McpClient]:
    from mcp import Client, StdioServerParameters

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
    """Connect Streamable HTTP with caller headers through the SDK's public API.

    ``streamable_http_client(url, http_client=...)`` is the documented way to
    supply an HTTP client, and ``httpx2`` is the SDK's own public HTTP library,
    so no private SDK module is imported. Redirects and timeouts mirror the
    SDK defaults: a short connect/write budget and a long read for SSE streams.
    """
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    timeout = httpx2.Timeout(_HTTP_TIMEOUT, read=_HTTP_SSE_READ_TIMEOUT)
    async with httpx2.AsyncClient(headers=dict(server.headers), timeout=timeout, follow_redirects=True) as http_client:
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
