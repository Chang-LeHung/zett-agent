"""MCP server discovery, tool dispatch, and transport cleanup."""

import asyncio
import json
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

import zett_agent.extensions.mcp as mcp_module
from zett_agent import (
    Agent,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AssistantMessage,
    McpConfiguration,
    McpExtension,
    McpHttpServer,
    McpStdioServer,
    ModelEvent,
    ModelResponse,
    RunCancelledEvent,
    ToolCall,
    ToolMessage,
)


class FakeClient:
    def __init__(self, pages, results=None):
        self.pages = pages
        self.results = results or {}
        self.calls = []

    async def list_tools(self, *, cursor=None):
        return self.pages[cursor]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return self.results[name]


class ClientFactory:
    def __init__(self, clients):
        self.clients = clients
        self.opened = []
        self.closed = []

    def __call__(self, server):
        @asynccontextmanager
        async def connect():
            self.opened.append(server.name)
            try:
                yield self.clients[server.name]
            finally:
                self.closed.append(server.name)

        return connect()


def remote_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Remote {name}",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )


class RequiredHeaderMiddleware:
    """Reject HTTP traffic missing the credential expected by the E2E server."""

    def __init__(self, app: Any, name: str, value: str) -> None:
        self.app = app
        self.name = name.lower().encode("ascii")
        self.value = value.encode("utf-8")
        self.received: list[bytes | None] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        received = headers.get(self.name)
        self.received.append(received)
        if received != self.value:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b"missing required MCP header"})
            return
        await self.app(scope, receive, send)


@asynccontextmanager
async def running_http_server(app: Any) -> AsyncGenerator[str, None]:
    """Run one real ASGI server on an ephemeral loopback TCP port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(200):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise TimeoutError("Test MCP server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


async def test_streamable_http_configuration_sends_headers_to_a_real_mcp_server(tmp_path):
    expected_header = "Bearer e2e-secret"
    mcp_server = MCPServer("zett-agent-header-e2e")

    @mcp_server.tool()
    def protected_echo(text: str) -> dict[str, str]:
        """Return text only after HTTP middleware has authenticated the request."""
        return {"echo": text}

    app = RequiredHeaderMiddleware(
        mcp_server.streamable_http_app(stateless_http=True),
        "authorization",
        expected_header,
    )
    async with running_http_server(app) as base_url:
        endpoint = f"{base_url}/mcp"
        async with httpx.AsyncClient() as unauthenticated:
            response = await unauthenticated.post(endpoint, content=b"{}")
        assert response.status_code == 401

        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "protected": {
                            "type": "streamable-http",
                            "url": endpoint,
                            "headers": {"Authorization": expected_header},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        class EchoModel:
            def __init__(self) -> None:
                self.requests = []

            async def stream(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    message = AssistantMessage(
                        tool_calls=(ToolCall("e2e-call", "protected__protected_echo", {"text": "hello"}),)
                    )
                else:
                    tool_result = next(item for item in request.messages if isinstance(item, ToolMessage))
                    message = AssistantMessage(content=f"MCP returned {tool_result.content}")
                yield ModelEvent.completed(ModelResponse(message))

        model = EchoModel()
        extension = McpExtension(config_path=config_path)
        agent = await Agent.create(model, config=AgentRunConfig("mcp-header-e2e"), extensions=[extension])

        answer = await agent.run("Call the protected MCP tool")

    assert json.loads(answer.content.removeprefix("MCP returned ")) == {"echo": "hello"}
    assert {tool.name for tool in model.requests[0].tools} == {"protected__protected_echo"}
    assert app.received[0] is None
    assert len(app.received) >= 2
    assert all(value == expected_header.encode() for value in app.received[1:])


async def test_mcp_extension_discovers_all_pages_executes_tool_and_closes_on_success():
    client = FakeClient(
        {
            None: ListToolsResult(tools=[remote_tool("search")], nextCursor="page-2"),
            "page-2": ListToolsResult(tools=[remote_tool("fetch")]),
        },
        {"search": CallToolResult(content=[], structuredContent={"matches": ["one"]})},
    )
    factory = ClientFactory({"docs": client})

    class Model:
        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(tool_calls=(ToolCall("mcp-1", "docs__search", {"query": "zett"}),))
                if len(self.requests) == 1
                else AssistantMessage(content="done")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    extension = McpExtension([McpHttpServer("docs", "http://127.0.0.1:8000/mcp")], client_factory=factory)
    agent = await Agent.create(model, config=AgentRunConfig("mcp-success"), extensions=[extension])

    await agent.run("Search docs")

    assert {tool.name for tool in model.requests[0].tools} == {"docs__search", "docs__fetch"}
    assert client.calls == [("search", {"query": "zett"})]
    result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert result.content == '{"matches": ["one"]}'
    assert factory.opened == ["docs"]
    assert factory.closed == ["docs"]


async def test_mcp_extension_preserves_unstructured_content_and_reports_remote_errors():
    client = FakeClient(
        {None: ListToolsResult(tools=[remote_tool("okay"), remote_tool("broken")])},
        {
            "okay": CallToolResult(content=[TextContent(text="plain result")]),
            "broken": CallToolResult(content=[TextContent(text="remote failure")], isError=True),
        },
    )
    factory = ClientFactory({"server": client})
    extension = McpExtension([McpStdioServer("server", "fake-server")], client_factory=factory)
    context = AgentRunContext(AgentRunConfig("direct"), AgentState(), {}, extensions=(extension,))
    await extension.on_tool(context)

    assert await context.tools["server__okay"]({"query": "x"}) == [{"type": "text", "text": "plain result"}]
    with pytest.raises(RuntimeError, match="remote failure"):
        await context.tools["server__broken"]({"query": "x"})

    await extension.on_error(context, RuntimeError("request failed"))
    assert factory.closed == ["server"]


async def test_mcp_extension_supports_unprefixed_tools_and_fallback_messages():
    unnamed = Tool(name="remote", inputSchema={"type": "object", "properties": {}})
    client = FakeClient(
        {None: ListToolsResult(tools=[unnamed])},
        {"remote": CallToolResult(content=[], isError=True)},
    )
    factory = ClientFactory({"server": client})
    extension = McpExtension(
        [McpHttpServer("server", "https://example.test/mcp")],
        namespace_tools=False,
        client_factory=factory,
    )
    context = AgentRunContext(AgentRunConfig("unprefixed"), AgentState(), {})
    await extension.on_tool(context)

    assert set(context.tools) == {"remote"}
    assert context.tools["remote"].description == "Call remote on the server MCP server."
    with pytest.raises(RuntimeError, match="MCP tool 'remote' failed"):
        await context.tools["remote"]({})
    await extension.on_success(context, AssistantMessage(content="done"))


async def test_mcp_extension_closes_connections_on_cancellation():
    client = FakeClient({None: ListToolsResult(tools=[])})
    factory = ClientFactory({"server": client})
    extension = McpExtension([McpHttpServer("server", "https://example.test/mcp")], client_factory=factory)
    context = AgentRunContext(AgentRunConfig("cancelled"), AgentState(), {}, extensions=(extension,))
    await extension.on_tool(context)

    await extension.on_event(
        context,
        RunCancelledEvent(
            previous_phase=context.state.phase,
            occurred_at=datetime.now(UTC),
            monotonic_ns=1,
        ),
    )

    assert factory.closed == ["server"]
    await extension.on_success(context, AssistantMessage(content="already closed"))
    assert factory.closed == ["server"]


async def test_mcp_extension_unwinds_open_connections_when_discovery_fails():
    class BrokenClient(FakeClient):
        async def list_tools(self, *, cursor=None):
            raise RuntimeError("discovery failed")

    factory = ClientFactory(
        {
            "first": FakeClient({None: ListToolsResult(tools=[])}),
            "broken": BrokenClient({}),
        }
    )
    extension = McpExtension(
        [
            McpHttpServer("first", "https://first.test/mcp"),
            McpHttpServer("broken", "https://broken.test/mcp"),
        ],
        client_factory=factory,
    )
    context = AgentRunContext(AgentRunConfig("failure"), AgentState(), {})

    with pytest.raises(RuntimeError, match="discovery failed"):
        await extension.on_tool(context)

    assert factory.opened == ["first", "broken"]
    assert factory.closed == ["broken", "first"]


def test_mcp_configuration_rejects_ambiguous_or_invalid_names():
    with pytest.raises(ValueError, match="name cannot be empty"):
        McpHttpServer(" ", "https://example.test/mcp")
    with pytest.raises(ValueError, match="URL cannot be empty"):
        McpHttpServer("server", " ")
    with pytest.raises(ValueError, match="command cannot be empty"):
        McpStdioServer("server", " ")
    with pytest.raises(ValueError, match="cannot contain"):
        McpHttpServer("bad__name", "https://example.test/mcp")
    with pytest.raises(ValueError, match="must be unique"):
        McpExtension(
            [
                McpHttpServer("same", "https://one.test/mcp"),
                McpStdioServer("same", "server"),
            ]
        )
    with pytest.raises(ValueError, match="header names"):
        McpHttpServer("server", "https://example.test/mcp", {"": "token"})
    with pytest.raises(ValueError, match="string value"):
        McpHttpServer("server", "https://example.test/mcp", {"Authorization": 42})


def test_default_client_factory_maps_http_and_stdio_configuration(monkeypatch, tmp_path):
    captured = []

    class FakeMcpSdkClient:
        def __init__(self, server):
            captured.append(server)

    monkeypatch.setattr(mcp_module, "Client", FakeMcpSdkClient)
    http = McpHttpServer("http", "https://example.test/mcp")
    stdio = McpStdioServer("stdio", "python", ("server.py",), {"TOKEN": "secret"}, tmp_path)

    assert isinstance(mcp_module._default_client_factory(http), FakeMcpSdkClient)
    assert isinstance(mcp_module._default_client_factory(stdio), FakeMcpSdkClient)
    assert captured[0] == http.url
    assert captured[1].command == "python"
    assert captured[1].args == ["server.py"]
    assert captured[1].env == {"TOKEN": "secret"}
    assert captured[1].cwd == tmp_path


async def test_default_client_factory_passes_headers_to_streamable_http_client(monkeypatch):
    captured = {}

    class FakeHttpClient:
        async def __aenter__(self):
            captured["http_entered"] = True
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            captured["http_closed"] = True

    class FakeSdkClient:
        def __init__(self, transport):
            captured["transport"] = transport

        async def __aenter__(self):
            captured["mcp_entered"] = True
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            captured["mcp_closed"] = True

    def fake_create_http_client(*, headers):
        captured["headers"] = headers
        return FakeHttpClient()

    def fake_streamable_http_client(url, *, http_client):
        captured["url"] = url
        captured["http_client"] = http_client
        return "transport"

    monkeypatch.setattr(mcp_module, "create_mcp_http_client", fake_create_http_client)
    monkeypatch.setattr(mcp_module, "streamable_http_client", fake_streamable_http_client)
    monkeypatch.setattr(mcp_module, "Client", FakeSdkClient)
    server = McpHttpServer(
        "secure",
        "https://secure.test/mcp",
        {"Authorization": "Bearer secret", "X-Tenant-ID": "tenant-1"},
    )

    assert "secret" not in repr(server)
    async with mcp_module._default_client_factory(server) as client:
        assert isinstance(client, FakeSdkClient)

    assert captured == {
        "headers": {"Authorization": "Bearer secret", "X-Tenant-ID": "tenant-1"},
        "http_entered": True,
        "url": "https://secure.test/mcp",
        "http_client": captured["http_client"],
        "transport": "transport",
        "mcp_entered": True,
        "mcp_closed": True,
        "http_closed": True,
    }


def test_mcp_extension_loads_default_streamable_http_configuration(monkeypatch, tmp_path):
    config_path = tmp_path / ".zett" / "mcp.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "docs": {"type": "streamable-http", "url": "https://docs.test/mcp"},
                    "search": {
                        "transport": "streamable",
                        "url": "https://search.test/mcp",
                        "headers": {"Authorization": "Bearer secret"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_module, "DEFAULT_MCP_CONFIG_PATH", config_path)

    extension = McpExtension()

    assert extension.config_path == config_path.resolve()
    assert extension.servers == (
        McpHttpServer("docs", "https://docs.test/mcp"),
        McpHttpServer("search", "https://search.test/mcp", {"Authorization": "Bearer secret"}),
    )


def test_mcp_extension_without_arguments_allows_a_missing_default_configuration(monkeypatch, tmp_path):
    config_path = tmp_path / "missing.json"
    monkeypatch.setattr(mcp_module, "DEFAULT_MCP_CONFIG_PATH", config_path)

    extension = McpExtension()

    assert extension.config_path == config_path.resolve()
    assert extension.servers == ()


def test_mcp_extension_merges_custom_configuration_and_explicit_servers(tmp_path):
    config_path = tmp_path / "custom-mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {"type": "http", "url": "https://remote.test/mcp"},
                }
            }
        ),
        encoding="utf-8",
    )

    extension = McpExtension(
        servers=[McpStdioServer("local", "python", ("server.py",))],
        config_path=config_path,
    )

    assert extension.servers == (
        McpHttpServer("remote", "https://remote.test/mcp"),
        McpStdioServer("local", "python", ("server.py",)),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a JSON object"),
        ({}, "must contain"),
        ({"servers": {}, "mcpServers": {}}, "must use exactly one"),
        ({"servers": []}, "'servers' must be a JSON object"),
        ({"servers": {"docs": []}}, "server 'docs' must be a JSON object"),
        ({"servers": {"docs": {"type": "stdio", "url": "https://docs.test"}}}, "Streamable HTTP"),
        ({"servers": {"docs": {"type": "streamable-http", "url": 42}}}, "string URL"),
        (
            {"servers": {"docs": {"type": "streamable-http", "url": "https://docs.test", "headers": []}}},
            "headers must be a JSON object",
        ),
        (
            {
                "servers": {
                    "docs": {
                        "type": "streamable-http",
                        "url": "https://docs.test",
                        "headers": {"Authorization": 42},
                    }
                }
            },
            "string value",
        ),
    ],
)
def test_load_mcp_servers_rejects_invalid_configuration(tmp_path, payload, message):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        McpConfiguration.from_file(config_path)


def test_load_mcp_servers_reports_invalid_json_and_missing_file(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid MCP configuration JSON"):
        McpConfiguration.from_file(config_path)
    with pytest.raises(FileNotFoundError):
        McpConfiguration.from_file(tmp_path / "missing.json")


def test_mcp_extension_rejects_duplicate_names_across_file_and_explicit_servers(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"servers": {"docs": {"type": "streamable", "url": "https://docs.test/mcp"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be unique"):
        McpExtension([McpHttpServer("docs", "https://other.test/mcp")], config_path=config_path)


def test_mcp_configuration_and_extension_accept_custom_server_root_keys(tmp_path):
    config_path = tmp_path / "custom-root.json"
    config_path.write_text(
        json.dumps(
            {
                "connections": {
                    "docs": {
                        "type": "streamable-http",
                        "url": "https://docs.test/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    configuration = McpConfiguration.from_file(config_path, server_keys=("connections",))
    extension = McpExtension(config_path=config_path, server_keys=("connections",))

    expected = (McpHttpServer("docs", "https://docs.test/mcp"),)
    assert configuration.servers == expected
    assert extension.servers == expected
    assert extension.server_keys == ("connections",)


@pytest.mark.parametrize(
    ("server_keys", "message"),
    [
        ((), "cannot be empty"),
        (("",), "non-empty strings"),
        (("servers", "servers"), "must be unique"),
    ],
)
def test_mcp_configuration_rejects_invalid_server_root_keys(server_keys, message):
    with pytest.raises(ValueError, match=message):
        McpConfiguration.from_mapping({"servers": {}}, server_keys=server_keys)
    with pytest.raises(ValueError, match=message):
        McpExtension(servers=[], server_keys=server_keys)
