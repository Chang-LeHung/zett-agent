"""MCP server discovery, tool dispatch, and transport cleanup."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

import zett_agent.extensions.mcp as mcp_module
from zett_agent import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentState,
    AssistantMessage,
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
    agent = await Agent.create(model, config=AgentConfig("mcp-success"), extensions=[extension])

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
    context = AgentContext(AgentConfig("direct"), AgentState(), {}, extensions=(extension,))
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
    context = AgentContext(AgentConfig("unprefixed"), AgentState(), {})
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
    context = AgentContext(AgentConfig("cancelled"), AgentState(), {}, extensions=(extension,))
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
    context = AgentContext(AgentConfig("failure"), AgentState(), {})

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
