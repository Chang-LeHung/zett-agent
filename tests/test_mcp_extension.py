"""MCP server discovery, tool dispatch, and transport cleanup."""

import asyncio
import contextlib
import json
import socket
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

import zett_agent.extensions.mcp as mcp_module
from zett_agent._compat import UTC
from zett_agent.agent import (
    Agent,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
)
from zett_agent.extensions.events import RunCancelledEvent
from zett_agent.extensions.mcp import (
    McpConfiguration,
    McpExtension,
    McpHttpServer,
    McpStdioServer,
)
from zett_agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
)
from zett_agent.tools.base import render_tool_guidance


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


async def test_streamable_http_configuration_sends_headers_to_a_real_mcp_server(tmp_path, monkeypatch):
    # Loopback E2E traffic must not follow a developer's HTTP(S)_PROXY.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
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


async def test_stdio_configuration_runs_a_real_mcp_server(tmp_path):
    server_path = tmp_path / "stdio_server.py"
    server_path.write_text(
        """from mcp.server.mcpserver import MCPServer

server = MCPServer("zett-agent-stdio-e2e")

@server.tool()
def echo(text: str) -> dict[str, str]:
    \"\"\"Echo text through the stdio transport.\"\"\"
    return {"echo": text}

server.run()
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": sys.executable,
                        "args": [str(server_path)],
                        "cwd": str(tmp_path),
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
                message = AssistantMessage(tool_calls=(ToolCall("stdio-call", "local__echo", {"text": "hello"}),))
            else:
                tool_result = next(item for item in request.messages if isinstance(item, ToolMessage))
                message = AssistantMessage(content=f"MCP returned {tool_result.content}")
            yield ModelEvent.completed(ModelResponse(message))

    model = EchoModel()
    extension = McpExtension(config_path=config_path)
    agent = await Agent.create(model, config=AgentRunConfig("mcp-stdio-e2e"), extensions=[extension])

    answer = await agent.run("Call the local MCP tool")

    assert json.loads(answer.content.removeprefix("MCP returned ")) == {"echo": "hello"}
    assert {tool.name for tool in model.requests[0].tools} == {"local__echo"}
    assert extension.servers == (McpStdioServer("local", sys.executable, (str(server_path),), None, str(tmp_path)),)


async def test_explicit_stdio_configuration_passes_args_env_and_cwd_to_real_server(tmp_path):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir()
    server_path = tmp_path / "process_server.py"
    server_path.write_text(
        """import os
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("zett-agent-stdio-process-e2e")
configured_argument = sys.argv[1]

@server.tool()
def inspect_process() -> dict[str, str]:
    \"\"\"Return process settings received from the MCP client.\"\"\"
    return {
        "argument": configured_argument,
        "environment": os.environ["ZETT_MCP_E2E"],
        "cwd": os.getcwd(),
    }

server.run()
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "runtime": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(server_path), "configured-argument"],
                        "env": {"ZETT_MCP_E2E": "configured-environment"},
                        "cwd": str(working_directory),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class ProcessModel:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                message = AssistantMessage(tool_calls=(ToolCall("process-call", "runtime__inspect_process", {}),))
            else:
                tool_result = next(item for item in request.messages if isinstance(item, ToolMessage))
                message = AssistantMessage(content=tool_result.content)
            yield ModelEvent.completed(ModelResponse(message))

    model = ProcessModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("mcp-stdio-process-e2e"),
        extensions=[McpExtension(config_path=config_path)],
    )

    answer = await agent.run("Inspect the MCP child process")

    assert json.loads(answer.content) == {
        "argument": "configured-argument",
        "environment": "configured-environment",
        "cwd": str(working_directory),
    }
    assert {tool.name for tool in model.requests[0].tools} == {"runtime__inspect_process"}


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


async def test_mcp_extension_describes_config_file_and_namespaced_tools(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"servers": {"docs": {"type": "streamable-http", "url": "https://example.test/mcp"}}}),
        encoding="utf-8",
    )
    extension = McpExtension(config_path=config_path)
    context = AgentRunContext(AgentRunConfig("mcp-state"), AgentState(), {})

    await extension.on_state(context)

    message = next(
        item
        for item in context.state.messages
        if isinstance(item, SystemMessage) and item.content.startswith("# MCP servers")
    )
    assert str(config_path.resolve()) in message.content
    assert "`servers` or `mcpServers`" in message.content
    assert "`<server>__<tool>`" in message.content
    assert "- docs: streamable HTTP" in message.content
    assert "example.test" not in message.content


async def test_mcp_extension_reports_configuration_source_without_servers(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"servers": {}}), encoding="utf-8")

    configured = AgentRunContext(AgentRunConfig("mcp-state-empty"), AgentState(), {})
    await McpExtension(config_path=config_path).on_state(configured)
    supplied = AgentRunContext(AgentRunConfig("mcp-state-supplied"), AgentState(), {})
    await McpExtension([McpStdioServer("local", "secret-server", args=("--token=abc",))]).on_state(supplied)

    configured_content = next(item.content for item in configured.state.messages if isinstance(item, SystemMessage))
    assert str(config_path.resolve()) in configured_content
    assert "No MCP servers are configured" in configured_content
    supplied_content = next(item.content for item in supplied.state.messages if isinstance(item, SystemMessage))
    assert "supplied by the application" in supplied_content
    assert "- local: stdio child process" in supplied_content
    assert "secret-server" not in supplied_content
    assert "abc" not in supplied_content


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


async def test_mcp_extension_skips_a_failing_server_and_reports_it_to_the_model():
    class BrokenClient(FakeClient):
        async def list_tools(self, *, cursor=None):
            raise RuntimeError("discovery failed")

    factory = ClientFactory(
        {
            "first": FakeClient({None: ListToolsResult(tools=[remote_tool("search")])}),
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

    await extension.on_tool(context)
    await extension.on_state(context)

    assert set(context.tools) == {"first__search"}
    assert "broken: streamable HTTP (unavailable in this request: RuntimeError)" in context.state.messages[0].content
    assert factory.closed == ["broken"]
    await extension.on_error(context, RuntimeError("done"))
    assert sorted(factory.closed) == ["broken", "first"]


async def test_mcp_extension_connects_servers_concurrently_and_lazily_with_a_warm_cache():
    gate = asyncio.Event()
    active = 0
    peak = 0

    class SlowFactory(ClientFactory):
        def __call__(self, server):
            @asynccontextmanager
            async def connect():
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                self.opened.append(server.name)
                if active == 3:
                    gate.set()
                await asyncio.wait_for(gate.wait(), 2)  # deadlocks if connections were sequential
                try:
                    yield self.clients[server.name]
                finally:
                    active -= 1
                    self.closed.append(server.name)

            return connect()

    names = ("a", "b", "c")
    factory = SlowFactory(
        {
            name: FakeClient(
                {None: ListToolsResult(tools=[remote_tool("t")])},
                {"t": CallToolResult(content=[TextContent(text=name)])},
            )
            for name in names
        }
    )
    extension = McpExtension([McpHttpServer(n, f"https://{n}.test/mcp") for n in names], client_factory=factory)

    first = AgentRunContext(AgentRunConfig("cold"), AgentState(), {})
    await extension.on_tool(first)
    assert peak == 3
    assert set(first.tools) == {"a__t", "b__t", "c__t"}
    await extension.on_success(first, AssistantMessage(content="done"))
    assert sorted(factory.closed) == ["a", "b", "c"]

    factory.opened.clear()
    second = AgentRunContext(AgentRunConfig("warm"), AgentState(), {})
    await extension.on_tool(second)
    assert factory.opened == []  # cached tool lists start nothing
    assert await second.tools["b__t"]({}) == [{"type": "text", "text": "b"}]
    assert factory.opened == ["b"]  # only the server that is actually used connects
    await extension.on_success(second, AssistantMessage(content="done"))
    assert factory.closed.count("b") == 2


async def test_mcp_extension_reloads_a_changed_configuration_file(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"a": {"type": "http", "url": "http://a"}}}), encoding="utf-8")
    extension = McpExtension(config_path=path)
    assert [server.name for server in extension.servers] == ["a"]

    path.write_text(
        json.dumps({"servers": {"a": {"type": "http", "url": "http://a"}, "b": {"type": "http", "url": "http://b"}}}),
        encoding="utf-8",
    )
    factory = ClientFactory(
        {name: FakeClient({None: ListToolsResult(tools=[remote_tool("t")])}) for name in ("a", "b")}
    )
    extension._client_factory = factory
    context = AgentRunContext(AgentRunConfig("reload"), AgentState(), {})
    await extension.on_tool(context)
    assert set(context.tools) == {"a__t", "b__t"}

    path.write_text("{not json", encoding="utf-8")  # a broken edit keeps the last good servers
    other = AgentRunContext(AgentRunConfig("reload-bad"), AgentState(), {})
    await extension.on_tool(other)
    assert set(other.tools) == {"a__t", "b__t"}
    await extension.on_success(context, AssistantMessage(content="x"))
    await extension.on_success(other, AssistantMessage(content="x"))


async def test_mcp_extension_closes_transports_after_compaction_and_via_request_finished():
    factory = ClientFactory({"server": FakeClient({None: ListToolsResult(tools=[remote_tool("t")])})})
    extension = McpExtension(
        [McpHttpServer("server", "https://example.test/mcp")], client_factory=factory, tools_cache_ttl=0
    )
    context = AgentRunContext(AgentRunConfig("compact"), AgentState(), {})
    await extension.on_tool(context)
    await extension.on_compact(context)
    assert factory.closed == ["server"]

    again = AgentRunContext(AgentRunConfig("finished"), AgentState(), {})
    await extension.on_tool(again)
    extension._request_finished(again)  # synchronous safety net
    await asyncio.sleep(0.05)
    assert factory.closed == ["server", "server"]


async def test_mcp_extension_times_out_slow_tools_and_connections():
    class SlowClient(FakeClient):
        async def call_tool(self, name, arguments=None):
            await asyncio.sleep(10)

    factory = ClientFactory({"slow": SlowClient({None: ListToolsResult(tools=[remote_tool("t")])})})
    extension = McpExtension([McpHttpServer("slow", "http://slow")], client_factory=factory, call_timeout=0.05)
    context = AgentRunContext(AgentRunConfig("slow-call"), AgentState(), {})
    await extension.on_tool(context)
    with pytest.raises(TimeoutError, match="timed out"):
        await context.tools["slow__t"]({})
    await extension.on_error(context, RuntimeError("x"))

    @asynccontextmanager
    async def hang(server):
        await asyncio.sleep(10)
        yield None

    stuck = McpExtension([McpHttpServer("stuck", "http://stuck")], client_factory=hang, connect_timeout=0.05)
    other = AgentRunContext(AgentRunConfig("slow-connect"), AgentState(), {})
    await stuck.on_tool(other)
    assert other.tools == {}


async def test_mcp_extension_rejects_repeating_cursors_and_sanitizes_tool_names():
    loop_client = FakeClient(
        {None: ListToolsResult(tools=[], nextCursor="x"), "x": ListToolsResult(tools=[], nextCursor="x")}
    )
    odd = FakeClient({None: ListToolsResult(tools=[remote_tool("a.b c"), remote_tool("n" * 100)])})
    factory = ClientFactory({"loop": loop_client, "odd": odd})
    extension = McpExtension(
        [McpHttpServer("loop", "http://loop"), McpHttpServer("odd", "http://odd")], client_factory=factory
    )
    context = AgentRunContext(AgentRunConfig("names"), AgentState(), {})
    await extension.on_tool(context)

    assert "loop__" not in "".join(context.tools)
    assert "odd__a_b_c" in context.tools
    assert all(len(name) <= 64 for name in context.tools)
    await extension.on_success(context, AssistantMessage(content="x"))


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

    # The extension imports the SDK inside the factory, so patch its source.
    monkeypatch.setattr("mcp.Client", FakeMcpSdkClient)
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

    def fake_create_http_client(*, headers, timeout, follow_redirects):
        captured["headers"] = headers
        captured["follow_redirects"] = follow_redirects
        captured["read_timeout"] = timeout.read
        return FakeHttpClient()

    def fake_streamable_http_client(url, *, http_client):
        captured["url"] = url
        captured["http_client"] = http_client
        return "transport"

    monkeypatch.setattr("httpx2.AsyncClient", fake_create_http_client)
    monkeypatch.setattr("mcp.client.streamable_http.streamable_http_client", fake_streamable_http_client)
    monkeypatch.setattr("mcp.Client", FakeSdkClient)
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
        "follow_redirects": True,
        "read_timeout": 300.0,
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


def test_mcp_configuration_loads_explicit_and_inferred_stdio_servers(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "implicit": {"command": "uvx", "args": ["implicit-server"]},
                    "explicit": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["server.py", "--quiet"],
                        "env": {"TOKEN": "secret"},
                        "cwd": "/tmp/project",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    configuration = McpConfiguration.from_file(config_path)

    assert configuration.servers == (
        McpStdioServer("implicit", "uvx", ("implicit-server",)),
        McpStdioServer("explicit", "python", ("server.py", "--quiet"), {"TOKEN": "secret"}, "/tmp/project"),
    )


def test_mcp_extension_without_arguments_allows_a_missing_default_configuration(monkeypatch, tmp_path):
    config_path = tmp_path / "missing.json"
    monkeypatch.setattr(mcp_module, "DEFAULT_MCP_CONFIG_PATH", config_path)

    extension = McpExtension()

    assert extension.config_path == config_path.resolve()
    assert extension.servers == ()


async def test_mcp_extension_keeps_a_configured_location_that_has_no_file_yet(tmp_path):
    config_path = tmp_path / "mcp.json"
    extension = McpExtension(config_path=config_path)
    context = AgentRunContext(AgentRunConfig("mcp-state-missing"), AgentState(), {})

    await extension.on_state(context)

    assert extension.config_path == config_path.resolve()
    assert extension.servers == ()
    message = next(item for item in context.state.messages if isinstance(item, SystemMessage))
    assert str(config_path.resolve()) in message.content
    assert "No MCP servers are configured" in message.content


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
        ({"servers": {"docs": {"type": "stdio", "url": "https://docs.test"}}}, "string command"),
        ({"servers": {"docs": {"type": "websocket", "url": "wss://docs.test"}}}, "unsupported transport"),
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
        ({"servers": {"local": {"command": "python", "args": "server.py"}}}, "array of strings"),
        ({"servers": {"local": {"command": "python", "args": [1]}}}, "array of strings"),
        ({"servers": {"local": {"command": "python", "env": []}}}, "object with string values"),
        ({"servers": {"local": {"command": "python", "env": {"TOKEN": 42}}}}, "object with string values"),
        ({"servers": {"local": {"command": "python", "cwd": []}}}, "cwd must be a string"),
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


async def test_mcp_extension_handles_duplicate_names_page_limits_and_repeated_setup():
    duplicate = FakeClient({None: ListToolsResult(tools=[remote_tool("same")])})
    endless = FakeClient(
        {
            None: ListToolsResult(tools=[], nextCursor="0"),
            **{str(i): ListToolsResult(tools=[], nextCursor=str(i + 1)) for i in range(200)},
        }
    )
    factory = ClientFactory({"one": duplicate, "two": duplicate, "endless": endless})
    extension = McpExtension(
        [McpHttpServer(n, f"http://{n}") for n in ("one", "two", "endless")],
        namespace_tools=False,
        client_factory=factory,
    )
    context = AgentRunContext(AgentRunConfig("dup"), AgentState(), {})
    await extension.on_tool(context)
    await extension.on_tool(context)  # a repeated setup closes the previous attempt first
    assert factory.closed.count("endless") == 2
    assert set(context.tools) == {"same"}
    await extension.on_state(context)
    assert "endless: streamable HTTP (unavailable" in context.state.messages[0].content
    await extension.on_success(context, AssistantMessage(content="x"))


async def test_mcp_extension_survives_close_errors_and_stuck_transports(monkeypatch):
    monkeypatch.setattr(mcp_module, "_CLOSE_TIMEOUT", 0.05)
    state = {"mode": "error"}

    @asynccontextmanager
    async def factory(server):
        try:
            yield FakeClient({None: ListToolsResult(tools=[])})
        finally:
            if state["mode"] == "error":
                raise RuntimeError("close failed")
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(10)  # ignores the first cancellation

    extension = McpExtension([McpHttpServer("s", "http://s")], client_factory=factory, tools_cache_ttl=0)
    for mode in ("error", "stuck"):
        state["mode"] = mode
        context = AgentRunContext(AgentRunConfig(mode), AgentState(), {})
        await extension.on_tool(context)
        await extension.on_success(context, AssistantMessage(content="done"))  # never raises

    connection = mcp_module._Connection(McpHttpServer("s", "http://s"), factory, 1)
    await connection.close()
    with pytest.raises(RuntimeError, match="already closed"):
        await connection.client()


async def test_mcp_extension_closes_everything_when_discovery_is_cancelled():
    started = asyncio.Event()

    @asynccontextmanager
    async def factory(server):
        started.set()
        await asyncio.sleep(10)
        yield None

    closed = ClientFactory({"fast": FakeClient({None: ListToolsResult(tools=[])})})
    extension = McpExtension(
        [McpHttpServer("fast", "http://fast")],
        client_factory=lambda server: factory(server),
    )
    context = AgentRunContext(AgentRunConfig("cancel-discovery"), AgentState(), {})
    task = asyncio.create_task(extension.on_tool(context))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert extension._requests == {}
    assert closed.opened == []


async def test_mcp_extension_tracks_the_configuration_file_lifecycle(tmp_path):
    path = tmp_path / "mcp.json"
    extension = McpExtension(config_path=path)
    assert extension.servers == ()
    path.write_text(json.dumps({"servers": {"a": {"type": "http", "url": "http://a"}}}), encoding="utf-8")
    context = AgentRunContext(AgentRunConfig("file"), AgentState(), {})
    await extension.on_state(context)  # reads the file lazily when no request owns the servers
    await extension.on_tool(context)
    assert [server.name for server in extension.servers] == ["a"]
    path.unlink()
    await extension.on_tool(context)
    assert extension.servers == ()
    with pytest.raises(ValueError, match="Invalid MCP configuration JSON"):
        path.write_text("{", encoding="utf-8")
        McpExtension(config_path=path)
    with pytest.raises(ValueError, match="Unsupported MCP server"):
        mcp_module._transport_label(object())


def test_non_strict_configuration_skips_invalid_entries_instead_of_failing_the_file(tmp_path, caplog):
    payload = {
        "servers": {
            "good": {"type": "http", "url": "http://good"},
            "bad-transport": {"type": "sse", "url": "http://sse"},
            "bad-url": {"type": "http", "url": "  "},
            "not-an-object": 3,
        }
    }
    with pytest.raises(ValueError, match="unsupported transport"):
        McpConfiguration.from_mapping(payload)  # strict is the default

    configuration = McpConfiguration.from_mapping(payload, strict=False)
    assert [server.name for server in configuration.servers] == ["good"]
    assert set(configuration.skipped) == {"bad-transport", "bad-url", "not-an-object"}
    assert "unsupported transport" in configuration.skipped["bad-transport"]
    assert "Skipping invalid MCP server" in caplog.text

    # A document with no usable server mapping still raises: nothing to salvage.
    with pytest.raises(ValueError, match="must contain one of"):
        McpConfiguration.from_mapping({}, strict=False)

    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported transport"):
        McpExtension(config_path=path)
    extension = McpExtension(config_path=path, strict_config=False)
    assert [server.name for server in extension.servers] == ["good"]
    assert [server.name for server in McpConfiguration.from_file(path, strict=False).servers] == ["good"]


async def test_mcp_tools_send_a_compact_schema_and_no_repeated_guidance():
    noisy = Tool(
        name="search",
        description="Search the docs.",
        inputSchema={
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "searchArguments",
            "type": "object",
            "properties": {
                "query": {"title": "Query", "type": "string", "description": "What to find"},
                "filters": {
                    "title": "Filters",
                    "type": "object",
                    "properties": {"title": {"title": "Title", "type": "string"}},
                    "additionalProperties": {"title": "Extra", "type": "number"},
                },
                "mode": {"title": "Mode", "anyOf": [{"title": "A", "const": "a"}, {"title": "B", "const": "b"}]},
            },
            "required": ["query"],
        },
    )
    factory = ClientFactory({"docs": FakeClient({None: ListToolsResult(tools=[noisy, remote_tool("fetch")])})})
    extension = McpExtension([McpHttpServer("docs", "http://docs")], client_factory=factory)
    context = AgentRunContext(AgentRunConfig("schema"), AgentState(), {})
    await extension.on_tool(context)

    search = context.tools["docs__search"]
    assert search.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to find"},
            # A property *named* "title" is a name, not an annotation, and is kept.
            "filters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "additionalProperties": {"type": "number"},
            },
            "mode": {"anyOf": [{"const": "a"}, {"const": "b"}]},
        },
        "required": ["query"],
    }
    # The server-level system message already explains namespacing; no per-tool boilerplate.
    assert all(registered.guidelines == () for registered in context.tools.values())
    assert render_tool_guidance(tuple(context.tools.values())) == ""
    await extension.on_success(context, AssistantMessage(content="done"))
