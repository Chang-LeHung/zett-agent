"""Exercise MCP registration, structured results, and cleanup with an in-memory client."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp_types import CallToolResult, ListToolsResult, Tool

from zett_agent.client import create_agent
from zett_agent.extensions.mcp import McpClient, McpExtension, McpHttpServer, McpServer
from zett_agent.messages import AssistantMessage, ToolCall, ToolMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions


class LocalMcpClient:
    closed = False

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        assert cursor is None
        return ListToolsResult(
            tools=[
                Tool(
                    name="answer",
                    description="Return the demonstration answer.",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )

    async def call_tool(self, name: str, arguments: dict | None = None) -> CallToolResult:
        assert name == "answer" and arguments == {}
        return CallToolResult(content=[], structuredContent={"answer": 42})


class McpModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            assert request.messages[-1].success
            assert request.messages[-1].content == '{"answer": 42}'
            message = AssistantMessage(content="MCP tool result: 42")
        else:
            assert any(definition.name == "demo__answer" for definition in request.tools)
            message = AssistantMessage(tool_calls=(ToolCall("mcp-1", "demo__answer"),))
        yield ModelEvent.completed(ModelResponse(message))


async def main() -> None:
    local = LocalMcpClient()

    @asynccontextmanager
    async def connect(server: McpServer) -> AsyncIterator[McpClient]:
        assert server.name == "demo"
        try:
            yield local
        finally:
            local.closed = True

    extension = McpExtension(
        servers=[McpHttpServer(name="demo", url="https://unused.example/mcp")],
        client_factory=connect,
    )
    client = await create_agent(McpModel(), extensions=[extension])
    reply = await client.run("Get the answer.")
    assert reply.content == "MCP tool result: 42"
    assert local.closed
    print(reply.content)
    print("MCP connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
