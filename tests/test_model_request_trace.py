from zett_agent import (
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ServerToolDefinition,
    ToolDefinition,
    UserMessage,
)
from zett_agent.extensions.model_request_trace import (
    MODEL_REQUEST_TRACE_ATTRIBUTE,
    ModelRequestTraceExtension,
)


async def test_model_request_trace_attaches_final_tool_definitions_to_response():
    extension = ModelRequestTraceExtension()
    context = AgentRunContext(AgentRunConfig("session"), AgentState(), {})
    request = ModelRequest(
        messages=(UserMessage(content="Inspect"),),
        tools=(
            ToolDefinition(
                "read_file",
                "Read a file.",
                {"type": "object", "properties": {"path": {"type": "string"}}},
                deferred=True,
            ),
        ),
        server_tools=(
            ServerToolDefinition(
                "web_search",
                {"name": "web_search", "max_uses": 3},
            ),
        ),
    )

    async def call_next(next_request: ModelRequest):
        assert next_request is request
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Done")))

    events = [event async for event in extension.on_model_request(context, request, call_next)]

    trace = events[-1].response.message.attributes[MODEL_REQUEST_TRACE_ATTRIBUTE]
    assert trace == {
        "schema_version": 1,
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                "deferred": True,
            },
        ],
        "server_tools": [
            {
                "type": "web_search",
                "configuration": {"name": "web_search", "max_uses": 3},
            },
        ],
    }
