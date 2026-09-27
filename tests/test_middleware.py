"""Provider-request and local-tool middleware chains."""

import asyncio
from contextlib import aclosing
from dataclasses import replace

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
)
from zett_agent.client import create_agent
from zett_agent.extensions.base import (
    AgentExtension,
    MiddlewareHook,
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
from zett_agent.sync import SyncAgent
from zett_agent.tools.base import current_tool_call_id, tool


def test_agent_extension_owns_the_middleware_contract() -> None:
    assert issubclass(AgentExtension, MiddlewareHook)


async def test_model_request_middleware_wraps_provider_outside_in_and_can_replace_request() -> None:
    order: list[str] = []

    class RequestMiddleware(AgentExtension):
        def __init__(self, name: str) -> None:
            self.label = name
            self.name = f"request-{name}"

        async def on_model_request(self, context, request, call_next):
            order.append(f"{self.label}:before:{context.config.session_id}")
            request = replace(
                request,
                messages=(*request.messages, SystemMessage(content=self.label)),
            )
            try:
                async with aclosing(call_next(request)) as events:
                    async for event in events:
                        yield event
            finally:
                order.append(f"{self.label}:after")

    class Model:
        async def stream(self, request):
            order.append("model")
            assert [message.content for message in request.messages[-2:]] == ["outer", "inner"]
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    client = await create_agent(
        Model(),
        config=AgentRunConfig("middleware-request"),
        extensions=[RequestMiddleware("outer"), RequestMiddleware("inner")],
    )

    result = await client.run("hello")

    assert result.content == "done"
    assert order == [
        "outer:before:middleware-request",
        "inner:before:middleware-request",
        "model",
        "inner:after",
        "outer:after",
    ]


async def test_tool_middleware_wraps_handler_and_can_adjust_arguments_and_result() -> None:
    order: list[str] = []

    @tool
    def calculate(value: int) -> str:
        """Calculate one test value.

        Args:
            value: Value to inspect.

        Snippet:
            calculate(value=1)

        Guidelines:
            - Use for middleware tests.
        """
        order.append(f"handler:{value}")
        return str(value)

    class ToolMiddleware(AgentExtension):
        def __init__(self, name: str) -> None:
            self.label = name
            self.name = f"tool-{name}"

        async def on_tool_call(self, context, call, call_next):
            order.append(f"{self.label}:before:{current_tool_call_id()}")
            call.arguments["value"] += 1
            result = await call_next()
            order.append(f"{self.label}:after")
            return f"{result}:{self.label}"

    class Model:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(tool_calls=(ToolCall("calculate-1", "calculate", {"value": 1}),))
                if len(self.requests) == 1
                else AssistantMessage(content="done")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("middleware-tool"),
        tools=[calculate],
        extensions=[ToolMiddleware("outer"), ToolMiddleware("inner")],
    )

    await agent.run("calculate")

    result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert result.content == "3:inner:outer"
    assert order == [
        "outer:before:calculate-1",
        "inner:before:calculate-1",
        "handler:3",
        "inner:after",
        "outer:after",
    ]


async def test_tool_middleware_isolated_for_parallel_calls_and_converts_errors_to_results() -> None:
    identities: dict[str, str | None] = {}

    @tool
    async def echo(value: str) -> str:
        """Return one value.

        Args:
            value: Value to return.

        Snippet:
            echo(value="hello")

        Guidelines:
            - Use for middleware tests.
        """
        await asyncio.sleep(0)
        return value

    class Guard(AgentExtension):
        async def on_tool_call(self, context, call, call_next):
            identities[call.id] = current_tool_call_id()
            if call.id == "blocked":
                raise PermissionError("blocked by middleware")
            return await call_next()

    class Model:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            message = (
                AssistantMessage(
                    tool_calls=(
                        ToolCall("allowed", "echo", {"value": "yes"}),
                        ToolCall("blocked", "echo", {"value": "no"}),
                    )
                )
                if len(self.requests) == 1
                else AssistantMessage(content="done")
            )
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("middleware-parallel"),
        tools=[echo],
        extensions=[Guard()],
    )

    await agent.run("run")

    assert identities == {"allowed": "allowed", "blocked": "blocked"}
    results = {
        message.tool_call_id: message for message in model.requests[1].messages if isinstance(message, ToolMessage)
    }
    assert results["allowed"].success
    assert not results["blocked"].success
    assert results["blocked"].content == "blocked by middleware"


def test_sync_agent_accepts_the_same_async_middleware() -> None:
    observed: list[str] = []

    class Observe(AgentExtension):
        async def on_model_request(self, context, request, call_next):
            observed.append(context.config.session_id)
            async with aclosing(call_next(request)) as events:
                async for event in events:
                    yield event

    class Model:
        async def stream(self, request):
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    with SyncAgent(
        Model(),
        config=AgentRunConfig("sync-middleware"),
        extensions=[Observe()],
    ) as agent:
        assert agent.run("hello").content == "done"

    assert observed == ["sync-middleware"]
