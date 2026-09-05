import json
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing

from .events import AgentEvent, AgentEventType
from .exceptions import AgentIterationLimitError, AgentProtocolError
from .messages import AnyMessage, AssistantMessage, SystemMessage, ToolMessage, UserMessage
from .model import AgentModel, ModelEventType, ModelRequest, ReasoningEffort
from .tools import AgentTool, render_tool_guidance


class Agent:
    """A model/tool loop. The caller owns history, persistence, and cancellation.

    Read stream() from top to bottom:
        input -> model -> tool results -> model -> final answer
    """

    def __init__(
        self,
        model: AgentModel,
        *,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        max_iterations: int = 36,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self.max_iterations = max_iterations

    async def run(
        self,
        message: UserMessage,
        *,
        history: Sequence[AnyMessage] = (),
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AssistantMessage:
        """Collect the stream and return the final answer."""
        async with aclosing(self.stream(message, history=history, reasoning_effort=reasoning_effort)) as events:
            async for event in events:
                if event.type == AgentEventType.RUN_COMPLETED and isinstance(event.message, AssistantMessage):
                    return event.message
        raise AgentProtocolError("The agent did not produce a final answer")

    async def stream(
        self,
        message: UserMessage,
        *,
        history: Sequence[AnyMessage] = (),
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AsyncIterator[AgentEvent]:
        """Stream each model and tool step; task cancellation propagates normally."""
        instructions = [self.system_prompt]
        instructions.extend(item.content for item in history if isinstance(item, SystemMessage))
        instructions.append(render_tool_guidance(tuple(self.tools.values())))
        prompt = "\n\n".join(part for part in instructions if part)
        messages: list[AnyMessage] = [SystemMessage(content=prompt)] if prompt else []
        messages.extend(item for item in history if not isinstance(item, SystemMessage))
        messages.append(message)

        for _ in range(self.max_iterations):
            request = ModelRequest(
                messages=tuple(messages),
                tools=tuple(tool.definition for tool in self.tools.values()),
                reasoning_effort=reasoning_effort,
            )
            yield AgentEvent(AgentEventType.MODEL_STARTED)
            response = None
            async with aclosing(self.model.stream(request)) as events:
                async for event in events:
                    if response is not None:
                        raise AgentProtocolError("Model emitted events after its final response")
                    match event.type:
                        case ModelEventType.TEXT_DELTA:
                            yield AgentEvent(AgentEventType.TEXT_DELTA, delta=event.delta)
                        case ModelEventType.REASONING_DELTA:
                            yield AgentEvent(AgentEventType.REASONING_DELTA, delta=event.delta)
                        case ModelEventType.TOOL_CALL_DELTA:
                            if event.tool_call_delta is None:
                                raise AgentProtocolError("Missing tool-call delta")
                            yield AgentEvent(AgentEventType.TOOL_CALL_DELTA, tool_call_delta=event.tool_call_delta)
                        case ModelEventType.RESPONSE:
                            if event.response is None:
                                raise AgentProtocolError("Missing model response")
                            response = event.response
            if response is None:
                raise AgentProtocolError("Model stream ended without a response")
            messages.append(response.message)
            yield AgentEvent(AgentEventType.MODEL_COMPLETED, response=response)
            if not response.message.tool_calls:
                yield AgentEvent(AgentEventType.RUN_COMPLETED, message=response.message)
                return

            for call in response.message.tool_calls:
                yield AgentEvent(AgentEventType.TOOL_STARTED, call=call)
                error = None
                try:
                    registered = self.tools.get(call.name)
                    if registered is None:
                        raise ValueError(f"Unknown tool: {call.name}")
                    output = await registered(call.arguments)
                    content = registered.serialize_result(output)
                except Exception as exc:
                    error = exc
                    content = json.dumps({"error": str(exc)})
                result = ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=content,
                    success=error is None,
                )
                messages.append(result)
                yield AgentEvent(
                    AgentEventType.TOOL_FAILED if error else AgentEventType.TOOL_COMPLETED,
                    call=call,
                    message=result,
                    error=error,
                )
        raise AgentIterationLimitError(f"Agent exceeded {self.max_iterations} model iterations")
