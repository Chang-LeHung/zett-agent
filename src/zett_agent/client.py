"""Optional creation and event-handling conveniences around the unchanged Agent."""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing

from .agent import Agent, AgentConfig
from .dispatcher import AgentEventDispatcher
from .events import AgentEvent, AgentEventType
from .exceptions import AgentProtocolError
from .extensions.base import AgentExtension
from .ids import new_uuid7
from .json_types import JsonValue
from .messages import AssistantMessage, UserMessage
from .model import AgentModel, ReasoningEffort
from .tools import AgentTool


class AgentClient:
    """Compose an initialized Agent with optional application event callbacks.

    The public agent attribute exposes the original runtime for advanced APIs,
    including emit_external_event(). Calling it directly bypasses this client's
    callbacks. No methods or attributes are injected into the original Agent.

    Callback failures propagate to the caller and close the underlying stream:
    unfinished requests are cancelled through the Agent's normal close path.
    If a callback fails after completion, the request remains completed.
    Shared handlers must distinguish sessions using event.session_id.

    Example:
        agent = await Agent.create(model, config=AgentConfig(session_id="existing"))
        client = AgentClient(agent, event_dispatcher=ConsoleEvents())
        reply = await client.run("Continue")
    """

    def __init__(self, agent: Agent, *, event_dispatcher: AgentEventDispatcher | None = None) -> None:
        self.agent = agent
        self.event_dispatcher = event_dispatcher

    async def stream(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Dispatch then yield each original event, in order, exactly once.

        Use aclosing(client.stream(...)) when stopping consumption early. Do
        not dispatch yielded events manually if a handler is already bound.
        """
        async with aclosing(
            self.agent.stream(message, config=config, reasoning_effort=reasoning_effort, metadata=metadata, tags=tags)
        ) as events:
            async for event in events:
                if self.event_dispatcher is not None:
                    await self.event_dispatcher.dispatch(event)
                yield event

    async def run(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage:
        """Collect a complete stream with callbacks and return its final answer."""
        result: AssistantMessage | None = None
        async with aclosing(
            self.stream(message, config=config, reasoning_effort=reasoning_effort, metadata=metadata, tags=tags)
        ) as events:
            async for event in events:
                if event.type == AgentEventType.RUN_COMPLETED and isinstance(event.message, AssistantMessage):
                    result = event.message
        if result is None:
            raise AgentProtocolError("The agent did not produce a final answer")
        return result


async def create_agent(
    model: AgentModel,
    *,
    config: AgentConfig | None = None,
    system_prompt: str = "You are a helpful assistant.",
    tools: Sequence[AgentTool] = (),
    extensions: Sequence[AgentExtension] | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    max_iterations: int = 36,
    max_internal_messages: int = 8,
    event_dispatcher: AgentEventDispatcher | None = None,
) -> AgentClient:
    """Create an initialized runtime wrapped in an event-aware AgentClient.

    Omit config for a new UUIDv7 session, or pass an existing session ID to
    restore history through your configured persistence extension. All runtime
    options pass through unchanged; extensions=None keeps Agent defaults.
    The supplied model and extension resources remain caller-owned.

    Example:
        from zett_agent import AgentEvent, AgentEventDispatcher, create_agent

        class ConsoleEvents(AgentEventDispatcher):
            async def on_text_delta_event(self, event: AgentEvent) -> None:
                print(event.delta, end="", flush=True)

        client = await create_agent(model, event_dispatcher=ConsoleEvents())
        reply = await client.run("Hello")
        await client.run("Continue in the same session")
    """
    agent = await Agent.create(
        model,
        config=config if config is not None else AgentConfig(session_id=new_uuid7()),
        system_prompt=system_prompt,
        tools=tools,
        extensions=extensions,
        reasoning_effort=reasoning_effort,
        max_iterations=max_iterations,
        max_internal_messages=max_internal_messages,
    )
    return AgentClient(agent, event_dispatcher=event_dispatcher)
