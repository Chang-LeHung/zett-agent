"""Typed synchronous entry points around the ordinary asynchronous Agent."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import copy_context
from typing import Any, Self

from .agent import Agent, AgentRunConfig
from .client import AgentClient
from .dispatcher import AgentEventDispatcher
from .events import AgentEvent
from .extensions.base import AgentExtension
from .extensions.external import ExternalEvent
from .ids import new_uuid7
from .json_types import JsonValue
from .messages import AssistantMessage, UserMessage
from .model import AgentModel, ModelEvent, ModelRequest, ReasoningEffort
from .sync_runtime import SyncMethodsMixin, SyncObject, SyncRuntime, SyncStream, _callback_loop, _worker
from .tools import AgentTool


class SyncModelAdapter(SyncMethodsMixin):
    """Use a model whose stream(request) returns an ordinary synchronous iterator.

    Iteration and cleanup run in workers, so a blocking SDK never freezes the
    Agent event loop. The wrapped model and its transport remain caller-owned.
    Synchronous next() cannot be forcibly terminated; cancellation waits for the
    current call to return. Configure SDK transport timeouts accordingly.

    Examples:
        Implement an entirely synchronous teaching model::

            class Echo:
                def stream(self, request):
                    yield ModelEvent.text("Hello")
                    yield ModelEvent.completed(
                        ModelResponse(AssistantMessage(content="Hello"))
                    )

            with SyncAgent(SyncModelAdapter(Echo())) as agent:
                print(agent.run("Hi").content)
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Translate one blocking iterator into the existing async model protocol."""
        context = copy_context()
        context.run(_callback_loop.set, asyncio.get_running_loop())
        source = await _worker(context.run, self.model.stream, request)
        source = iter(source)
        sentinel = object()
        try:
            while True:
                event = await _worker(context.run, next, source, sentinel)
                if event is sentinel:
                    break
                yield event
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                await _worker(context.run, close)


class SyncAgent(SyncObject[Agent]):
    """An initialized Agent callable from ordinary synchronous Python code.

    All requests share one event loop so SDK connections and session resources
    survive across calls. Different sessions may be used by different calling
    threads. The underlying Agent still rejects overlapping requests within one
    session. Provider and extension resources remain caller-owned.

    The synchronous interface does not change Extension execution: lifecycle
    hooks remain async and receive the original AgentRunContext on the background
    loop. A supplied runtime is shared and is not closed when this view exits.

    Examples:
        Use a supplied model, persistence, and synchronous event handlers::

            with SyncAgent(model, extensions=[history]) as agent:
                answer = agent.run("Hello")
                with agent.stream("Continue") as events:
                    for event in events:
                        print(event.type, event.delta)

        Share a scope when a provider also needs explicit cleanup::

            with SyncRuntime() as runtime:
                provider = runtime.call(DeepSeekProvider, model="deepseek-chat")
                try:
                    with SyncAgent(provider, runtime=runtime) as agent:
                        print(agent.run("Hello").content)
                finally:
                    runtime.call(provider.aclose)
    """

    def __init__(
        self,
        model: AgentModel,
        *,
        config: AgentRunConfig | None = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        parallel_tool_call: bool = True,
        max_iterations: int = 36,
        max_internal_messages: int = 8,
        event_dispatcher: AgentEventDispatcher | None = None,
        runtime: SyncRuntime | None = None,
    ) -> None:
        owner = runtime if runtime is not None else SyncRuntime()
        try:
            agent = owner.call(
                Agent.create,
                model,
                config=config if config is not None else AgentRunConfig(session_id=new_uuid7()),
                system_prompt=system_prompt,
                tools=tools,
                extensions=extensions,
                reasoning_effort=reasoning_effort,
                parallel_tool_call=parallel_tool_call,
                max_iterations=max_iterations,
                max_internal_messages=max_internal_messages,
            )
        except BaseException:
            if runtime is None:
                owner.close()
            raise
        super().__init__(agent, runtime=owner)
        object.__setattr__(self, "_owns_runtime", runtime is None)
        object.__setattr__(self, "_client", AgentClient(agent, event_dispatcher=event_dispatcher))

    def run(
        self,
        message: UserMessage | str,
        *,
        config: AgentRunConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage:
        """Return a final answer, preserving original errors and event callbacks."""
        return self.runtime.call(
            self._client.run,
            message,
            config=config,
            model=model,
            reasoning_effort=reasoning_effort,
            parallel_tool_call=parallel_tool_call,
            metadata=metadata,
            tags=tags,
        )

    def stream(
        self,
        message: UserMessage | str,
        *,
        config: AgentRunConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> SyncStream[AgentEvent]:
        """Return a lazy event iterator; the request starts on the first next().

        Use with agent.stream(...) as events when breaking early so cancellation
        and cleanup finish immediately. A plain for loop does not close an
        iterator on break. The same AgentEvent types as Agent.stream are yielded.
        """
        return self.runtime.stream(
            self._client.stream,
            message,
            config=config,
            model=model,
            reasoning_effort=reasoning_effort,
            parallel_tool_call=parallel_tool_call,
            metadata=metadata,
            tags=tags,
        )

    def emit_external_event(self, event: ExternalEvent, *, config: AgentRunConfig | None = None) -> list[str]:
        """Route Ask User, Plan Mode, and steering replies while generation waits."""
        return self.runtime.call(self.wrapped.emit_external_event, event, config=config)

    def close(self) -> None:
        """Close the owned runtime; an explicitly supplied runtime stays open."""
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self) -> Self:
        super().__enter__()
        return self


def create_agent_sync(
    model: AgentModel,
    *,
    config: AgentRunConfig | None = None,
    system_prompt: str = "You are a helpful assistant.",
    tools: Sequence[AgentTool] = (),
    extensions: Sequence[AgentExtension] | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    parallel_tool_call: bool = True,
    max_iterations: int = 36,
    max_internal_messages: int = 8,
    event_dispatcher: AgentEventDispatcher | None = None,
    runtime: SyncRuntime | None = None,
) -> SyncAgent:
    """Create a synchronous Agent with the same options as create_agent.

    Examples:
        No asyncio.run() or await is needed::

            with create_agent_sync(model) as agent:
                print(agent.run("Hello").content)
    """
    return SyncAgent(
        model,
        config=config,
        system_prompt=system_prompt,
        tools=tools,
        extensions=extensions,
        reasoning_effort=reasoning_effort,
        parallel_tool_call=parallel_tool_call,
        max_iterations=max_iterations,
        max_internal_messages=max_internal_messages,
        event_dispatcher=event_dispatcher,
        runtime=runtime,
    )
