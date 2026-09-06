import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Self, overload

from .events import AgentEvent, AgentEventType, AgentPhase, AgentPhaseTransitionMixin, ModelOutputTracker
from .exceptions import AgentIterationLimitError, AgentProtocolError
from .extension_events import ExtensionEvent, MessageAppendedEvent, MessageTiming
from .extension_hooks import AgentExtension
from .external_events import ExternalEvent
from .json_types import JsonValue, json_object
from .messages import AnyMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from .model import AgentModel, ModelEventType, ModelRequest, ModelResponse, ReasoningEffort
from .tools import AgentTool


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configuration that identifies one agent run."""

    session_id: str
    request_id: str | None = None
    parent_session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if self.request_id is not None and not self.request_id.strip():
            raise ValueError("request_id cannot be empty")
        if self.parent_session_id is not None:
            if not self.parent_session_id.strip():
                raise ValueError("parent_session_id cannot be empty")
            if self.parent_session_id == self.session_id:
                raise ValueError("parent_session_id must differ from session_id")


@dataclass(slots=True)
class AgentState:
    """Mutable conversation state created independently for one request."""

    messages: list[AnyMessage] = field(default_factory=list)
    phase: AgentPhase = AgentPhase.CREATED
    # Parent conversation when this request belongs to a delegated subagent.
    parent_session_id: str | None = None


@dataclass(slots=True, weakref_slot=True, eq=False)
class AgentContext:
    """Per-run references shared by all lifecycle hooks.

    Register request-scoped tools during on_tool(). Mutate state.messages and
    tools in place to update the current request without leaking registrations
    into another request or session.
    """

    # Configuration for this invocation.
    config: AgentConfig
    # Conversation state populated for the current request.
    state: AgentState
    # Live registry used for model schemas and tool execution.
    tools: dict[str, AgentTool]
    # Fixed priority order for this run; no event history is retained.
    extensions: tuple["AgentExtension", ...] = ()
    # Model owned by the current Agent; setup extensions may inspect it when
    # constructing request-scoped capabilities such as default subagents.
    model: AgentModel | None = None
    # Application input available to extensions and Raw Log persistence only.
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    # Request classifications available to extensions and Raw Log persistence.
    tags: dict[str, JsonValue] = field(default_factory=dict)

    def register_tool(self, tool: AgentTool) -> None:
        """Register one request-scoped tool while rejecting ambiguous names."""
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self.tools[tool.name] = tool

    async def publish(self, event: ExtensionEvent) -> None:
        """Deliver an event sequentially to all registered extensions.

        Handler errors propagate immediately and stop delivery. Already completed
        work is not rolled back. Subscribers may retain events themselves.

        Example:
            await context.publish(CompactionEvent(
                compressed_from=1, compressed_to=20,
                kept_from=21, kept_to=30, summary="Earlier decisions...",
            ))
        """
        for extension in self.extensions:
            await extension.on_event(self, event)

    async def append_message(self, message: AnyMessage, timing: MessageTiming) -> None:
        """Append one newly produced message and publish its Raw Log event.

        Runtime code must use this method for user, assistant, and tool messages
        so in-memory context and persistence notifications cannot drift apart.
        Restored history and generated system instructions are existing context,
        not new Raw Log messages, and therefore do not use this method.

        The message is visible in ``state.messages`` before subscribers run.
        Subscriber failures propagate and do not roll back the in-memory append.
        """
        self.state.messages.append(message)
        await self.publish(MessageAppendedEvent(message, timing))


class Agent(AgentPhaseTransitionMixin):
    """A small stateful model/tool loop with optional lifecycle extensions."""

    def __init__(
        self,
        model: AgentModel,
        *,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        max_iterations: int = 36,
    ) -> None:
        """Create an agent with its own default extensions.

        Omit extensions to enable InMemoryMessageAccumulator and
        ToolGuidelinesExtension. An explicit sequence replaces those defaults;
        pass an empty sequence to disable all extensions.

        Example:
            agent = await Agent.create(
                model,
                config=config,
                tools=[read_file],
                reasoning_effort=ReasoningEffort.HIGH,
            )
        """
        from .extensions import InMemoryMessageAccumulator, ToolGuidelinesExtension

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        configured_extensions = (
            (InMemoryMessageAccumulator(), ToolGuidelinesExtension()) if extensions is None else tuple(extensions)
        )
        # Python's sort is stable, so extensions sharing a priority preserve the
        # caller's registration order. Every lifecycle path uses this tuple.
        self.extensions = tuple(sorted(configured_extensions, key=lambda extension: extension.priority))
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.state = AgentState()
        self._initialized_config: AgentConfig | None = None
        self._request_active = False
        self.max_iterations = max_iterations

    async def initialize(self, *, config: AgentConfig) -> None:
        """Bind the default session used when a request omits config.

        Example:
            agent = Agent(model)
            await agent.initialize(config=AgentConfig(session_id="session-42"))
            reply = await agent.run("Hello")
        """
        if self._initialized_config is not None:
            if self._initialized_config.session_id != config.session_id:
                raise AgentProtocolError("Agent is already initialized for another session")
            return
        self._initialized_config = config

    def emit_external_event(self, event: ExternalEvent) -> bool:
        """Broadcast external input and report whether any extension accepted it.

        Every registered extension receives the event in priority order,
        including extensions after the first one that accepts it. This keeps the
        caller independent from the extension that owns a protocol.

        Example:
            accepted = agent.emit_external_event(
                ExternalEvent(
                    name="ask_user_response",
                    payload={"session_id": "session-42", "tool_call_id": "call-1", "answer": "Yes"},
                )
            )
        """
        accepted = False
        for extension in self.extensions:
            if extension.accept(event):
                accepted = True
        return accepted

    @classmethod
    async def create(
        cls,
        model: AgentModel,
        *,
        config: AgentConfig,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        max_iterations: int = 36,
    ) -> Self:
        """Construct and initialize an Agent before returning it.

        Example:
            agent = await Agent.create(model, config=AgentConfig(session_id="session-42"))
            reply = await agent.run("Hello")
        """
        agent = cls(
            model,
            system_prompt=system_prompt,
            tools=tools,
            extensions=extensions,
            reasoning_effort=reasoning_effort,
            max_iterations=max_iterations,
        )
        await agent.initialize(config=config)
        return agent

    @overload
    async def run(
        self,
        message: UserMessage,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage: ...

    @overload
    async def run(
        self,
        message: str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage: ...

    async def run(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage:
        """Collect one run and return its final answer.

        Omit reasoning_effort to use the default configured on this Agent.

        Example:
            agent = await Agent.create(model, config=AgentConfig(session_id="session-42"))
            reply = await agent.run(
                "Summarize this conversation.",
                config=AgentConfig(session_id="session-42"),
                metadata={"source": "editor"},
                tags={"domain": "notes"},
            )
        """
        result: AssistantMessage | None = None
        async with aclosing(
            self.stream(
                message,
                config=config,
                reasoning_effort=reasoning_effort,
                metadata=metadata,
                tags=tags,
            )
        ) as events:
            async for event in events:
                if event.type == AgentEventType.RUN_COMPLETED and isinstance(event.message, AssistantMessage):
                    result = event.message
        if result is not None:
            return result
        raise AgentProtocolError("The agent did not produce a final answer")

    @overload
    def stream(
        self,
        message: UserMessage,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    @overload
    def stream(
        self,
        message: str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def stream(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream one run while retaining messages in the agent state.

        Extensions receive the same mutable state and may add, remove, or replace
        messages at each lifecycle hook. Omit reasoning_effort to use the Agent's
        configured default; an explicit value overrides it for this request only.

        Example:
            agent = await Agent.create(model, config=config, extensions=[history_extension])
            async for event in agent.stream(
                "Continue the summary.",
                config=AgentConfig(session_id="session-42"),
                reasoning_effort=ReasoningEffort.HIGH,
                metadata={"source": "command-palette"},
                tags={"intent": "summary"},
            ):
                print(event.type, event.session_id)
        """
        if self._initialized_config is None:
            raise AgentProtocolError(
                "Agent is not initialized; await agent.initialize(config=...) or Agent.create(...)"
            )
        if self._request_active or not self.state.phase.accepts_new_request:
            raise AgentProtocolError(
                f"Agent cannot start a new request while its current phase is {self.state.phase.value!r}"
            )
        config = config or self._initialized_config
        request_reasoning_effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        messages = [SystemMessage(content=self.system_prompt)] if self.system_prompt else []
        # State and dynamic tools are request-scoped. Constructor tools are copied
        # so extension registrations cannot leak into later requests or sessions.
        state = AgentState(
            messages=messages,
            parent_session_id=config.parent_session_id,
        )
        self.state = state
        context = AgentContext(
            config=config,
            state=state,
            tools=dict(self.tools),
            extensions=self.extensions,
            model=self.model,
            metadata=json_object(metadata, field_name="Context metadata"),
            tags=json_object(tags, field_name="Context tags", nonempty_keys=True),
        )
        user_message = message if isinstance(message, UserMessage) else UserMessage(content=message)
        self._request_active = True
        try:
            await self._start_context_loading(context)
            await self._notify_on_tool(context)
            await self._notify_on_message(context)
            await self._finish_context_loading(context)
            await self._notify_before_run(context)
            await context.append_message(user_message, MessageTiming.instant())

            for _ in range(self.max_iterations):
                await self._notify_before_model(context)

                async with aclosing(self._before_model_events(context)) as preprocessing:
                    async for event in preprocessing:
                        await self._apply_extension_event_phase(context, event)
                        yield event

                request = ModelRequest(
                    messages=tuple(state.messages),
                    tools=tuple(tool.definition for tool in context.tools.values()),
                    reasoning_effort=request_reasoning_effort,
                )
                model_started = await self._start_model_generation(context)
                output_tracker = ModelOutputTracker()
                yield AgentEvent(
                    AgentEventType.MODEL_STARTED,
                    session_id=config.session_id,
                    phase=state.phase,
                )
                response = None
                async with aclosing(self.model.stream(request)) as events:
                    async for event in events:
                        if response is not None:
                            raise AgentProtocolError("Model emitted events after its final response")
                        await output_tracker.observe(context, event)
                        match event.type:
                            case ModelEventType.TEXT_DELTA:
                                yield AgentEvent(
                                    AgentEventType.TEXT_DELTA,
                                    session_id=config.session_id,
                                    phase=state.phase,
                                    delta=event.delta,
                                )
                            case ModelEventType.REASONING_DELTA:
                                yield AgentEvent(
                                    AgentEventType.REASONING_DELTA,
                                    session_id=config.session_id,
                                    phase=state.phase,
                                    delta=event.delta,
                                )
                            case ModelEventType.TOOL_CALL_DELTA:
                                if event.tool_call_delta is None:
                                    raise AgentProtocolError("Missing tool-call delta")
                                yield AgentEvent(
                                    AgentEventType.TOOL_CALL_DELTA,
                                    session_id=config.session_id,
                                    phase=state.phase,
                                    tool_call_delta=event.tool_call_delta,
                                )
                            case ModelEventType.RESPONSE:
                                if event.response is None:
                                    raise AgentProtocolError("Missing model response")
                                response = event.response
                if response is None:
                    raise AgentProtocolError("Model stream ended without a response")

                model_completed = await self._finish_model_generation(context)
                await context.append_message(
                    response.message,
                    output_tracker.message_timing(model_started, model_completed),
                )
                await self._notify_after_model(context, response)
                yield AgentEvent(
                    AgentEventType.MODEL_COMPLETED,
                    session_id=config.session_id,
                    phase=state.phase,
                    response=response,
                )
                if not response.message.tool_calls:
                    await self._notify_after_run(context, response.message)
                    await self._notify_on_success(context, response.message)
                    await self._complete_request(context)
                    yield AgentEvent(
                        AgentEventType.RUN_COMPLETED,
                        session_id=config.session_id,
                        phase=state.phase,
                        message=response.message,
                    )
                    return

                async with aclosing(self._execute_tools(context, response.message.tool_calls)) as tool_events:
                    async for event in tool_events:
                        yield event
            raise AgentIterationLimitError(f"Agent exceeded {self.max_iterations} model iterations")
        except (asyncio.CancelledError, GeneratorExit) as cancellation:
            try:
                await self._cancel_request(context)
            except Exception as notification_error:
                cancellation.add_note(f"Cancellation notification failed: {notification_error!r}")
            raise
        except Exception as error:
            # A terminal transition is already committed before subscribers run.
            # Never replace the original error with an illegal terminal transition
            # or a secondary failure in an error-reporting hook.
            if state.phase in self._ACTIVE_PHASES:
                try:
                    await self._fail_request(context)
                except Exception as notification_error:
                    error.add_note(f"Failure notification failed: {notification_error!r}")
            try:
                await self._notify_error(context, error)
            except Exception as notification_error:
                error.add_note(f"Error hook failed: {notification_error!r}")
            raise
        finally:
            self._request_active = False

    async def _execute_tools(
        self,
        context: AgentContext,
        calls: Sequence[ToolCall],
    ) -> AsyncIterator[AgentEvent]:
        for call in calls:
            await self._notify_before_tool(context, call)
            async with aclosing(self._before_tool_events(context, call)) as preprocessing:
                async for event in preprocessing:
                    yield event
            tool_started = await self._start_tool_execution(context)
            yield AgentEvent(
                AgentEventType.TOOL_STARTED,
                session_id=context.config.session_id,
                phase=context.state.phase,
                tool_calls=[call],
            )
            error = None
            try:
                registered = context.tools.get(call.name)
                if registered is None:
                    raise ValueError(f"Unknown tool: {call.name}")
                output = await registered(call.arguments)
                content = registered.serialize_result(output)
            except Exception as tool_error:
                error = tool_error
                content = json.dumps({"error": str(tool_error)})
            result = ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                success=error is None,
            )
            tool_completed = await self._finish_tool_execution(context)
            await context.append_message(
                result,
                MessageTiming(
                    started_at=tool_started.occurred_at,
                    completed_at=tool_completed.occurred_at,
                    duration_ns=max(0, tool_completed.monotonic_ns - tool_started.monotonic_ns),
                ),
            )
            await self._notify_after_tool(context, call, result)
            yield AgentEvent(
                AgentEventType.TOOL_FAILED if error else AgentEventType.TOOL_COMPLETED,
                session_id=context.config.session_id,
                phase=context.state.phase,
                tool_calls=[call],
                message=result,
                error=error,
            )

    async def _apply_extension_event_phase(self, context: AgentContext, event: AgentEvent) -> None:
        """Validate and apply phase transitions represented by extension events."""
        match event.type:
            case AgentEventType.COMPACTION_STARTED:
                await self._start_compaction(context)
            case AgentEventType.COMPACTION_TEXT_DELTA | AgentEventType.COMPACTION_REASONING_DELTA:
                self._require_phase(context.state, AgentPhase.COMPACTING)
            case AgentEventType.COMPACTION_COMPLETED:
                await self._finish_compaction(context)
            case AgentEventType.CUSTOM:
                pass
            case _:
                raise AgentProtocolError(f"Extension emitted unsupported pre-model event: {event.type.value!r}")
        event.phase = context.state.phase

    async def _before_tool_events(self, context: AgentContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Forward custom events and close each extension iterator on exit."""
        for extension in self.extensions:
            async with aclosing(extension.before_tool_events(context, call)) as events:
                async for event in events:
                    if event.type != AgentEventType.CUSTOM:
                        raise AgentProtocolError(f"Extension emitted unsupported pre-tool event: {event.type.value!r}")
                    event.phase = context.state.phase
                    yield event

    async def _notify_on_tool(self, context: AgentContext) -> None:
        """Let every extension register request-scoped tools in priority order."""
        for extension in self.extensions:
            await extension.on_tool(context)

    async def _notify_on_message(self, context: AgentContext) -> None:
        """Let every extension populate the request state in priority order."""
        for extension in self.extensions:
            await extension.on_message(context)

    async def _notify_before_run(self, context: AgentContext) -> None:
        for extension in self.extensions:
            await extension.before_run(context)

    async def _notify_before_model(self, context: AgentContext) -> None:
        for extension in self.extensions:
            await extension.before_model(context)

    async def _before_model_events(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        for extension in self.extensions:
            async with aclosing(extension.before_model_events(context)) as events:
                async for event in events:
                    yield event

    async def _notify_after_model(self, context: AgentContext, response: ModelResponse) -> None:
        for extension in self.extensions:
            await extension.after_model(context, response)

    async def _notify_before_tool(self, context: AgentContext, call: ToolCall) -> None:
        for extension in self.extensions:
            await extension.before_tool(context, call)

    async def _notify_after_tool(
        self,
        context: AgentContext,
        call: ToolCall,
        result: ToolMessage,
    ) -> None:
        for extension in self.extensions:
            await extension.after_tool(context, call, result)

    async def _notify_after_run(self, context: AgentContext, result: AssistantMessage) -> None:
        for extension in self.extensions:
            await extension.after_run(context, result)

    async def _notify_on_success(self, context: AgentContext, result: AssistantMessage) -> None:
        """Notify extensions after successful post-run processing."""
        for extension in self.extensions:
            await extension.on_success(context, result)

    async def _notify_error(self, context: AgentContext, error: Exception) -> None:
        for extension in self.extensions:
            await extension.on_error(context, error)
