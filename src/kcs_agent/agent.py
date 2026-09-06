import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Self, overload

from .events import AgentEvent, AgentEventType, AgentPhase, AgentPhaseTransitionMixin, ModelOutputTracker
from .exceptions import AgentIterationLimitError, AgentProtocolError
from .extension_events import ExtensionEvent, MessageAppendedEvent, MessageTiming
from .messages import AnyMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from .model import AgentModel, ModelEventType, ModelRequest, ModelResponse, ReasoningEffort
from .tools import AgentTool


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configuration that identifies one agent run."""

    session_id: str
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if self.request_id is not None and not self.request_id.strip():
            raise ValueError("request_id cannot be empty")


@dataclass(slots=True)
class AgentState:
    """Mutable conversation state created independently for one request."""

    messages: list[AnyMessage] = field(default_factory=list)
    phase: AgentPhase = AgentPhase.CREATED


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
    # Fixed subscriber order for this run; no event history is retained.
    extensions: tuple["AgentExtension", ...] = ()

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


class AgentExtension:
    """Observe or modify messages at each stage of the model-tool loop.

    Complete request state machine. The enclosing ACTIVE REQUEST box gives
    every active phase the same failure and cancellation exits. Bidirectional
    arrows enter an operation to the right and return to READY on completion::

                +-----------------+
                |     CREATED     |
                +-----------------+
                         |
        +----------------+-----------------------------------------------+
        | ACTIVE REQUEST v                                               |
        |       +-----------------+                                      |
        |       | LOADING_CONTEXT |                                      |
        |       +-----------------+                                      |
        |                | on_tool(): register request tools            |
        |                | on_message(): restore context                 |
        |                v                                               |
        |       +-----------------+ compact / done  +-----------------+  |
        |       |                 |<--------------->|   COMPACTING    |  |
        |       |                 |                 +-----------------+  |
        |       |                 |                                      |
        |       |                 | generate / done +-----------------+  |
        |       |      READY      |<--------------->|   GENERATING    |  |
        |       |                 |                 +-----------------+  |
        |       |                 |                                      |
        |       |                 | tool / done     +-----------------+  |
        |       |                 |<--------------->|  RUNNING_TOOL   |  |
        |       +-----------------+                 +-----------------+  |
        |                | final answer: after_run(), on_success()       |
        |                |                                               |
        +----------------+-----------------------------------+-----------+
                         v                                   | any active phase
                +-----------------+              +-----------+-----------+
                |    COMPLETED    |              | exception             | cancellation
                +-----------------+              v                       v
                                           +-----------+           +-------------+
                                           |  FAILED   |           |  CANCELLED  |
                                           +-----------+           +-------------+
                                           on_error()              RunCancelledEvent
                                                                  re-raise cancellation
                                           re-raise error

    READY runs before_run() and append_message(UserMessage) once, then
    before_model() and before_model_events() before each model call. Optional
    compaction streams started, reasoning/text deltas, and completed events.
    GENERATING streams model output; append_message(AssistantMessage) and
    after_model() run after returning to READY. Each tool runs before_tool(),
    enters RUNNING_TOOL, returns to READY, then appends its ToolMessage through
    append_message() and runs after_tool(). All tools finish before the loop
    calls the model again. after_run() and on_success() precede COMPLETED.
    Every phase change publishes PhaseTransitionEvent. Cancellation propagates
    CancelledError or GeneratorExit without invoking on_error().

    Internal message updates:
        - Every request starts with a new AgentState containing the system prompt.
        - on_tool builds the request-scoped tool registry before messages load.
        - on_message fills that state on every request before user input is appended.
        - The user message is appended after before_run and before before_model.
        - The complete assistant message is appended before after_model.
        - Each tool result is appended before after_tool.

    Hooks run in registration order and share the same mutable AgentState.
    Changes to state.messages are visible to subsequent hooks and model calls.
    before_tool and after_tool run for each requested tool; before_model runs
    again only after all tools in that model response have been processed.

    Uncaught agent, model, or hook exceptions trigger on_error before propagating
    to the caller. Tool execution errors become failed ToolMessages and follow
    after_tool instead. Task cancellation does not trigger on_error.
    """

    async def on_tool(self, context: AgentContext) -> None:
        """Register tools before any extension restores or injects messages.

        All extensions finish this hook before the first on_message() call, so
        prompt extensions can reliably inspect the complete request tool set.

        Example:
            async def on_tool(self, context):
                context.register_tool(read_file)
        """

    async def on_message(self, context: AgentContext) -> None:
        """Fill a fresh state before every request's user input is appended.

        Example:
            async def on_message(self, context):
                context.state.messages.insert(0, SystemMessage(content="Use concise answers."))
        """

    async def before_run(self, context: AgentContext) -> None:
        """Run before the new user message is appended."""

    async def before_model(self, context: AgentContext) -> None:
        """Run immediately before each model request is assembled."""

    async def before_model_events(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Stream extension-owned events before a primary model request.

        This hook is intended for visible preprocessing operations such as
        context compaction. Yield typed AgentEvents so Web and TUI clients can
        represent the operation as its own state instead of model generation.

        Example:
            yield AgentEvent(
                AgentEventType.CUSTOM,
                session_id=context.config.session_id,
                data={"name": "retrieval_progress", "completed": 3, "total": 10},
            )

        CUSTOM events are forwarded unchanged except for the runtime phase;
        they do not trigger a state transition.
        """
        if False:
            yield AgentEvent(AgentEventType.MODEL_STARTED, context.config.session_id)

    async def after_model(
        self,
        context: AgentContext,
        response: ModelResponse,
    ) -> None:
        """Run after the complete assistant message is appended."""

    async def before_tool(self, context: AgentContext, call: ToolCall) -> None:
        """Run before one requested tool is invoked."""

    async def before_tool_events(self, context: AgentContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after before_tool and before each tool starts.

        Hooks run in registration order while the request remains READY.
        Hook failures abort the request before invoking the tool. Closing the
        consumer closes this iterator so its finally blocks can release resources.

        Example:
            yield AgentEvent(
                AgentEventType.CUSTOM,
                session_id=context.config.session_id,
                data={"name": "tool_preparation", "tool_call_id": call.id},
            )
        """
        if False:
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id)

    async def after_tool(
        self,
        context: AgentContext,
        call: ToolCall,
        result: ToolMessage,
    ) -> None:
        """Run after one tool result is appended, including failed results."""

    async def after_run(
        self,
        context: AgentContext,
        result: AssistantMessage,
    ) -> None:
        """Run after a successful final answer is appended."""

    async def on_success(self, context: AgentContext, result: AssistantMessage) -> None:
        """Run once after all after_run hooks succeed, before RUN_COMPLETED.

        This callback observes a completed request, not an intermediate model
        step. Failed or cancelled requests do not trigger it. Callback errors
        propagate through on_error and prevent RUN_COMPLETED from being emitted.

        Example:
            async def on_success(self, context, result):
                await save_answer(context.config.session_id, result.content)
        """

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Process a published notification; inspect its concrete type with match."""

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Run before an agent error is propagated to the caller."""


class Agent(AgentPhaseTransitionMixin):
    """A small stateful model/tool loop with optional lifecycle extensions."""

    def __init__(
        self,
        model: AgentModel,
        *,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
        max_iterations: int = 36,
    ) -> None:
        """Create an agent with its own default extensions.

        Omit extensions to enable InMemoryMessageAccumulator and
        ToolGuidelinesExtension. An explicit sequence replaces those defaults;
        pass an empty sequence to disable all extensions.

        Example:
            agent = await Agent.create(model, config=config, tools=[read_file])
        """
        from .extensions import InMemoryMessageAccumulator, ToolGuidelinesExtension

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self.extensions = (
            (InMemoryMessageAccumulator(), ToolGuidelinesExtension()) if extensions is None else tuple(extensions)
        )
        self.system_prompt = system_prompt
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

    @classmethod
    async def create(
        cls,
        model: AgentModel,
        *,
        config: AgentConfig,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
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
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AssistantMessage: ...

    @overload
    async def run(
        self,
        message: str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AssistantMessage: ...

    async def run(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AssistantMessage:
        """Collect one run and return its final answer.

        Example:
            agent = await Agent.create(model, config=AgentConfig(session_id="session-42"))
            reply = await agent.run(
                "Summarize this conversation.",
                config=AgentConfig(session_id="session-42"),
            )
        """
        result: AssistantMessage | None = None
        async with aclosing(self.stream(message, config=config, reasoning_effort=reasoning_effort)) as events:
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
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AsyncIterator[AgentEvent]: ...

    @overload
    def stream(
        self,
        message: str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AsyncIterator[AgentEvent]: ...

    async def stream(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    ) -> AsyncIterator[AgentEvent]:
        """Stream one run while retaining messages in the agent state.

        Extensions receive the same mutable state and may add, remove, or replace
        messages at each lifecycle hook.

        Example:
            agent = await Agent.create(model, config=config, extensions=[history_extension])
            async for event in agent.stream(
                UserMessage(content="Continue the summary."),
                config=AgentConfig(session_id="session-42"),
                reasoning_effort=ReasoningEffort.HIGH,
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
        messages = [SystemMessage(content=self.system_prompt)] if self.system_prompt else []
        # State and dynamic tools are request-scoped. Constructor tools are copied
        # so extension registrations cannot leak into later requests or sessions.
        state = AgentState(messages=messages)
        self.state = state
        context = AgentContext(config=config, state=state, tools=dict(self.tools), extensions=self.extensions)
        user_message = UserMessage(content=message) if isinstance(message, str) else message
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
                    reasoning_effort=reasoning_effort,
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
                call=call,
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
                call=call,
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
        """Let every extension register request-scoped tools in order."""
        for extension in self.extensions:
            await extension.on_tool(context)

    async def _notify_on_message(self, context: AgentContext) -> None:
        """Let every extension populate the fresh request state in order."""
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
