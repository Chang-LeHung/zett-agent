from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import RLock
from time import monotonic_ns
from typing import TYPE_CHECKING, Self, overload

from .events import AgentEvent, AgentEventType, AgentPhase, AgentPhaseTransitionMixin, ModelOutputTracker
from .exceptions import AgentIterationLimitError, AgentProtocolError
from .json_types import JsonValue, json_object
from .messages import AgentMessage, AnyMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from .model import (
    AgentModel,
    ModelEventType,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ReasoningEffort,
    ServerToolDefinition,
)
from .sync_runtime import SyncMethodsMixin
from .tools import AgentTool, ToolExecutionMode
from .tools.base import _bind_tool_call, _reset_tool_call

if TYPE_CHECKING:
    from .extensions.base import AgentExtension
    from .extensions.events import ExtensionEvent, MessageTiming
    from .extensions.external import ExternalEvent


@dataclass(frozen=True, slots=True)
class _ToolInvocation:
    """Completed handler output retained until ordered lifecycle finalization."""

    call: ToolCall
    result: ToolMessage
    error: Exception | None
    timing: MessageTiming


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Identify a conversation and optionally one request within it.

    Attributes:
        session_id: Stable non-empty conversation identifier, reused across turns.
        request_id: Optional application correlation ID for this invocation.
            It is not an idempotency key; reusing it does not deduplicate messages.
        parent_session_id: Parent conversation when running a delegated child.
            Must differ from session_id.

    Raises:
        ValueError: If an identity is empty or a session references itself.

    Examples:
        A root conversation needs only a session ID::

            config = AgentConfig(session_id="chat-42")

        A delegated agent identifies its parent separately::

            child = AgentConfig(
                session_id="review-42",
                request_id="request-7",
                parent_session_id="chat-42",
            )

    .. note::
        ``request_id`` is correlation data, not an idempotency key. Reusing it
        does not suppress a second message or model call.

    .. seealso::
        :class:`~zett_agent.SQLiteSessionStorage` for persisted session identity,
        and :class:`~zett_agent.SubAgentDefinition` for child-session setup.
    """

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
    """Mutable conversation state created independently for one request.

    Extensions populate ``messages`` during :meth:`AgentSetupHooksMixin.on_state`.
    The runtime owns ``phase`` and validates every transition; extensions should
    observe phase events instead of assigning it directly.

    .. zett-diagram:: agent-state

        +---------------------+
        | CREATED             |
        +---------------------+
                   |
                   v
        +---------------------+
        | LOADING_CONTEXT     |
        +---------------------+
                   |
                   v
        +---------------------+
        | READY               |<------------+
        +---------------------+             |
                   |                        |
                   v                        |
        +---------------------+             |
        | GENERATING          |-------------+
        +---------------------+

    A later request for the same session receives a fresh ``AgentState``. History
    survives only when a memory or persistence extension restores it.

    .. seealso::
        :class:`~zett_agent.AgentPhase` documents the state machine, while
        :class:`~zett_agent.SessionView` describes restored persistent context.
    """

    messages: list[AnyMessage] = field(default_factory=list)
    phase: AgentPhase = AgentPhase.CREATED
    # Parent conversation when this request belongs to a delegated subagent.
    parent_session_id: str | None = None


@dataclass(slots=True, weakref_slot=True, eq=False)
class AgentContext:
    """Per-run references shared by all lifecycle hooks.

    Register request-scoped local and server tools during on_tool(). Mutate
    state.messages and either tool registry in place to update the current
    request without leaking registrations into another request or session.

    .. note::
        Context identity is request-scoped. It is safe to use a context as a key
        for temporary extension state only when every success, error, and
        cancellation path removes that entry.

    Examples:
        Register a tool for only the current request::

            async def on_tool(self, context: AgentContext) -> None:
                context.register_tool(read_file)

        Register a provider-hosted tool without adding a local executor::

            async def on_tool(self, context: AgentContext) -> None:
                context.register_server_tool(
                    ServerToolDefinition(
                        type="web_search",
                        configuration={"search_context_size": "medium"},
                    )
                )

        Publish an internal notification to every extension::

            await context.publish(MyIndexReadyEvent(document_count=12))

        Or deliver it to one named extension::

            await context.publish(event, target="SearchIndexExtension")

    .. seealso::
        :class:`~zett_agent.AgentExtension` for lifecycle hooks and
        :class:`~zett_agent.AgentTool` for request-scoped tool registration.
    """

    # Configuration for this invocation.
    config: AgentConfig
    # Conversation state populated for the current request.
    state: AgentState
    # Live registry used for model schemas and tool execution.
    tools: dict[str, AgentTool]
    # Fixed priority order for this run; no event history is retained.
    extensions: tuple[AgentExtension, ...] = ()
    # Model resolved for this request from its explicit override or Agent default;
    # setup extensions may inspect it when creating request-scoped capabilities.
    model: AgentModel | None = None
    # Application input available to extensions and Raw Log persistence only.
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    # Request classifications available to extensions and Raw Log persistence.
    tags: dict[str, JsonValue] = field(default_factory=dict)
    # New input before it is appended. Setup hooks may replace this message to
    # implement explicit command modes while preserving its raw form in attributes.
    input_message: UserMessage | None = None
    # Provider-hosted tools registered for this request, keyed by wire type.
    # They are sent to the model but never dispatched by the local tool loop.
    server_tools: dict[str, ServerToolDefinition] = field(default_factory=dict)

    def register_tool(self, tool: AgentTool) -> None:
        """Register one request-scoped tool while rejecting ambiguous names."""
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        # Tool metadata is mutable; a copied registry alone would still share
        # schemas and descriptions between sessions. Keep the callable itself.
        self.tools[tool.name] = replace(tool, parameters=deepcopy(tool.parameters))

    def register_server_tool(self, tool: ServerToolDefinition) -> None:
        """Register one request-scoped provider tool by its unique wire type.

        The opaque configuration is copied because extensions may reuse one
        definition across concurrent sessions. Server tools are model-side
        capabilities and therefore never appear in :attr:`tools`, which is the
        registry used by the Agent's local executor.

        Raises:
            ValueError: If this request already contains the same server-tool
                type. Registering two configurations under one discriminator
                would make provider behavior order-dependent.
        """
        if tool.type in self.server_tools:
            raise ValueError(f"Server tool {tool.type!r} is already registered")
        self.server_tools[tool.type] = replace(tool, configuration=deepcopy(tool.configuration))

    async def publish(self, event: ExtensionEvent, *, target: str | None = None) -> None:
        """Deliver an event to one named extension or broadcast it by default.

        A target must match exactly one registered extension name. Unknown or
        ambiguous targets raise ValueError without delivering the event. During
        broadcast, handler errors propagate immediately and stop delivery.
        Already completed work is not rolled back. Subscribers may retain events.

        Examples:
            Usage::

                await context.publish(CompactionEvent(
                    compressed_from=1, compressed_to=20,
                    kept_from=21, kept_to=30, summary="Earlier decisions...",
                ))

                await context.publish(event, target="MetricsExtension")
        """
        extensions = self.extensions
        if target is not None:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Extension event target must be a non-empty string")
            extensions = tuple(extension for extension in extensions if extension.name == target)
            if not extensions:
                raise ValueError(f"Unknown extension event target: {target!r}")
            if len(extensions) > 1:
                raise ValueError(f"Ambiguous extension event target: {target!r}")
        for extension in extensions:
            await extension.on_event(self, event)

    async def append_message(
        self,
        message: AnyMessage,
        timing: MessageTiming,
        usage: ModelUsage | None = None,
    ) -> None:
        """Append one newly produced message and publish its Raw Log event.

        Runtime code must use this method for user, assistant, and tool messages
        so in-memory context and persistence notifications cannot drift apart.
        Restored history and generated system instructions are existing context,
        not new Raw Log messages, and therefore do not use this method.

        The message is visible in ``state.messages`` before subscribers run.
        Subscriber failures propagate and do not roll back the in-memory append.
        """
        from .extensions.events import MessageAppendedEvent

        self.state.messages.append(message)
        await self.publish(MessageAppendedEvent(message, timing, usage))


class Agent(AgentPhaseTransitionMixin, SyncMethodsMixin):
    """Run a model, execute requested tools, and repeat until an answer is ready.

    ``Agent`` is the low-level runtime. Most applications can use
    :func:`create_agent`, which initializes it and optionally dispatches events.
    Direct construction is useful when initialization must happen later.

    .. zett-diagram:: agent-loop

        +----------------------+
        | UserMessage          |
        +----------------------+
                   |
                   v
        +----------------------+                     +----------------+
        | model                |----- ToolCall ----->| tool           |
        |                      |<--- ToolMessage ----|                |
        +----------------------+                     +----------------+
                   |
                   v
        +----------------------+
        | AssistantMessage     |
        +----------------------+

    Tools are parallel by default. When parallel calling is enabled, parallel
    tools from one model response run together before explicitly serial tools.
    Hooks, Raw Log appends, and completion events remain ordered. Streaming
    deltas are display data; only the final :class:`ModelResponse` is appended
    as an assistant message.

    Examples:
        Prefer the asynchronous factory when using the runtime directly::

            agent = await Agent.create(
                model,
                config=AgentConfig(session_id="chat-42"),
                tools=[read_file, grep],
                reasoning_effort=ReasoningEffort.HIGH,
            )
            answer = await agent.run("Find where retries are configured")
            print(answer.content)

        Stream visible events and always close an abandoned iterator::

            from contextlib import aclosing

            async with aclosing(agent.stream("Inspect the tests")) as events:
                async for event in events:
                    if event.type is AgentEventType.TEXT_DELTA:
                        print(event.delta, end="", flush=True)

    .. warning::
        An ``Agent`` must be initialized before ``run`` or ``stream``. Use
        :meth:`create` or await :meth:`initialize` after direct construction.

    .. seealso::
        :func:`~zett_agent.create_agent` is the convenient initialized entry
        point. :class:`~zett_agent.AgentClient` adds application event dispatch,
        and :doc:`/concepts/lifecycle` explains the complete loop.
    """

    def __init__(
        self,
        model: AgentModel | None,
        *,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        parallel_tool_call: bool = True,
        max_iterations: int = 36,
        max_internal_messages: int = 8,
    ) -> None:
        """Create an agent with its own default extensions.

        Omit extensions to enable InMemoryMessageAccumulator and
        ToolGuidelinesExtension. An explicit sequence replaces those defaults.
        InternalMessageExtension and SteeringExtension are Agent-owned built-ins
        and remain enabled even when an empty extension sequence is supplied.
        Steering checks run after every completed model/tool operation, before
        internal input or remaining tool calls. It never interrupts in-flight work.

        Extension names must be unique, including built-ins. max_iterations
        limits model calls for each UserMessage or AgentMessage;
        tool round trips consume that message's budget. max_internal_messages
        separately limits internal inputs per run/stream. Zero disables internal
        processing. Exceeding either limit raises
        AgentIterationLimitError; pending messages are discarded on failure.

        Args:
            model: Default provider adapter. It may be None when every request
                supplies an explicit model to run() or stream().
            system_prompt: Initial instructions rebuilt for each request.
            tools: Static tool definitions with unique names.
            extensions: Unique named extensions sorted by ascending priority.
                None installs defaults; an empty sequence omits optional defaults.
            reasoning_effort: Default provider-neutral reasoning level.
            parallel_tool_call: Allow providers to return multiple tool calls
                and concurrently execute tools not explicitly marked serial.
            max_iterations: Model-call budget per user/internal message.
            max_internal_messages: Internal input budget per request; zero disables it.

        Note:
            Construction alone does not initialize the runtime. Await
            initialize() or use the asynchronous create() factory before running.

        Examples:
            Usage::

                agent = await Agent.create(
                    model,
                    config=config,
                    tools=[read_file],
                    reasoning_effort=ReasoningEffort.HIGH,
                )
        """
        from .extensions.internal_message import InternalMessageExtension
        from .extensions.memory import InMemoryMessageAccumulator
        from .extensions.steering import SteeringExtension
        from .extensions.tool_guidelines import ToolGuidelinesExtension

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not isinstance(parallel_tool_call, bool):
            raise ValueError("parallel_tool_call must be a boolean")
        if isinstance(max_internal_messages, bool) or not isinstance(max_internal_messages, int):
            raise ValueError("max_internal_messages must be a non-negative integer")
        if max_internal_messages < 0:
            raise ValueError("max_internal_messages must be a non-negative integer")
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        configured_extensions = list(
            (InMemoryMessageAccumulator(), ToolGuidelinesExtension()) if extensions is None else tuple(extensions)
        )
        if any(isinstance(extension, InternalMessageExtension) for extension in configured_extensions):
            raise ValueError("InternalMessageExtension is an Agent built-in and cannot be supplied in extensions")
        self._internal_message_extension = InternalMessageExtension()
        configured_extensions.append(self._internal_message_extension)
        if any(isinstance(extension, SteeringExtension) for extension in configured_extensions):
            raise ValueError("SteeringExtension is an Agent built-in and cannot be supplied in extensions")
        self._steering_extension = SteeringExtension()
        configured_extensions.append(self._steering_extension)
        names: set[str] = set()
        for extension in configured_extensions:
            name = extension.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Extension name must be a non-empty string")
            if name in names:
                raise ValueError(f"Duplicate extension name: {name!r}")
            names.add(name)
        # Python's sort is stable, so extensions sharing a priority preserve the
        # caller's registration order. Every lifecycle path uses this tuple.
        self.extensions = tuple(sorted(configured_extensions, key=lambda extension: extension.priority))
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.parallel_tool_call = parallel_tool_call
        self.state = AgentState()
        # Runtime data is session-scoped. state remains the most recently
        # started state for single-session callers; use get_state for routing.
        self._states: dict[str, AgentState] = {}
        self._active_configs: dict[str, AgentConfig] = {}
        self._sessions_lock = RLock()
        self._initialized_config: AgentConfig | None = None
        self.max_iterations = max_iterations
        self.max_internal_messages = max_internal_messages

    async def initialize(self, *, config: AgentConfig) -> None:
        """Bind the default session used when a request omits config.

        Examples:
            Usage::

                agent = Agent(model)
                await agent.initialize(config=AgentConfig(session_id="session-42"))
                reply = await agent.run("Hello")
        """
        if self._initialized_config is not None:
            if self._initialized_config.session_id != config.session_id:
                raise AgentProtocolError("Agent is already initialized for another session")
            return
        self._initialized_config = config

    def get_state(self, session_id: str) -> AgentState | None:
        """Return one session's latest state, or None before its first request.

        Concurrent callers must select by session rather than reading state,
        which is only the most recently started request's convenience view.
        """
        with self._sessions_lock:
            return self._states.get(session_id)

    def emit_external_event(self, event: ExternalEvent, *, config: AgentConfig | None = None) -> list[str]:
        """Broadcast config and event separately; return accepting extension names.

        Without explicit config, use the sole active request, then the initialized
        session, or None before initialization. Extensions own routing and
        acceptance decisions. The original event is forwarded without copying,
        interpreting, or modifying its payload.

        Every registered extension receives the event in priority order,
        including extensions after the first one that accepts it. This keeps the
        caller independent from the extension that owns a protocol.

        Examples:
            Usage::

                accepted = agent.emit_external_event(
                    ExternalEvent(
                        name="ask_user_response",
                        payload={"tool_call_id": "call-1", "answer": "Yes"},
                    ),
                    config=AgentConfig(session_id="session-42", request_id="request-1"),
                )
        """
        with self._sessions_lock:
            if config is None and len(self._active_configs) > 1:
                raise AgentProtocolError("External events require config when multiple sessions are active")
            routing = (
                config if config is not None else next(iter(self._active_configs.values()), self._initialized_config)
            )
        accepted: list[str] = []
        for extension in self.extensions:
            if extension.accept(routing, event):
                accepted.append(extension.name)
        return accepted

    @classmethod
    async def create(
        cls,
        model: AgentModel | None,
        *,
        config: AgentConfig,
        system_prompt: str = "You are a helpful assistant.",
        tools: Sequence[AgentTool] = (),
        extensions: Sequence[AgentExtension] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        parallel_tool_call: bool = True,
        max_iterations: int = 36,
        max_internal_messages: int = 8,
    ) -> Self:
        """Construct and initialize an Agent before returning it.

        Args:
            model: Provider-neutral streaming model adapter.
            config: Initial session identity used when requests omit config.
            system_prompt: Initial instructions for every fresh request state.
            tools: Static tools registered before extension setup.
            extensions: Optional lifecycle extensions; None keeps defaults.
            reasoning_effort: Default reasoning level for future requests.
            parallel_tool_call: Enable provider and local parallel tool calls.
            max_iterations: Maximum model calls per user/internal input.
            max_internal_messages: Maximum internal continuations per request.

        Returns:
            An initialized instance of the class on which create() was called.

        Examples:
            Usage::

                agent = await Agent.create(model, config=AgentConfig(session_id="session-42"))
                reply = await agent.run("Hello")
        """
        agent = cls(
            model,
            system_prompt=system_prompt,
            tools=tools,
            extensions=extensions,
            reasoning_effort=reasoning_effort,
            parallel_tool_call=parallel_tool_call,
            max_iterations=max_iterations,
            max_internal_messages=max_internal_messages,
        )
        await agent.initialize(config=config)
        return agent

    @overload
    async def run(
        self,
        message: UserMessage,
        *,
        config: AgentConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage: ...

    @overload
    async def run(
        self,
        message: str,
        *,
        config: AgentConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage: ...

    async def run(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage:
        """Collect one run and return its final answer.

        Omit reasoning_effort to use the default configured on this Agent.

        Args:
            message: User text or a typed message containing text/image blocks.
            config: Request identity; omitted to reuse the initialized session.
            model: Request-specific model; omitted to use the Agent default.
            reasoning_effort: Optional reasoning override for this request.
            parallel_tool_call: Optional parallel-call override for this request.
            metadata: JSON data for extensions and Raw Log persistence, not a prompt.
            tags: JSON classifications for extensions and Raw Log persistence.

        Returns:
            The final assistant answer, after all stream cleanup completes.

        Raises:
            AgentProtocolError: If uninitialized or no terminal answer is produced.
            AgentIterationLimitError: If the model/internal-message budget is exhausted.

        Examples:
            Usage::

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
                model=model,
                reasoning_effort=reasoning_effort,
                parallel_tool_call=parallel_tool_call,
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
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    @overload
    def stream(
        self,
        message: str,
        *,
        config: AgentConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def stream(
        self,
        message: UserMessage | str,
        *,
        config: AgentConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream one run while retaining messages in the agent state.

        Extensions receive the same mutable state and may add, remove, or replace
        messages at each lifecycle hook. Omit reasoning_effort to use the Agent's
        configured default; an explicit value overrides it for this request only.

        Args:
            message: Plain user text or a typed multimodal UserMessage.
            config: Session/request identity; defaults to the initialized configuration.
            model: Request-specific model; omitted to use the Agent default.
            reasoning_effort: Reasoning override limited to this request.
            parallel_tool_call: Parallel-call override limited to this request.
            metadata: Request data visible to extensions and persistence.
            tags: Request classifications visible to extensions and persistence.

        Yields:
            AgentEvent objects in execution order, ending in RUN_COMPLETED on success.

        Note:
            Model errors and cancellation propagate as exceptions. Use
            contextlib.aclosing when stopping iteration early. Concurrent requests
            must use different session IDs; same-session overlap is rejected.

        Examples:
            Usage::

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
        user_message = message if isinstance(message, UserMessage) else UserMessage(content=message)
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        allow_parallel = self.parallel_tool_call if parallel_tool_call is None else parallel_tool_call
        if not isinstance(allow_parallel, bool):
            raise ValueError("parallel_tool_call must be a boolean")
        context = self._prepare_request_context(config, model, metadata, tags, user_message)
        try:
            await self._open_request(context)
            async with aclosing(self._stream_loop(context, effort, allow_parallel)) as events:
                async for event in events:
                    yield event
        except (asyncio.CancelledError, GeneratorExit) as cancellation:
            await self._handle_request_cancellation(context, cancellation)
            raise
        except Exception as error:
            await self._handle_request_failure(context, error)
            raise
        finally:
            self._release_request(context)

    def _prepare_request_context(
        self,
        config: AgentConfig | None,
        model: AgentModel | None,
        metadata: Mapping[str, JsonValue] | None,
        tags: Mapping[str, JsonValue] | None,
        input_message: UserMessage,
    ) -> AgentContext:
        """Validate input, create isolated state/tools, and claim the session."""
        if self._initialized_config is None:
            raise AgentProtocolError(
                "Agent is not initialized; await agent.initialize(config=...) or Agent.create(...)"
            )
        config = config or self._initialized_config
        resolved_model = model if model is not None else self.model
        if resolved_model is None:
            raise AgentProtocolError("This request requires a model because the Agent has no default model")
        messages = [SystemMessage(content=self.system_prompt)] if self.system_prompt else []
        # State and dynamic tools are request-scoped. Constructor tools are copied
        # so extension registrations cannot leak into later requests or sessions.
        state = AgentState(
            messages=messages,
            parent_session_id=config.parent_session_id,
        )
        context = AgentContext(
            config=config,
            state=state,
            tools={},
            extensions=self.extensions,
            model=resolved_model,
            metadata=json_object(metadata, field_name="Context metadata"),
            tags=json_object(tags, field_name="Context tags", nonempty_keys=True),
            input_message=input_message,
        )
        for registered in self.tools.values():
            context.register_tool(registered)
        with self._sessions_lock:
            previous = self._states.get(config.session_id)
            if config.session_id in self._active_configs or (previous and not previous.phase.accepts_new_request):
                phase = previous.phase.value if previous is not None else "active"
                raise AgentProtocolError(f"Agent cannot start a new request while its current phase is {phase!r}")
            self._active_configs[config.session_id] = config
            self._states[config.session_id] = state
            self.state = state
        return context

    async def _open_request(self, context: AgentContext) -> None:
        """Open inboxes, restore history through hooks, then append the new input."""
        from .extensions.events import MessageTiming

        self._internal_message_extension.open(context)
        self._steering_extension.open(context)
        await self._start_context_loading(context)
        await self._notify_on_tool(context)
        await self._notify_on_state(context)
        await self._notify_on_message(context)
        await self._finish_context_loading(context)
        if not isinstance(context.input_message, UserMessage):
            raise AgentProtocolError("Request setup must produce a UserMessage")
        await context.append_message(context.input_message, MessageTiming.instant())
        await self._notify_before_run(context)

    async def _stream_loop(
        self,
        context: AgentContext,
        reasoning_effort: ReasoningEffort,
        parallel_tool_call: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Alternate model and tool steps, processing queued inputs before completion."""
        from .extensions.events import MessageTiming

        config, state = context.config, context.state
        active_input: AgentMessage | UserMessage | None = None
        message_iterations = 0
        processed_internal_messages = 0
        while True:
            if message_iterations >= self.max_iterations:
                raise AgentIterationLimitError(f"Agent exceeded {self.max_iterations} model iterations")
            message_iterations += 1
            response = None
            async with aclosing(self._stream_model_step(context, reasoning_effort, parallel_tool_call)) as model_events:
                async for event in model_events:
                    if event.type == AgentEventType.MODEL_COMPLETED:
                        response = event.response
                    yield event
            if response is None:
                raise AgentProtocolError("Model step ended without a completed response")
            if not response.message.tool_calls:
                if active_input is not None:
                    internal = isinstance(active_input, AgentMessage)
                    yield AgentEvent(
                        AgentEventType.INTERNAL_MESSAGE_COMPLETED if internal else AgentEventType.STEERING_COMPLETED,
                        session_id=config.session_id,
                        phase=state.phase,
                        message=response.message,
                        internal_message=active_input if internal else None,
                        steering_message=active_input if isinstance(active_input, UserMessage) else None,
                    )
                    active_input = None

                steering, next_input = self._select_pending_input(context)
                if steering is not None:
                    async with aclosing(self._start_steering(context, steering)) as steering_events:
                        async for event in steering_events:
                            yield event
                    active_input = steering
                    message_iterations = 0
                    continue
                if next_input is not None:
                    if processed_internal_messages >= self.max_internal_messages:
                        raise AgentIterationLimitError(
                            f"Agent exceeded {self.max_internal_messages} internal messages in one request"
                        )
                    processed_internal_messages += 1
                    message_iterations = 0
                    yield AgentEvent(
                        AgentEventType.INTERNAL_MESSAGE_STARTED,
                        session_id=config.session_id,
                        phase=state.phase,
                        internal_message=next_input,
                    )
                    await context.append_message(next_input, MessageTiming.instant())
                    active_input = next_input
                    continue

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

            steering = self._steering_extension.take(context)
            if steering is not None:
                async with aclosing(
                    self._start_steering(context, steering, response.message.tool_calls, active_input)
                ) as steering_events:
                    async for event in steering_events:
                        yield event
                active_input = steering
                message_iterations = 0
                continue

            async with aclosing(
                self._execute_tools(context, response.message.tool_calls, parallel_tool_call, active_input)
            ) as tool_events:
                async for event in tool_events:
                    if event.type == AgentEventType.STEERING_STARTED:
                        active_input = event.steering_message
                        message_iterations = 0
                    yield event

    async def _stream_model_step(
        self,
        context: AgentContext,
        reasoning_effort: ReasoningEffort,
        parallel_tool_call: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Run preprocessing and one model call, validate its stream, and persist output."""
        config, state = context.config, context.state
        request = ModelRequest(
            messages=tuple(state.messages),
            tools=tuple(tool.definition for tool in context.tools.values()),
            server_tools=tuple(context.server_tools.values()),
            reasoning_effort=reasoning_effort,
            parallel_tool_call=parallel_tool_call,
        )
        await self._notify_before_model(context, request)

        async with aclosing(self._before_model_events(context, request)) as preprocessing:
            async for event in preprocessing:
                await self._apply_extension_event_phase(context, event)
                yield event

        request = self._refresh_model_request(context, request)
        model_started = await self._start_model_generation(context)
        output_tracker = ModelOutputTracker()
        yield AgentEvent(
            AgentEventType.MODEL_STARTED,
            session_id=config.session_id,
            phase=state.phase,
        )
        response = None
        model = context.model
        if model is None:
            raise AgentProtocolError("The request context has no model")
        async with aclosing(model.stream(request)) as events:
            async for event in events:
                if response is not None:
                    raise AgentProtocolError("Model emitted events after its final response")
                for boundary in await output_tracker.observe(context, event):
                    yield AgentEvent(boundary, session_id=config.session_id, phase=state.phase)
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
                    case ModelEventType.SERVER_TOOL_STARTED:
                        if event.server_tool_call is None:
                            raise AgentProtocolError("Missing server-tool call")
                        yield AgentEvent(
                            AgentEventType.SERVER_TOOL_STARTED,
                            session_id=config.session_id,
                            phase=state.phase,
                            server_tool_call=event.server_tool_call,
                        )
                    case ModelEventType.SERVER_TOOL_INPUT_DELTA:
                        if event.server_tool_input_delta is None:
                            raise AgentProtocolError("Missing server-tool input delta")
                        yield AgentEvent(
                            AgentEventType.SERVER_TOOL_INPUT_DELTA,
                            session_id=config.session_id,
                            phase=state.phase,
                            server_tool_input_delta=event.server_tool_input_delta,
                        )
                    case ModelEventType.SERVER_TOOL_COMPLETED | ModelEventType.SERVER_TOOL_FAILED:
                        if event.server_tool_result is None:
                            raise AgentProtocolError("Missing server-tool result")
                        yield AgentEvent(
                            AgentEventType(event.type.value),
                            session_id=config.session_id,
                            phase=state.phase,
                            server_tool_result=event.server_tool_result,
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
            response.usage,
        )
        await self._notify_after_model(context, response)
        yield AgentEvent(
            AgentEventType.MODEL_COMPLETED,
            session_id=config.session_id,
            phase=state.phase,
            response=response,
        )

        async with aclosing(self._after_model_events(context, response)) as events:
            async for event in events:
                yield event

    def _select_pending_input(self, context: AgentContext) -> tuple[UserMessage | None, AgentMessage | None]:
        """Reserve steering first, or internal input, without a queue-closing race."""
        internal: AgentMessage | None = None

        def reserve_internal_input() -> bool:
            nonlocal internal
            internal = self._internal_message_extension.next_message(context)
            return internal is not None

        steering = self._steering_extension.select_next(context, reserve_internal_input)
        return steering, internal

    async def _handle_request_cancellation(self, context: AgentContext, cancellation: BaseException) -> None:
        """Close interrupted tool calls, then publish the request cancellation.

        A completed assistant tool-call message is persisted before execution
        starts. If cancellation interrupts a tool, leaving that message without
        one ToolMessage per call makes the next provider request invalid. Write
        explicit failed observations while persistence bookkeeping is still
        active so the restored conversation remains protocol-complete.
        """
        try:
            await self._cancel_unanswered_tool_calls(context)
            await self._cancel_request(context)
        except Exception as notification_error:
            cancellation.add_note(f"Cancellation notification failed: {notification_error!r}")

    async def _cancel_unanswered_tool_calls(self, context: AgentContext) -> None:
        """Append cancellation results for the latest unfinished tool-call batch."""
        from .extensions.events import MessageTiming

        pending: dict[str, ToolCall] = {}
        for message in context.state.messages:
            match message:
                case AssistantMessage(tool_calls=calls) if calls:
                    pending = {call.id: call for call in calls}
                case ToolMessage(tool_call_id=tool_call_id) if pending:
                    pending.pop(tool_call_id, None)
                case _:
                    # A valid tool batch is contiguous. Another message means
                    # this is older malformed history, not the operation that
                    # the current cancellation interrupted.
                    if pending:
                        pending.clear()

        for call in pending.values():
            await context.append_message(
                ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=json.dumps({"cancelled": True, "reason": "Request cancelled before tool completion"}),
                    success=False,
                ),
                MessageTiming.instant(),
            )

    async def _handle_request_failure(self, context: AgentContext, error: Exception) -> None:
        """Report failure without masking the original error or terminal phase."""
        # A terminal transition is already committed before subscribers run.
        # Never replace the original error with an illegal terminal transition
        # or a secondary failure in an error-reporting hook.
        if context.state.phase in self._ACTIVE_PHASES:
            try:
                await self._fail_request(context)
            except Exception as notification_error:
                error.add_note(f"Failure notification failed: {notification_error!r}")
        try:
            await self._notify_error(context, error)
        except Exception as notification_error:
            error.add_note(f"Error hook failed: {notification_error!r}")

    def _release_request(self, context: AgentContext) -> None:
        """Clear only this request's inboxes and release its session for reuse."""
        self._steering_extension.close(context)
        self._internal_message_extension.close(context)
        with self._sessions_lock:
            self._active_configs.pop(context.config.session_id, None)

    async def _execute_tools(
        self,
        context: AgentContext,
        calls: Sequence[ToolCall],
        parallel_tool_call: bool,
        active_input: AgentMessage | UserMessage | None = None,
    ) -> AsyncIterator[AgentEvent]:
        parallel_calls: list[ToolCall] = []
        serial_calls: list[ToolCall] = []
        for call in calls:
            registered = context.tools.get(call.name)
            if (
                parallel_tool_call
                and registered is not None
                and registered.execution_mode is ToolExecutionMode.PARALLEL
            ):
                parallel_calls.append(call)
            else:
                serial_calls.append(call)

        if parallel_calls:
            async with aclosing(self._execute_parallel_tools(context, parallel_calls)) as events:
                async for event in events:
                    yield event
            steering = self._steering_extension.take(context)
            if steering is not None:
                async with aclosing(
                    self._start_steering(context, steering, serial_calls, active_input)
                ) as steering_events:
                    async for event in steering_events:
                        yield event
                return

        async with aclosing(self._execute_serial_tools(context, serial_calls, active_input)) as events:
            async for event in events:
                yield event

    async def _execute_parallel_tools(
        self,
        context: AgentContext,
        calls: Sequence[ToolCall],
    ) -> AsyncIterator[AgentEvent]:
        """Run one parallel batch and stream each result as its handler finishes."""
        for call in calls:
            await self._notify_before_tool(context, call)
            async with aclosing(self._before_tool_events(context, call)) as preprocessing:
                async for event in preprocessing:
                    yield event

        await self._start_tool_execution(context)
        yield AgentEvent(
            AgentEventType.TOOL_STARTED,
            session_id=context.config.session_id,
            phase=context.state.phase,
            tool_calls=list(calls),
        )
        tasks = [asyncio.create_task(self._invoke_tool(context, call)) for call in calls]
        remaining = len(tasks)
        try:
            for completed in asyncio.as_completed(tasks):
                invocation = await completed
                remaining -= 1
                if remaining == 0:
                    await self._finish_tool_execution(context)
                async with aclosing(self._finalize_tool(context, invocation)) as events:
                    async for event in events:
                        yield event
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _execute_serial_tools(
        self,
        context: AgentContext,
        calls: Sequence[ToolCall],
        active_input: AgentMessage | UserMessage | None,
    ) -> AsyncIterator[AgentEvent]:
        """Run serial calls one at a time and stop at a steering boundary."""
        for index, call in enumerate(calls):
            await self._notify_before_tool(context, call)
            async with aclosing(self._before_tool_events(context, call)) as preprocessing:
                async for event in preprocessing:
                    yield event
            await self._start_tool_execution(context)
            yield AgentEvent(
                AgentEventType.TOOL_STARTED,
                session_id=context.config.session_id,
                phase=context.state.phase,
                tool_calls=[call],
            )
            invocation = await self._invoke_tool(context, call)
            await self._finish_tool_execution(context)
            async with aclosing(self._finalize_tool(context, invocation)) as events:
                async for event in events:
                    yield event
            steering = self._steering_extension.take(context)
            if steering is not None:
                async with aclosing(
                    self._start_steering(context, steering, calls[index + 1 :], active_input)
                ) as steering_events:
                    async for event in steering_events:
                        yield event
                return

    @staticmethod
    async def _invoke_tool(context: AgentContext, call: ToolCall) -> _ToolInvocation:
        """Run only a handler concurrently; lifecycle hooks remain serialized."""
        from .extensions.events import MessageTiming

        started_at = datetime.now(UTC)
        started_ns = monotonic_ns()
        error = None
        token = _bind_tool_call(call.id)
        try:
            try:
                registered = context.tools.get(call.name)
                if registered is None:
                    raise ValueError(f"Unknown tool: {call.name}")
                output = await registered(call.arguments)
                content = registered.serialize_result(output)
            except Exception as tool_error:
                error = tool_error
                content = json.dumps({"error": str(tool_error)})
        finally:
            _reset_tool_call(token)
        completed_ns = monotonic_ns()
        completed_at = datetime.now(UTC)
        result = ToolMessage(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            success=error is None,
        )
        return _ToolInvocation(
            call=call,
            result=result,
            error=error,
            timing=MessageTiming(
                started_at=started_at,
                completed_at=completed_at,
                duration_ns=max(0, completed_ns - started_ns),
            ),
        )

    async def _finalize_tool(
        self,
        context: AgentContext,
        invocation: _ToolInvocation,
    ) -> AsyncIterator[AgentEvent]:
        """Publish one completed invocation and append its result to context."""
        call, result, error = invocation.call, invocation.result, invocation.error
        await self._notify_after_tool(context, call, result, error)
        await context.append_message(result, invocation.timing)
        yield AgentEvent(
            AgentEventType.TOOL_FAILED if error else AgentEventType.TOOL_COMPLETED,
            session_id=context.config.session_id,
            phase=context.state.phase,
            tool_calls=[call],
            message=result,
            error=error,
        )
        async with aclosing(self._after_tool_events(context, call, result, error)) as events:
            async for event in events:
                yield event

    async def _start_steering(
        self,
        context: AgentContext,
        message: UserMessage,
        pending_calls: Sequence[ToolCall] = (),
        active_input: AgentMessage | UserMessage | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Finish pending tool protocol entries before appending urgent user input.

        Skipped calls are never invoked and do not trigger execution hooks. Their
        explicit failed ToolMessages preserve the provider's call/result pairing
        and Raw Log history. Interrupted inputs never emit a false completion.
        """
        from .extensions.events import MessageTiming

        if active_input is not None:
            internal = isinstance(active_input, AgentMessage)
            yield AgentEvent(
                AgentEventType.INTERNAL_MESSAGE_INTERRUPTED if internal else AgentEventType.STEERING_INTERRUPTED,
                session_id=context.config.session_id,
                phase=context.state.phase,
                internal_message=active_input if internal else None,
                steering_message=active_input if isinstance(active_input, UserMessage) else None,
            )
        for call in pending_calls:
            result = ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=json.dumps({"skipped": True, "reason": "Superseded by a steering message"}),
                success=False,
            )
            await context.append_message(result, MessageTiming.instant())
            yield AgentEvent(
                AgentEventType.TOOL_SKIPPED,
                session_id=context.config.session_id,
                phase=context.state.phase,
                tool_calls=[call],
                message=result,
            )
        await context.append_message(message, MessageTiming.instant())
        yield AgentEvent(
            AgentEventType.STEERING_STARTED,
            session_id=context.config.session_id,
            phase=context.state.phase,
            steering_message=message,
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

    async def _after_model_events(self, context: AgentContext, response: ModelResponse) -> AsyncIterator[AgentEvent]:
        for extension in self.extensions:
            async with aclosing(extension.after_model_events(context, response)) as events:
                async for event in events:
                    self._validate_post_operation_event(context, event)
                    yield event

    async def _after_tool_events(
        self,
        context: AgentContext,
        call: ToolCall,
        result: ToolMessage,
        error: Exception | None,
    ) -> AsyncIterator[AgentEvent]:
        for extension in self.extensions:
            async with aclosing(extension.after_tool_events(context, call, result, error)) as events:
                async for event in events:
                    self._validate_post_operation_event(context, event)
                    yield event

    @staticmethod
    def _validate_post_operation_event(context: AgentContext, event: AgentEvent) -> None:
        """Do not let a post-operation hook impersonate runtime or other-session events."""
        if event.type is not AgentEventType.CUSTOM:
            raise AgentProtocolError("Post-operation hooks may emit only CUSTOM events")
        if event.session_id != context.config.session_id:
            raise AgentProtocolError("Post-operation event belongs to another session")

    async def _notify_on_state(self, context: AgentContext) -> None:
        """Restore history and system instructions before input conversion."""
        for extension in self.extensions:
            await extension.on_state(context)

    async def _notify_on_message(self, context: AgentContext) -> None:
        """Let extensions transform current input before append and persistence."""
        for extension in self.extensions:
            await extension.on_message(context)

    async def _notify_before_run(self, context: AgentContext) -> None:
        for extension in self.extensions:
            await extension.before_run(context)

    @staticmethod
    def _refresh_model_request(context: AgentContext, request: ModelRequest) -> ModelRequest:
        """Capture current context while retaining per-call model options.

        Preprocessing can replace history or tool registrations. Refresh before
        each observer and the provider so neither sees stale pre-compaction input.
        """
        return replace(
            request,
            messages=tuple(context.state.messages),
            tools=tuple(tool.definition for tool in context.tools.values()),
            server_tools=tuple(context.server_tools.values()),
        )

    async def _notify_before_model(self, context: AgentContext, request: ModelRequest) -> None:
        for extension in self.extensions:
            await extension.before_model(context, self._refresh_model_request(context, request))

    async def _before_model_events(self, context: AgentContext, request: ModelRequest) -> AsyncIterator[AgentEvent]:
        for extension in self.extensions:
            async with aclosing(
                extension.before_model_events(context, self._refresh_model_request(context, request))
            ) as events:
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
        error: Exception | None,
    ) -> None:
        for extension in self.extensions:
            await extension.after_tool(context, call, result, error)

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
