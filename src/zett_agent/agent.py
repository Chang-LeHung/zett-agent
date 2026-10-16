from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import TYPE_CHECKING, Any, overload

from ._agent.hooks import ExtensionHooksMixin
from ._agent.lifecycle import RequestLifecycleMixin, final_answer
from ._agent.model_step import ModelStepMixin
from ._agent.recovery import RecoveryMixin
from ._agent.tool_runner import ToolRunnerMixin
from ._agent.turns import TurnLoopMixin
from ._compat import Self
from .event_queue import AgentEventQueue
from .events import AgentEvent, AgentEventType, AgentPhase, AgentPhaseTransitionMixin
from .exceptions import AgentProtocolError
from .ids import new_uuid7
from .json_types import JsonValue
from .messages import AnyMessage, AssistantMessage, UserMessage
from .model import (
    AgentModel,
    ModelUsage,
    ReasoningEffort,
    ServerToolDefinition,
)
from .resume import ResumePoint
from .sync_runtime import SyncMethodsMixin
from .tools.base import AgentTool

if TYPE_CHECKING:
    from .extensions.base import AgentExtension
    from .extensions.events import ExtensionEvent, MessageTiming
    from .extensions.external import ExternalEvent


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    """Identify a conversation and optionally one request within it.

    Attributes:
        session_id: Conversation identifier, reused across turns. None or an
            empty value is replaced by a new UUIDv7 when the runtime starts.
        request_id: Optional application correlation ID for this invocation.
            It is not an idempotency key; reusing it does not deduplicate messages.
        parent_session_id: Parent conversation when running a delegated child.
            Must differ from session_id.
        cache_key: Prompt cache routing key declared to the provider. None
            reuses session_id, which keeps every step of one conversation on the
            same cached prefix; set it to route several conversations onto one
            prefix instead.

    Raises:
        ValueError: If an identity is empty or a session references itself.

    Examples:
        A root conversation needs only a session ID::

            config = AgentRunConfig(session_id="chat-42")

        A delegated agent identifies its parent separately::

            child = AgentRunConfig(
                session_id="review-42",
                request_id="request-7",
                parent_session_id="chat-42",
            )

    .. note::
        ``request_id`` is correlation data, not an idempotency key. Reusing it
        does not suppress a second message or model call.

    .. seealso::
        :class:`~zett_agent.storage.SQLiteSessionStorage` for persisted session identity,
        and :class:`~zett_agent.extensions.subagent.SubAgentDefinition` for child-session setup.
    """

    session_id: str | None = None
    request_id: str | None = None
    parent_session_id: str | None = None
    cache_key: str | None = None

    def __post_init__(self) -> None:
        if self.session_id is not None:
            normalized_session_id = self.session_id.strip()
            if not normalized_session_id:
                object.__setattr__(self, "session_id", None)
            elif normalized_session_id != self.session_id:
                object.__setattr__(self, "session_id", normalized_session_id)
        if self.request_id is not None:
            normalized_request_id = self.request_id.strip()
            if not normalized_request_id:
                object.__setattr__(self, "request_id", None)
            elif normalized_request_id != self.request_id:
                object.__setattr__(self, "request_id", normalized_request_id)
        if self.parent_session_id is not None:
            if not self.parent_session_id.strip():
                raise ValueError("parent_session_id cannot be empty")
            if self.session_id is not None and self.parent_session_id == self.session_id:
                raise ValueError("parent_session_id must differ from session_id")
        if self.cache_key is not None:
            normalized_cache_key = self.cache_key.strip()
            if not normalized_cache_key:
                object.__setattr__(self, "cache_key", None)
            elif normalized_cache_key != self.cache_key:
                object.__setattr__(self, "cache_key", normalized_cache_key)


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
        :class:`~zett_agent.events.AgentPhase` documents the state machine, while
        :class:`~zett_agent.extensions.persistence.SessionView` describes restored persistent context.
    """

    _messages: list[AnyMessage] = field(default_factory=list)
    phase: AgentPhase = AgentPhase.CREATED
    # Parent conversation when this request belongs to a delegated subagent.
    parent_session_id: str | None = None

    @property
    def messages(self) -> tuple[AnyMessage, ...]:
        """Return a read-only view of the current message sequence."""
        return tuple(self._messages)


#: Python 3.11 added ``weakref_slot``. On 3.10 a slotted dataclass has no room
#: for ``__weakref__`` (``slots=True`` also rejects a class that declares
#: ``__slots__`` itself), while extensions key request state by this context in
#: a ``WeakKeyDictionary``, so 3.10 keeps the ordinary instance dictionary.
_RUN_CONTEXT_DATACLASS = {"slots": True, "weakref_slot": True} if sys.version_info >= (3, 11) else {}


@dataclass(eq=False, **_RUN_CONTEXT_DATACLASS)
class AgentRunContext:
    """Per-run references shared by all lifecycle hooks.

    Register request-scoped local and server tools during on_tool(). Add or
    replace context messages through :meth:`add_message` and
    :meth:`replace_messages`; mutate either tool registry in place to update
    the current request without leaking registrations into another request or
    session.

    .. note::
        Context identity is request-scoped. It is safe to use a context as a key
        for temporary extension state only when every success, error, and
        cancellation path removes that entry.

    Examples:
        Register a tool for only the current request::

            async def on_tool(self, context: AgentRunContext) -> None:
                context.register_tool(read_file)

        Register a provider-hosted tool without adding a local executor::

            async def on_tool(self, context: AgentRunContext) -> None:
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
        :class:`~zett_agent.extensions.base.AgentExtension` for lifecycle hooks and
        :class:`~zett_agent.tools.base.AgentTool` for request-scoped tool registration.
    """

    # Configuration for this invocation.
    config: AgentRunConfig
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
    # One request-owned channel. Internal code emits here; only Agent.stream
    # consumes it and exposes an async iterator to callers.
    event_queue: AgentEventQueue = field(default_factory=AgentEventQueue)
    # New messages waiting for their single MessageAppendedEvent.
    _pending_message_events: list[tuple[AnyMessage, MessageTiming, ModelUsage | None]] = field(
        default_factory=list,
        repr=False,
    )

    async def emit(self, event: AgentEvent) -> None:
        """Queue one extension-owned CUSTOM or compaction event for the caller."""
        if event.type not in {
            AgentEventType.CUSTOM,
            AgentEventType.COMPACTION_STARTED,
            AgentEventType.COMPACTION_TEXT_DELTA,
            AgentEventType.COMPACTION_REASONING_DELTA,
            AgentEventType.COMPACTION_COMPLETED,
        }:
            raise AgentProtocolError(f"Extensions cannot emit runtime-owned event type {event.type.value!r}")
        await self._emit(event)

    async def _emit_runtime(self, event: AgentEvent) -> None:
        """Queue one core-owned event without exposing runtime event ownership."""
        await self._emit(event)

    async def _emit_event(self, event_type: AgentEventType, **fields: Any) -> None:
        """Queue one core-owned event of ``event_type`` for this session.

        The session is filled in, and the phase is stamped as the event is queued, so a caller names
        only the type and the fields that type carries.
        """
        await self._emit_runtime(AgentEvent(event_type, session_id=self.config.session_id, **fields))

    async def _emit(self, event: AgentEvent) -> None:
        """Validate, phase-stamp, and enqueue one outward event."""
        if event.session_id != self.config.session_id:
            raise AgentProtocolError("Agent event belongs to another session")
        match event.type:
            case AgentEventType.COMPACTION_STARTED:
                await AgentPhaseTransitionMixin._transition_phase(
                    self,
                    AgentPhase.COMPACTING,
                    expected=(AgentPhase.READY,),
                )
            case AgentEventType.COMPACTION_TEXT_DELTA | AgentEventType.COMPACTION_REASONING_DELTA:
                AgentPhaseTransitionMixin._require_phase(self.state, AgentPhase.COMPACTING)
            case AgentEventType.COMPACTION_COMPLETED:
                await AgentPhaseTransitionMixin._transition_phase(
                    self,
                    AgentPhase.READY,
                    expected=(AgentPhase.COMPACTING,),
                )
            case _:
                pass
        event.phase = self.state.phase
        await self.event_queue.put(event)

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
        """Append one new message and notify persistence subscribers.

        Runtime code must use this method for produced messages so in-memory
        context and persistence notifications cannot drift apart. Subscribers
        decide from ``message.persist`` whether the message belongs in storage.

        The message is visible in ``state.messages`` before subscribers run.
        Subscriber failures propagate and do not roll back the in-memory append.
        """
        self.state._messages.append(message)
        await self._publish_message_appended(message, timing, usage)

    def add_message(
        self,
        message: AnyMessage,
        *,
        index: int | None = None,
        usage: ModelUsage | None = None,
    ) -> None:
        """Add one extension-owned message and queue its append event.

        The queued event is flushed by the runtime after setup hooks or before
        the next model request. Extensions must use this method instead of
        mutating ``AgentState.messages`` directly.
        """
        from .extensions.events import MessageTiming

        if index is None:
            self.state._messages.append(message)
        else:
            self.state._messages.insert(index, message)
        self._pending_message_events.append((message, MessageTiming.instant(), usage))

    def replace_messages(
        self,
        messages: Sequence[AnyMessage],
        *,
        emit_new: bool = True,
    ) -> None:
        """Replace model context and optionally queue events for newly added messages.

        Restored history and compaction use ``emit_new=False`` because those
        messages are already represented by storage records or a snapshot.
        """
        old_ids = {id(message) for message in self.state._messages}
        replacement = list(messages)
        self.state._messages[:] = replacement
        if not emit_new:
            return
        for message in replacement:
            if id(message) not in old_ids:
                self.add_message_event(message)

    def add_message_event(self, message: AnyMessage, usage: ModelUsage | None = None) -> None:
        """Queue one append event for a message already present in state."""
        from .extensions.events import MessageTiming

        self._pending_message_events.append((message, MessageTiming.instant(), usage))

    async def flush_message_events(self) -> None:
        """Publish every queued message event exactly once."""
        pending = self._pending_message_events
        self._pending_message_events = []
        by_message_id = {id(message): (message, timing, usage) for message, timing, usage in pending}
        positions = {id(message): index for index, message in enumerate(self.state._messages)}
        ordered = sorted(
            (item for item in by_message_id.values() if id(item[0]) in positions),
            key=lambda item: positions[id(item[0])],
        )
        for message, timing, usage in ordered:
            await self._publish_message_appended(message, timing, usage)

    async def _publish_message_appended(
        self,
        message: AnyMessage,
        timing: MessageTiming,
        usage: ModelUsage | None = None,
    ) -> None:
        """Notify subscribers once for one message already present in context."""
        from .extensions.events import MessageAppendedEvent

        await self.publish(MessageAppendedEvent(message, timing, usage))


class Agent(
    ExtensionHooksMixin,
    RequestLifecycleMixin,
    TurnLoopMixin,
    ModelStepMixin,
    ToolRunnerMixin,
    RecoveryMixin,
    AgentPhaseTransitionMixin,
    SyncMethodsMixin,
):
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
                config=AgentRunConfig(session_id="chat-42"),
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
        :func:`~zett_agent.client.create_agent` is the convenient initialized entry
        point. :class:`~zett_agent.client.AgentClient` adds application event dispatch,
        and :doc:`/concepts/lifecycle` explains the complete loop.
    """

    #: How many times in a row a ``paused`` stop is continued without asking anyone. A
    #: provider returns ``paused`` when it hands control back mid-turn (Anthropic does so
    #: when a server-tool loop reaches its limit) and expects the turn to be sent again. At
    #: this many in a row the run ends and ``MODEL_STOPPED`` reports the pause. Zero never
    #: continues. Set it on an instance to change it.
    max_pause_continuations: int = 5

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
                Their inherited middleware hooks wrap model requests and local
                tool handlers in the same priority order.
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
        self.extensions, self._internal_message_extension, self._steering_extension = self._assemble_extensions(
            extensions
        )
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.parallel_tool_call = parallel_tool_call
        self.state = AgentState()
        # Runtime data is session-scoped. state remains the most recently
        # started state for single-session callers; use get_state for routing.
        self._states: dict[str, AgentState] = {}
        self._active_configs: dict[str, AgentRunConfig] = {}
        self._sessions_lock = RLock()
        self._initialized_config: AgentRunConfig | None = None
        self.max_iterations = max_iterations
        self.max_internal_messages = max_internal_messages

    async def initialize(self, *, config: AgentRunConfig | None = None) -> None:
        """Bind the default session used when a request omits config.

        Examples:
            Usage::

                agent = Agent(model)
                await agent.initialize(config=AgentRunConfig(session_id="session-42"))
                reply = await agent.run("Hello")
        """
        requested = config or AgentRunConfig()
        if self._initialized_config is not None:
            if requested.session_id is None:
                return
            if self._initialized_config.session_id != requested.session_id:
                raise AgentProtocolError("Agent is already initialized for another session")
            return
        session_id = requested.session_id or new_uuid7()
        if requested.parent_session_id == session_id:
            raise ValueError("parent_session_id must differ from session_id")
        self._initialized_config = (
            requested if requested.session_id is not None else replace(requested, session_id=session_id)
        )

    def get_state(self, session_id: str) -> AgentState | None:
        """Return one session's latest state, or None before its first request.

        Concurrent callers must select by session rather than reading state,
        which is only the most recently started request's convenience view.
        """
        with self._sessions_lock:
            return self._states.get(session_id)

    @property
    def pending(self) -> bool:
        """Return whether this Agent owns at least one unfinished request.

        A request becomes pending when stream execution claims its session and
        remains pending through model calls, tools, external waits, and terminal
        hooks. Success, failure, cancellation, and explicit stream closure all
        clear it. Concurrent sessions keep the value true until the last one
        exits.
        """
        with self._sessions_lock:
            return bool(self._active_configs)

    def is_conflict(self, session_id: str) -> bool:
        """Return whether another request currently owns ``session_id``.

        ``run()`` and ``stream()`` use the same check while atomically claiming
        the session, so concurrent callers cannot both start work for one
        conversation. Requests for different sessions remain independent.

        Args:
            session_id: Non-empty conversation identity to inspect.

        Raises:
            ValueError: If ``session_id`` is empty.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id cannot be empty")
        with self._sessions_lock:
            previous = self._states.get(session_id)
            return session_id in self._active_configs or bool(previous and not previous.phase.accepts_new_request)

    def emit_external_event(self, event: ExternalEvent, *, config: AgentRunConfig | None = None) -> list[str]:
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
                    config=AgentRunConfig(session_id="session-42", request_id="request-1"),
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
        config: AgentRunConfig | None = None,
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
                None generates a UUIDv7 session during initialization.
            system_prompt: Initial instructions for every fresh request state.
            tools: Static tools registered before extension setup.
            extensions: Optional lifecycle and middleware extensions; None keeps
                defaults.
            reasoning_effort: Default reasoning level for future requests.
            parallel_tool_call: Enable provider and local parallel tool calls.
            max_iterations: Maximum model calls per user/internal input.
            max_internal_messages: Maximum internal continuations per request.

        Returns:
            An initialized instance of the class on which create() was called.

        Examples:
            Usage::

                agent = await Agent.create(model, config=AgentRunConfig(session_id="session-42"))
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
        config: AgentRunConfig | None = None,
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
        config: AgentRunConfig | None = None,
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
        config: AgentRunConfig | None = None,
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

                agent = await Agent.create(model, config=AgentRunConfig(session_id="session-42"))
                reply = await agent.run(
                    "Summarize this conversation.",
                    config=AgentRunConfig(session_id="session-42"),
                    metadata={"source": "editor"},
                    tags={"domain": "notes"},
                )
        """
        return await final_answer(
            self.stream(
                message,
                config=config,
                model=model,
                reasoning_effort=reasoning_effort,
                parallel_tool_call=parallel_tool_call,
                metadata=metadata,
                tags=tags,
            )
        )

    async def resume_point(
        self,
        *,
        config: AgentRunConfig | None = None,
        model: AgentModel | None = None,
    ) -> ResumePoint | None:
        """Report where a session was interrupted, without changing it.

        A request can end without finishing: the connection drops, the user stops it, the
        process is killed. Nothing records that, so the session is restored exactly as a
        request restores it and its last message is read. Nothing is written and no model is
        called.

        Args:
            config: Session identity; omitted to use the initialized session.
            model: Only needed when the Agent has no default model.

        Returns:
            Where it was interrupted and what :meth:`resume` would do, or None when there is
            nothing to resume: the session is empty, its last request finished, or it ended
            in a refusal, a full context window, or a stop this library does not know.
            :meth:`resume` raises for those.

        Examples:
            Offer a resume only when there is one::

                point = await agent.resume_point()
                if point is not None:
                    print(point.description)
        """
        context = self._prepare_request_context(config, model, None, None, None)
        async with self._request_scope(context):
            return await self._inspect_session(context)

    async def resume(
        self,
        *,
        config: AgentRunConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AssistantMessage:
        """Pick an interrupted session up where it stopped and return the final answer.

        This is :meth:`run` without a new user message. The session is restored, its last
        message says how it was interrupted, and the request carries on from there; see
        :class:`~zett_agent.resume.ResumeKind` for what is done in each case. Pass the same
        model, tools and settings the interrupted request had: none of them is stored.

        Args:
            config: Session identity; omitted to use the initialized session.
            model: Request-specific model; omitted to use the Agent default.
            reasoning_effort: Optional reasoning override for this request.
            parallel_tool_call: Optional parallel-call override for this request.
            metadata: JSON data for extensions and Raw Log persistence, not a prompt.
            tags: JSON classifications for extensions and Raw Log persistence.

        Returns:
            The final assistant answer.

        Raises:
            AgentResumeError: If there is nothing to resume. Ask :meth:`resume_point` first
                to find out without an exception.
            AgentIterationLimitError: If the model/internal-message budget is exhausted.

        Examples:
            Carry on after a failure::

                try:
                    reply = await agent.run("Refactor the parser")
                except ProviderTimeoutError:
                    reply = await agent.resume()
        """
        return await final_answer(
            self.stream_resume(
                config=config,
                model=model,
                reasoning_effort=reasoning_effort,
                parallel_tool_call=parallel_tool_call,
                metadata=metadata,
                tags=tags,
            )
        )

    async def stream_resume(
        self,
        *,
        config: AgentRunConfig | None = None,
        model: AgentModel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        parallel_tool_call: bool | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream :meth:`resume`; the events are those of :meth:`stream`.

        Args:
            config: Session identity; omitted to use the initialized session.
            model: Request-specific model; omitted to use the Agent default.
            reasoning_effort: Reasoning override limited to this request.
            parallel_tool_call: Parallel-call override limited to this request.
            metadata: Request data visible to extensions and persistence.
            tags: Request classifications visible to extensions and persistence.

        Yields:
            AgentEvent objects in execution order, ending in RUN_COMPLETED on success.

        Note:
            The session is only known once it is restored, so a session with nothing to resume
            raises :class:`~zett_agent.exceptions.AgentResumeError` as the request fails, and
            runs the error hooks like any failed request. Use :meth:`resume_point` to ask first.
            The ``on_message`` hook is not called, because there is no new input for it to
            transform, and a goal that was armed for the interrupted request is not re-armed.
        """
        effort, allow_parallel = self._resolve_run_options(reasoning_effort, parallel_tool_call)
        context = self._prepare_request_context(config, model, metadata, tags, None)
        work = self._run_request(context, effort, allow_parallel, resume=True)
        async with aclosing(self._pump_events(context, work)) as events:
            async for event in events:
                yield event

    @overload
    def stream(
        self,
        message: UserMessage,
        *,
        config: AgentRunConfig | None = None,
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
        config: AgentRunConfig | None = None,
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
        config: AgentRunConfig | None = None,
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
                    config=AgentRunConfig(session_id="session-42"),
                    reasoning_effort=ReasoningEffort.HIGH,
                    metadata={"source": "command-palette"},
                    tags={"intent": "summary"},
                ):
                    print(event.type, event.session_id)
        """
        user_message = message if isinstance(message, UserMessage) else UserMessage(content=message)
        effort, allow_parallel = self._resolve_run_options(reasoning_effort, parallel_tool_call)
        context = self._prepare_request_context(config, model, metadata, tags, user_message)
        work = self._run_request(context, effort, allow_parallel)
        async with aclosing(self._pump_events(context, work)) as events:
            async for event in events:
                yield event

    async def compact(
        self,
        *,
        config: AgentRunConfig | None = None,
        model: AgentModel | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Compact the session's history without a user message or a model turn.

        The session is restored exactly as a request restores it — ``on_tool``,
        then ``on_state`` — and every extension is asked to compact through its
        ``on_compact`` hook. Nothing else happens: no input message is appended
        and no primary model call is made, so a caller can summarize on demand
        instead of waiting for the context to grow past a budget.

        Args:
            config: Session identity; defaults to the initialized configuration.
            model: Model an extension compacts with when it has none of its own.
            metadata: Request data visible to extensions and persistence.
            tags: Request classifications visible to extensions and persistence.

        Yields:
            AgentEvent objects in execution order: whatever the extensions emit
            while compacting, and nothing at all when they find nothing to
            compact.

        Note:
            A pass that changes nothing ends without a terminal event, which is
            why :meth:`AgentClient.compact` exists for callers that want one
            result instead of an event stream. Concurrent requests must use
            different session IDs; a maintenance pass claims its session like a
            request does.
        """
        context = self._prepare_request_context(config, model, metadata, tags, None)
        pass_ = self._run_compaction(context)
        async with aclosing(self._pump_events(context, pass_, ends_silently=True)) as events:
            async for event in events:
                yield event
