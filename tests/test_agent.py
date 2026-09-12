import asyncio

import pytest

from zett_agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    AgentExtension,
    AgentIterationLimitError,
    AgentPhase,
    AgentPhaseTransitionMixin,
    AgentProtocolError,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AssistantMessage,
    ContentCompletedEvent,
    ContentStartedEvent,
    InMemoryMessageAccumulator,
    InternalMessageExtension,
    MessageAppendedEvent,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    PhaseTransitionEvent,
    ReasoningCompletedEvent,
    ReasoningEffort,
    ReasoningStartedEvent,
    ServerToolDefinition,
    SteeringExtension,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
    UserMessage,
    tool,
)

CONFIG = AgentRunConfig(session_id="test-session")


def test_agent_config_validates_session_identity_fields() -> None:
    config = AgentRunConfig("session", request_id="request", parent_session_id="parent")

    assert (config.session_id, config.request_id, config.parent_session_id) == ("session", "request", "parent")


async def test_phase_transition_mixin_validates_predecessors() -> None:
    transitions: list[PhaseTransitionEvent] = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, PhaseTransitionEvent):
                assert context.state.phase == event.current_phase
                transitions.append(event)

    machine = AgentPhaseTransitionMixin()
    state = AgentState()
    context = AgentRunContext(CONFIG, state, {}, (Observer(),))

    await machine._start_context_loading(context)
    await machine._finish_context_loading(context)
    await machine._start_compaction(context)
    await machine._finish_compaction(context)
    await machine._start_model_generation(context)
    await machine._finish_model_generation(context)
    await machine._start_tool_execution(context)
    await machine._finish_tool_execution(context)
    await machine._complete_request(context)

    assert state.phase == AgentPhase.COMPLETED
    assert [(event.previous_phase, event.current_phase) for event in transitions] == [
        (AgentPhase.CREATED, AgentPhase.LOADING_CONTEXT),
        (AgentPhase.LOADING_CONTEXT, AgentPhase.READY),
        (AgentPhase.READY, AgentPhase.COMPACTING),
        (AgentPhase.COMPACTING, AgentPhase.READY),
        (AgentPhase.READY, AgentPhase.GENERATING),
        (AgentPhase.GENERATING, AgentPhase.READY),
        (AgentPhase.READY, AgentPhase.RUNNING_TOOL),
        (AgentPhase.RUNNING_TOOL, AgentPhase.READY),
        (AgentPhase.READY, AgentPhase.COMPLETED),
    ]
    with pytest.raises(AgentProtocolError, match="Invalid agent phase transition"):
        await machine._start_model_generation(context)
    assert state.phase == AgentPhase.COMPLETED
    assert len(transitions) == 9


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
async def test_phase_transition_mixin_supports_terminal_paths(terminal: str) -> None:
    transitions: list[PhaseTransitionEvent] = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, PhaseTransitionEvent):
                transitions.append(event)

    machine = AgentPhaseTransitionMixin()
    state = AgentState()
    context = AgentRunContext(CONFIG, state, {}, (Observer(),))
    await machine._start_context_loading(context)

    if terminal == "failed":
        await machine._fail_request(context)
        assert state.phase == AgentPhase.FAILED
    else:
        await machine._cancel_request(context)
        assert state.phase == AgentPhase.CANCELLED
    assert transitions[-1].previous_phase == AgentPhase.LOADING_CONTEXT
    assert transitions[-1].current_phase == state.phase
    assert transitions[-1].occurred_at.tzinfo is not None
    assert transitions[-1].monotonic_ns > 0


async def test_uninitialized_agent_rejects_requests_without_calling_model():
    model = ScriptedModel(AssistantMessage(content="Ready"))
    agent = Agent(model)
    with pytest.raises(AgentProtocolError, match="not initialized"):
        await agent.run("Hello", config=CONFIG)
    with pytest.raises(AgentProtocolError, match="not initialized"):
        _ = [event async for event in agent.stream("Hello")]
    assert model.requests == []
    await agent.initialize(config=CONFIG)
    assert (await agent.run("Hello")).content == "Ready"


async def test_run_exposes_metadata_and_tags_only_through_agent_context():
    observed = []

    class ContextObserver(AgentExtension):
        async def before_run(self, context):
            observed.append((context.metadata, context.tags))

    model = ScriptedModel(AssistantMessage(content="Done"))
    agent = await Agent.create(model, config=CONFIG, extensions=[ContextObserver()])

    await agent.run(
        "Classify this input",
        metadata={"source": "clipboard", "asset_ids": ["asset-1"]},
        tags={"domain": "python"},
    )

    user = next(message for message in model.requests[0].messages if isinstance(message, UserMessage))
    assert user == UserMessage(content="Classify this input")
    assert observed == [({"source": "clipboard", "asset_ids": ["asset-1"]}, {"domain": "python"})]


async def test_user_message_input_can_use_separate_extension_context():
    observed = []

    class ContextObserver(AgentExtension):
        async def before_run(self, context):
            observed.append(context.metadata)

    model = ScriptedModel(AssistantMessage(content="Done"))
    agent = await Agent.create(model, config=CONFIG, extensions=[ContextObserver()])
    await agent.run(UserMessage(content="Input"), metadata={"source": "editor"})

    assert observed == [{"source": "editor"}]
    assert UserMessage(content="Input") in model.requests[0].messages


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"metadata": {1: "invalid"}}, "metadata keys"),
        ({"tags": {"": "invalid"}}, "tags keys"),
        ({"metadata": {"value": object()}}, "JSON serializable"),
        ({"metadata": {"value": (1, 2)}}, "JSON serializable"),
        ({"tags": {"score": float("nan")}}, "JSON serializable"),
    ],
)
async def test_run_rejects_invalid_extension_context(kwargs, error):
    model = ScriptedModel(AssistantMessage(content="Unused"))
    agent = await Agent.create(model, config=CONFIG)

    with pytest.raises(ValueError, match=error):
        await agent.run("Input", **kwargs)

    assert model.requests == []


async def test_failed_initialization_does_not_enable_requests():
    class Failure(AgentExtension):
        async def on_state(self, context):
            raise RuntimeError("Restore failed")

    agent = Agent(ScriptedModel(), extensions=[Failure()])
    await agent.initialize(config=CONFIG)
    with pytest.raises(RuntimeError, match="Restore failed"):
        await agent.run("Hello")


async def test_create_preserves_subclass_and_restores_when_a_request_starts():
    class Restore(AgentExtension):
        async def on_state(self, context):
            await asyncio.sleep(0)
            context.state.messages.append(UserMessage(content="Restored"))

    class CustomAgent(Agent):
        pass

    agent = await CustomAgent.create(
        ScriptedModel(AssistantMessage(content="Done")), config=CONFIG, extensions=[Restore()]
    )
    assert isinstance(agent, CustomAgent)
    assert agent.state.messages == []
    await agent.run("Hello")
    assert agent.state.messages[1].content == "Restored"


@pytest.mark.parametrize("failure_hook", ["after_run", "on_success"])
async def test_success_callback_failures_prevent_completion(failure_hook):
    calls = []

    class Callback(AgentExtension):
        async def after_run(self, context, result):
            if failure_hook == "after_run":
                raise RuntimeError("Callback failed")

        async def on_success(self, context, result):
            calls.append("success")
            raise RuntimeError("Callback failed")

        async def on_error(self, context, error):
            calls.append("error")

    agent = await Agent.create(ScriptedModel(AssistantMessage(content="Done")), extensions=[Callback()], config=CONFIG)
    events = []
    with pytest.raises(RuntimeError, match="Callback failed"):
        async for event in agent.stream("Hello", config=CONFIG):
            events.append(event)
    assert calls == (["error"] if failure_hook == "after_run" else ["success", "error"])
    assert all(event.type != AgentEventType.RUN_COMPLETED for event in events)


async def test_context_registers_request_scoped_tools_before_messages_load():
    contexts: list[AgentRunContext] = []
    calls: list[str] = []

    class RegisterTools(AgentExtension):
        async def on_tool(self, context: AgentRunContext) -> None:
            calls.append("on_tool")
            context.register_tool(add)

        async def on_state(self, context: AgentRunContext) -> None:
            calls.append("on_state")
            assert add.name in context.tools

        async def before_run(self, context: AgentRunContext) -> None:
            contexts.append(context)

        async def before_tool(self, context: AgentRunContext, call: ToolCall) -> None:
            assert context is contexts[-1]

    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 2, "right": 3}),)),
        AssistantMessage(content="5"),
        AssistantMessage(content="Again"),
    )
    agent = await Agent.create(model, extensions=[RegisterTools(), ToolGuidelinesExtension()], config=CONFIG)
    await agent.run("Add", config=CONFIG)
    next_config = AgentRunConfig(session_id=CONFIG.session_id)
    await agent.run("Continue", config=next_config)
    assert contexts[0] is not contexts[1]
    assert contexts[1].config is next_config
    assert contexts[0].state is not contexts[1].state
    assert contexts[1].state is agent.state
    assert contexts[0].tools is not contexts[1].tools
    assert contexts[0].tools is not agent.tools
    assert contexts[1].tools is not agent.tools
    assert agent.tools == {}
    assert calls == ["on_tool", "on_state", "on_tool", "on_state"]
    assert model.requests[0].tools == (add.definition,)
    assert any("## add\n- Use for exact addition." in message.content for message in model.requests[0].messages)
    assert model.requests[1].messages[-1].content == "5"


async def test_extension_registers_request_scoped_server_tools() -> None:
    definition = ServerToolDefinition(
        type="web_search",
        configuration={"filters": {"allowed_domains": ["example.com"]}},
    )
    contexts: list[AgentRunContext] = []

    class RegisterServerTool(AgentExtension):
        async def on_tool(self, context: AgentRunContext) -> None:
            context.register_server_tool(definition)

        async def on_state(self, context: AgentRunContext) -> None:
            assert tuple(context.server_tools) == ("web_search",)
            contexts.append(context)

    model = ScriptedModel(AssistantMessage(content="First"), AssistantMessage(content="Second"))
    agent = await Agent.create(model, extensions=[RegisterServerTool()], config=CONFIG)

    await agent.run("Search", config=CONFIG)
    await agent.run("Search again", config=AgentRunConfig(session_id=CONFIG.session_id))

    assert model.requests[0].server_tools == (definition,)
    assert model.requests[1].server_tools == (definition,)
    assert contexts[0].server_tools is not contexts[1].server_tools
    assert contexts[0].server_tools["web_search"] is not definition
    assert contexts[0].server_tools["web_search"] is not contexts[1].server_tools["web_search"]
    assert agent.tools == {}


async def test_context_rejects_duplicate_server_tool_registration() -> None:
    class DuplicateServerTools(AgentExtension):
        async def on_tool(self, context: AgentRunContext) -> None:
            context.register_server_tool(ServerToolDefinition(type="web_search"))
            context.register_server_tool(
                ServerToolDefinition(type="web_search", configuration={"search_context_size": "high"})
            )

    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="unused")),
        extensions=[DuplicateServerTools()],
        config=CONFIG,
    )

    with pytest.raises(ValueError, match="Server tool 'web_search' is already registered"):
        await agent.run("Hello")

    assert agent.state.phase == AgentPhase.FAILED


async def test_before_model_server_tool_registration_reaches_later_hooks_and_provider() -> None:
    seen: list[tuple[ServerToolDefinition, ...]] = []

    class RegisterBeforeModel(AgentExtension):
        name = "register-server-tool"

        async def before_model(self, context: AgentRunContext, request: ModelRequest) -> None:
            assert request.server_tools == ()
            context.register_server_tool(ServerToolDefinition(type="web_search"))

    class ObserveBeforeModel(AgentExtension):
        name = "observe-server-tool"

        async def before_model(self, context: AgentRunContext, request: ModelRequest) -> None:
            seen.append(tuple(request.server_tools))

    model = ScriptedModel(AssistantMessage(content="Done"))
    agent = await Agent.create(
        model,
        extensions=[RegisterBeforeModel(), ObserveBeforeModel()],
        config=CONFIG,
    )

    await agent.run("Search")

    expected = (ServerToolDefinition(type="web_search"),)
    assert seen == [expected]
    assert model.requests[0].server_tools == expected


async def test_all_tool_hooks_finish_before_any_message_hook() -> None:
    calls: list[str] = []

    class GuidanceFirst(ToolGuidelinesExtension):
        async def on_state(self, context: AgentRunContext) -> None:
            calls.append("guidance:on_state")
            await super().on_state(context)

    class RegisterLast(AgentExtension):
        async def on_tool(self, context: AgentRunContext) -> None:
            calls.append("register:on_tool")
            context.register_tool(add)

        async def on_state(self, context: AgentRunContext) -> None:
            calls.append("register:on_state")

    model = ScriptedModel(AssistantMessage(content="Done"))
    agent = await Agent.create(
        model,
        extensions=[GuidanceFirst(), RegisterLast()],
        config=CONFIG,
    )

    await agent.run("Describe your tools")

    assert calls == ["register:on_tool", "guidance:on_state", "register:on_state"]
    assert model.requests[0].tools == (add.definition,)
    assert any("## add\n- Use for exact addition." in message.content for message in model.requests[0].messages)


async def test_context_rejects_duplicate_tool_registration() -> None:
    class DuplicateRegistration(AgentExtension):
        async def on_tool(self, context: AgentRunContext) -> None:
            context.register_tool(add)

    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="unused")),
        tools=[add],
        extensions=[DuplicateRegistration()],
        config=CONFIG,
    )

    with pytest.raises(ValueError, match="already registered"):
        await agent.run("Hello")

    assert agent.state.phase == AgentPhase.FAILED


async def test_extension_can_stream_typed_events_before_the_primary_model():
    class VisiblePreprocessing(AgentExtension):
        async def before_model_events(self, context, request):
            yield AgentEvent(AgentEventType.COMPACTION_STARTED, context.config.session_id)
            yield AgentEvent(AgentEventType.COMPACTION_COMPLETED, context.config.session_id, applied=False)

    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="Done")),
        extensions=[VisiblePreprocessing()],
        config=CONFIG,
    )
    events = [event async for event in agent.stream("Hello")]
    assert [event.type for event in events[:3]] == [
        AgentEventType.COMPACTION_STARTED,
        AgentEventType.COMPACTION_COMPLETED,
        AgentEventType.MODEL_STARTED,
    ]
    assert events[2].phase == AgentPhase.GENERATING
    assert agent.state.phase == AgentPhase.COMPLETED


async def test_default_extensions_keep_agent_histories_isolated():
    first = await Agent.create(ScriptedModel(AssistantMessage(content="First reply")), config=CONFIG)
    second = await Agent.create(ScriptedModel(AssistantMessage(content="Second reply")), config=CONFIG)
    await first.run("First", config=CONFIG)
    await second.run("Second", config=CONFIG)
    first_memory = next(
        extension for extension in first.extensions if isinstance(extension, InMemoryMessageAccumulator)
    )
    second_memory = next(
        extension for extension in second.extensions if isinstance(extension, InMemoryMessageAccumulator)
    )
    guidance = next(extension for extension in first.extensions if isinstance(extension, ToolGuidelinesExtension))
    assert isinstance(first_memory, InMemoryMessageAccumulator)
    assert isinstance(guidance, ToolGuidelinesExtension)
    assert first_memory is not second_memory
    assert [message.content for message in first_memory.messages(CONFIG.session_id)] == [
        "First",
        "First reply",
    ]
    assert [message.content for message in second_memory.messages(CONFIG.session_id)] == [
        "Second",
        "Second reply",
    ]
    builtins = (await Agent.create(ScriptedModel(), extensions=[], config=CONFIG)).extensions
    assert len(builtins) == 2
    assert isinstance(builtins[0], InternalMessageExtension)
    assert isinstance(builtins[1], SteeringExtension)


async def test_extensions_run_by_stable_ascending_priority():
    calls: list[str] = []

    class RecordingExtension(AgentExtension):
        def __init__(self, name: str, priority: int) -> None:
            self.name = name
            self.priority = priority

        async def on_state(self, context):
            calls.append(self.name)

    late = RecordingExtension("late", 200)
    equal_first = RecordingExtension("equal-first", 100)
    early = RecordingExtension("early", 10)
    equal_second = RecordingExtension("equal-second", 100)
    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="Done")),
        extensions=[late, equal_first, early, equal_second],
        config=CONFIG,
    )

    await agent.run("Hello")

    assert tuple(
        extension
        for extension in agent.extensions
        if not isinstance(extension, (InternalMessageExtension, SteeringExtension))
    ) == (
        early,
        equal_first,
        equal_second,
        late,
    )
    assert calls == ["early", "equal-first", "equal-second", "late"]


@tool(guidelines="Use for exact addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


class ScriptedModel:
    def __init__(self, *messages: AssistantMessage):
        self.messages = list(messages)
        self.requests: list[ModelRequest] = []

    async def stream(self, request):
        self.requests.append(request)
        message = self.messages.pop(0)
        if message.content:
            yield ModelEvent.reasoning("Checking.")
            yield ModelEvent.text(message.content)
        yield ModelEvent.completed(ModelResponse(message))


async def test_model_output_boundaries_publish_in_order_and_measure_assistant_message():
    observed = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            observed.append(event)

    agent = await Agent.create(
        ScriptedModel(AssistantMessage(content="Answer")),
        config=CONFIG,
        extensions=[Observer()],
    )
    await agent.run("Question")

    lifecycle_types = (
        ReasoningStartedEvent,
        ReasoningCompletedEvent,
        ContentStartedEvent,
        ContentCompletedEvent,
    )
    lifecycle = [event for event in observed if isinstance(event, lifecycle_types)]
    assert [type(event) for event in lifecycle] == list(lifecycle_types)
    assert [event.occurred_at for event in lifecycle] == sorted(event.occurred_at for event in lifecycle)
    assert [event.monotonic_ns for event in lifecycle] == sorted(event.monotonic_ns for event in lifecycle)

    appended = [event for event in observed if isinstance(event, MessageAppendedEvent)]
    assert appended[0].timing.duration_ns == 0
    assistant_timing = appended[-1].timing
    assert assistant_timing.duration_ns >= assistant_timing.reasoning_duration_ns >= 0
    assert assistant_timing.duration_ns >= assistant_timing.content_duration_ns >= 0
    assert assistant_timing.reasoning_started_at == lifecycle[0].occurred_at
    assert assistant_timing.reasoning_completed_at == lifecycle[1].occurred_at
    assert assistant_timing.content_started_at == lifecycle[2].occurred_at
    assert assistant_timing.content_completed_at == lifecycle[3].occurred_at


async def test_tool_message_uses_running_tool_transition_for_duration():
    appended = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            if isinstance(event, MessageAppendedEvent):
                appended.append(event)

    agent = await Agent.create(
        ScriptedModel(
            AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 2, "right": 3}),)),
            AssistantMessage(content="5"),
        ),
        config=CONFIG,
        tools=[add],
        extensions=[Observer()],
    )
    await agent.run("Add")

    tool_event = next(event for event in appended if isinstance(event.message, ToolMessage))
    assert tool_event.timing.duration_ns >= 0
    assert tool_event.timing.reasoning_duration_ns is None
    assert tool_event.timing.content_duration_ns is None


async def test_model_tool_model_loop_preserves_order_and_usage_response():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 2, "right": 3}),)),
        AssistantMessage(content="5"),
    )
    events = [
        event
        async for event in (await Agent.create(model, tools=[add], config=CONFIG)).stream("Add 2 and 3", config=CONFIG)
    ]
    assert [event.type for event in events] == [
        AgentEventType.MODEL_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_STARTED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.MODEL_STARTED,
        AgentEventType.REASONING_STARTED,
        AgentEventType.REASONING_DELTA,
        AgentEventType.REASONING_COMPLETED,
        AgentEventType.CONTENT_STARTED,
        AgentEventType.TEXT_DELTA,
        AgentEventType.CONTENT_COMPLETED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.RUN_COMPLETED,
    ]
    assert events[-1].message.content == "5"
    assert {event.session_id for event in events} == {CONFIG.session_id}
    assert isinstance(model.requests[1].messages[-1], ToolMessage)
    assert model.requests[1].messages[-1].content == "5"
    assert model.requests[1].messages[-2].tool_calls[0].id == "c1"


async def test_run_returns_a_plain_assistant_message():
    model = ScriptedModel(AssistantMessage(content="Hello"))
    reply = await (await Agent.create(model, config=CONFIG)).run(
        "Hi", config=CONFIG, reasoning_effort=ReasoningEffort.LOW
    )
    assert reply.content == "Hello"
    assert model.requests[0].reasoning_effort == ReasoningEffort.LOW


async def test_requests_inherit_agent_reasoning_effort_and_allow_per_run_override():
    model = ScriptedModel(
        AssistantMessage(content="Default"),
        AssistantMessage(content="Override"),
        AssistantMessage(content="Default again"),
    )
    agent = await Agent.create(
        model,
        config=CONFIG,
        reasoning_effort=ReasoningEffort.HIGH,
    )

    await agent.run("First")
    _ = [event async for event in agent.stream("Second", reasoning_effort=ReasoningEffort.LOW)]
    await agent.run("Third", reasoning_effort=None)

    assert [request.reasoning_effort for request in model.requests] == [
        ReasoningEffort.HIGH,
        ReasoningEffort.LOW,
        ReasoningEffort.HIGH,
    ]
    assert agent.reasoning_effort == ReasoningEffort.HIGH


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("c1", "missing", {}),
        ToolCall("c1", "add", {"left": "invalid", "right": 1}),
    ],
)
async def test_tool_error_returns_to_model(call):
    model = ScriptedModel(AssistantMessage(tool_calls=(call,)), AssistantMessage(content="Please clarify."))
    events = [
        event async for event in (await Agent.create(model, tools=[add], config=CONFIG)).stream("Do it", config=CONFIG)
    ]
    assert any(event.type == AgentEventType.TOOL_FAILED for event in events)
    assert model.requests[1].messages[-1].success is False
    assert events[-1].message.content == "Please clarify."


async def test_after_tool_can_inspect_exception_and_modify_result_before_append():
    class ToolFailure(RuntimeError):
        def __init__(self) -> None:
            super().__init__("public failure")
            self.detail = {"retryable": True, "status": 503}

    failure = ToolFailure()

    @tool(guidelines="Use to exercise failure normalization.")
    def fail() -> None:
        """Always fail with structured runtime details."""
        raise failure

    class NormalizeFailure(AgentExtension):
        appended: ToolMessage | None = None

        async def after_tool(self, context, call, result, error):
            assert error is failure
            assert result not in context.state.messages
            result.content = '{"error":"temporarily unavailable","retryable":true}'
            result.attributes["status"] = failure.detail["status"]

        async def on_event(self, context, event):
            if isinstance(event, MessageAppendedEvent) and isinstance(event.message, ToolMessage):
                self.appended = event.message

    extension = NormalizeFailure()
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "fail", {}),)),
        AssistantMessage(content="Try again later."),
    )
    events = [
        event
        async for event in (await Agent.create(model, tools=[fail], extensions=[extension], config=CONFIG)).stream(
            "Do it", config=CONFIG
        )
    ]

    result = next(event.message for event in events if event.type is AgentEventType.TOOL_FAILED)
    assert result is extension.appended
    assert result.attributes == {"status": 503}
    assert model.requests[1].messages[-1] is result
    assert model.requests[1].messages[-1].content == '{"error":"temporarily unavailable","retryable":true}'


async def test_agent_state_retains_messages_between_runs():
    history = [UserMessage(content="Old question"), AssistantMessage(content="Old reply")]
    model = ScriptedModel(AssistantMessage(content="New reply"), AssistantMessage(content="Other reply"))

    class Restore(AgentExtension):
        async def on_state(self, context):
            context.state.messages.extend(history)

    accumulator = InMemoryMessageAccumulator()
    agent = await Agent.create(model, system_prompt="", extensions=[Restore(), accumulator], config=CONFIG)
    await agent.run(UserMessage(content="Latest"), config=CONFIG)
    await agent.run("Follow up", config=CONFIG)
    assert [item.content for item in model.requests[0].messages] == ["Old question", "Old reply", "Latest"]
    assert [item.content for item in model.requests[1].messages] == [
        "Old question",
        "Old reply",
        "Latest",
        "New reply",
        "Follow up",
    ]
    assert len(history) == 2


async def test_tool_guidance_follows_all_instructions_without_duplication():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 1, "right": 2}),)),
        AssistantMessage(content="3"),
    )

    class Restore(AgentExtension):
        async def on_state(self, context):
            context.state.messages.append(SystemMessage(content="Snapshot"))

    agent = await Agent.create(
        model,
        system_prompt="Base",
        tools=[add],
        extensions=[Restore(), ToolGuidelinesExtension()],
        config=CONFIG,
    )
    await agent.run("Add", config=CONFIG)
    for request in model.requests:
        prompt = "\n\n".join(message.content for message in request.messages if isinstance(message, SystemMessage))
        assert prompt.startswith("Base\n\nSnapshot")
        assert prompt.endswith("## add\n- Use for exact addition.")
        assert prompt.count("# Tool guidelines") == 1
    assert sum("# Tool guidelines" in message.content for message in agent.state.messages) == 1


async def test_iteration_limit_is_explicit():
    model = ScriptedModel(AssistantMessage(tool_calls=(ToolCall("c", "add", {"left": 1, "right": 2}),)))
    with pytest.raises(AgentIterationLimitError):
        await (await Agent.create(model, tools=[add], max_iterations=1, config=CONFIG)).run("Add", config=CONFIG)


@pytest.mark.parametrize("extra_response", [False, True])
async def test_invalid_model_stream_fails(extra_response):
    class InvalidModel:
        async def stream(self, request):
            if extra_response:
                yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))
            yield ModelEvent.text("unexpected")

    with pytest.raises(AgentProtocolError):
        await (await Agent.create(InvalidModel(), config=CONFIG)).run("Hi", config=CONFIG)


async def test_task_cancellation_closes_provider_stream():
    waiting, closed = asyncio.Event(), asyncio.Event()

    class SlowModel:
        async def stream(self, request):
            try:
                yield ModelEvent.text("partial")
                waiting.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

    task = asyncio.create_task((await Agent.create(SlowModel(), config=CONFIG)).run("Hi", config=CONFIG))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_agent_rejects_a_concurrent_request_while_generating():
    started, release = asyncio.Event(), asyncio.Event()

    class BlockingModel:
        async def stream(self, request):
            started.set()
            await release.wait()
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Done")))

    agent = await Agent.create(BlockingModel(), config=CONFIG)
    active_request = asyncio.create_task(agent.run("First"))
    await asyncio.wait_for(started.wait(), timeout=2)

    assert agent.state.phase == AgentPhase.GENERATING
    with pytest.raises(AgentProtocolError, match="current phase is 'generating'"):
        await agent.run("Second")

    release.set()
    assert (await active_request).content == "Done"
    assert agent.state.phase == AgentPhase.COMPLETED


async def test_duplicate_tools_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        (await Agent.create(ScriptedModel(), tools=[add, add], config=CONFIG))


def test_agent_config_rejects_an_empty_session_id():
    with pytest.raises(ValueError, match="session_id"):
        AgentRunConfig(session_id="  ")

    with pytest.raises(ValueError, match="request_id"):
        AgentRunConfig(session_id="session", request_id="  ")


async def test_agent_rejects_invalid_initialization_transitions():
    with pytest.raises(ValueError, match="max_iterations"):
        Agent(ScriptedModel(), max_iterations=0)

    agent = Agent(ScriptedModel())
    await agent.initialize(config=CONFIG)
    with pytest.raises(AgentProtocolError, match="another session"):
        await agent.initialize(config=AgentRunConfig("other-session"))


async def test_extensions_receive_all_success_hooks_and_can_modify_messages():
    calls: list[str] = []

    class RecordingExtension(AgentExtension):
        async def on_tool(self, context):
            calls.append("on_tool")

        async def on_state(self, context):
            calls.append("on_state")
            context.state.messages.append(SystemMessage(content=f"Loaded {context.config.session_id}"))

        async def on_message(self, context):
            calls.append("on_message")

        async def before_run(self, context):
            calls.append("before_run")
            assert context.state.messages[-1].content == "Add"

        async def before_model(self, context, request):
            calls.append("before_model")

        async def after_model(self, context, response):
            calls.append("after_model")

        async def before_tool(self, context, call):
            calls.append("before_tool")

        async def after_tool(self, context, call, result, error):
            calls.append("after_tool")
            assert error is None

        async def after_run(self, context, result):
            calls.append("after_run")

        async def on_success(self, context, result):
            calls.append("on_success")
            assert result.content == "5"

    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c1", "add", {"left": 2, "right": 3}),)),
        AssistantMessage(content="5"),
    )
    agent = await Agent.create(model, tools=[add], extensions=[RecordingExtension()], config=CONFIG)
    await agent.run("Add", config=CONFIG)

    assert calls == [
        "on_tool",
        "on_state",
        "on_message",
        "before_run",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "before_model",
        "after_model",
        "after_run",
        "on_success",
    ]
    assert [message.content for message in model.requests[0].messages[:2]] == [
        "You are a helpful assistant.",
        "Loaded test-session",
    ]


async def test_message_injection_runs_for_each_fresh_request_state():
    class Instructions(AgentExtension):
        calls = 0

        async def on_state(self, context):
            self.calls += 1
            context.state.messages.insert(1, SystemMessage(content="Extra instructions"))

    extension = Instructions()
    model = ScriptedModel(AssistantMessage(content="First"), AssistantMessage(content="Second"))
    agent = Agent(model, system_prompt="Base", extensions=[extension])
    assert agent.state.messages == []
    await agent.initialize(config=CONFIG)
    await agent.initialize(config=CONFIG)
    assert extension.calls == 0
    await agent.run("Hello", config=CONFIG)
    await agent.run("Continue", config=CONFIG)
    assert extension.calls == 2
    assert model.requests[1].messages[:2] == (
        SystemMessage(content="Base"),
        SystemMessage(content="Extra instructions"),
    )


async def test_extension_receives_errors():
    errors: list[Exception] = []

    class EmptyModel:
        async def stream(self, request):
            if False:
                yield

    class ErrorExtension(AgentExtension):
        async def on_error(self, context, error):
            errors.append(error)

    with pytest.raises(AgentProtocolError):
        await (await Agent.create(EmptyModel(), extensions=[ErrorExtension()], config=CONFIG)).run(
            "Fail", config=CONFIG
        )
    assert len(errors) == 1


async def test_in_memory_accumulator_reuses_messages_across_agent_instances():
    accumulator = InMemoryMessageAccumulator()
    first_model = ScriptedModel(AssistantMessage(content="First reply"))
    await (await Agent.create(first_model, system_prompt="", extensions=[accumulator], config=CONFIG)).run(
        "First", config=CONFIG
    )

    second_model = ScriptedModel(AssistantMessage(content="Second reply"))
    second_agent = await Agent.create(second_model, system_prompt="", extensions=[accumulator], config=CONFIG)
    await second_agent.run("Second", config=CONFIG)

    assert [message.content for message in second_model.requests[0].messages] == [
        "First",
        "First reply",
        "Second",
    ]
    assert [message.content for message in accumulator.messages(CONFIG.session_id)] == [
        "First",
        "First reply",
        "Second",
        "Second reply",
    ]
    accumulator.clear(CONFIG.session_id)
    assert accumulator.messages(CONFIG.session_id) == ()


async def test_in_memory_accumulator_rebuilds_current_system_instructions():
    accumulator = InMemoryMessageAccumulator()
    await (
        await Agent.create(
            ScriptedModel(AssistantMessage(content="First reply")),
            system_prompt="Old instructions",
            extensions=[accumulator],
            config=CONFIG,
        )
    ).run("First")

    model = ScriptedModel(AssistantMessage(content="Second reply"))
    agent = await Agent.create(
        model,
        system_prompt="Current instructions",
        extensions=[accumulator],
        config=CONFIG,
    )
    await agent.run("Second")

    instructions = [message.content for message in model.requests[0].messages if isinstance(message, SystemMessage)]
    assert instructions == ["Current instructions"]
    assert all(not isinstance(message, SystemMessage) for message in accumulator.messages(CONFIG.session_id))


async def test_in_memory_accumulator_retains_tool_calls_and_results():
    accumulator = InMemoryMessageAccumulator()
    first_model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("call-1", "add", {"left": 1, "right": 2}),)),
        AssistantMessage(content="Three"),
    )
    first = await Agent.create(
        first_model,
        system_prompt="",
        tools=[add],
        extensions=[accumulator],
        config=CONFIG,
    )
    await first.run("Add the numbers")

    remembered = accumulator.messages(CONFIG.session_id)
    assert isinstance(remembered[1], AssistantMessage)
    assert remembered[1].tool_calls == (ToolCall("call-1", "add", {"left": 1, "right": 2}),)
    assert isinstance(remembered[2], ToolMessage)
    assert remembered[2].tool_call_id == "call-1"

    next_model = ScriptedModel(AssistantMessage(content="Continued"))
    second = await Agent.create(next_model, system_prompt="", extensions=[accumulator], config=CONFIG)
    await second.run("Continue")
    assert next_model.requests[0].messages[:-1] == remembered
