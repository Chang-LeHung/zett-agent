import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentEventType,
    AgentExtension,
    AgentPhase,
    AgentProtocolError,
    AgentState,
    AssistantMessage,
    CompactionEvent,
    CompactionExtension,
    ModelEvent,
    ModelEventType,
    ModelResponse,
    SystemMessage,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
    UserMessage,
)
from zett_agent.compaction import CompactedMessage


class SummaryModel:
    def __init__(self, text="Remember the blue notebook."):
        self.text = text
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.text(self.text)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=self.text)))


class Collector(AgentExtension):
    def __init__(self):
        self.events = []

    async def on_event(self, context, event):
        self.events.append(event)


def context(messages):
    return AgentContext(AgentConfig(session_id="test"), AgentState(messages=messages), {}, (Collector(),))


async def compact(extension, state):
    return [event async for event in extension.before_model_events(state)]


async def test_compaction_preserves_instructions_and_whole_tool_turn():
    model = SummaryModel()
    instructions = SystemMessage(content="Base instructions")
    old = [UserMessage(content="Old " * 500), AssistantMessage(content="Old answer")]
    recent = [
        UserMessage(content="Latest"),
        AssistantMessage(tool_calls=(ToolCall("c1", "lookup"),)),
        ToolMessage(tool_call_id="c1", name="lookup", content="Result"),
    ]
    state = context([instructions, *old, *recent])
    original_list = state.state.messages
    extension = CompactionExtension(model, max_tokens=100, keep_recent_tokens=1)
    events = await compact(extension, state)
    assert state.state.messages is original_list
    assert state.state.messages[0] is instructions
    assert isinstance(state.state.messages[1], CompactedMessage)
    assert state.state.messages[2:] == recent
    assert model.requests[0].messages[1:-1] == tuple(old)
    record = state.extensions[0].events[0]
    assert (record.compressed_from, record.compressed_to) == (1, 2)
    assert (record.kept_from, record.kept_to) == (3, 5)
    assert not model.requests[0].tools
    # No completed new turn: do not summarize the same checkpoint again.
    await compact(extension, state)
    assert len(model.requests) == 1
    # Next turn: fold the previous checkpoint and newly old messages together.
    checkpoint = state.state.messages[1]
    state.state.messages.extend([AssistantMessage(content="More " * 300), UserMessage(content="Next")])
    await compact(extension, state)
    assert model.requests[1].messages[1] is checkpoint
    assert sum(isinstance(m, CompactedMessage) for m in state.state.messages) == 1
    assert len(state.extensions[0].events) == 2
    assert events[0].type == AgentEventType.COMPACTION_STARTED
    assert events[-1].type == AgentEventType.COMPACTION_COMPLETED


@pytest.mark.parametrize("limit,recent", [(100000, 1), (1, 20)])
async def test_no_model_call_below_threshold_or_without_old_turns(limit, recent):
    model = SummaryModel()
    state = context([UserMessage(content="Hello"), AssistantMessage(content="Hi")])
    await compact(CompactionExtension(model, max_tokens=limit, keep_recent_tokens=recent), state)
    assert model.requests == []


@pytest.mark.parametrize("summary", ["", "Huge " * 1000])
async def test_invalid_or_larger_summary_leaves_context_unchanged(summary):
    model = SummaryModel(summary)
    messages = [UserMessage(content="Old " * 100), AssistantMessage(content="Answer"), UserMessage(content="New")]
    state = context(messages.copy())
    extension = CompactionExtension(model, max_tokens=1, keep_recent_tokens=1)
    if not summary:
        with pytest.raises(AgentProtocolError):
            await compact(extension, state)
    else:
        await compact(extension, state)
    assert state.state.messages == messages
    assert state.extensions[0].events == []


@pytest.mark.parametrize("fail", [False, True])
async def test_compaction_event_delivered_after_context_update(fail):
    observed = []

    class Observer(AgentExtension):
        async def on_event(self, context, event):
            match event:
                case CompactionEvent():
                    assert any(
                        isinstance(message, CompactedMessage) and message.content == event.summary
                        for message in context.state.messages
                    )
                    observed.append(event)

    class PrimaryModel:
        async def stream(self, request):
            if fail:
                raise RuntimeError("Model failed")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Done")))

    class Restore(AgentExtension):
        async def on_message(self, context):
            context.state.messages.extend([UserMessage(content="Old " * 500), AssistantMessage(content="Answer")])

    agent = await Agent.create(
        PrimaryModel(),
        extensions=[Restore(), CompactionExtension(SummaryModel(), max_tokens=100, keep_recent_tokens=1), Observer()],
        config=AgentConfig(session_id="test"),
    )
    streamed = []
    if fail:
        with pytest.raises(RuntimeError, match="Model failed"):
            async for event in agent.stream("Latest", config=AgentConfig(session_id="test")):
                streamed.append(event)
    else:
        streamed = [event async for event in agent.stream("Latest", config=AgentConfig(session_id="test"))]
    assert len(observed) == 1
    assert observed[0].kept_from == observed[0].kept_to == 3
    assert [event.type for event in streamed[:3]] == [
        AgentEventType.COMPACTION_STARTED,
        AgentEventType.COMPACTION_TEXT_DELTA,
        AgentEventType.COMPACTION_COMPLETED,
    ]
    assert [event.phase for event in streamed[:3]] == [
        AgentPhase.COMPACTING,
        AgentPhase.COMPACTING,
        AgentPhase.READY,
    ]
    assert streamed[2].compaction is observed[0]
    assert streamed[2].applied is True


async def test_recent_retention_uses_token_budget_not_message_count():
    def count_tokens(messages):
        return sum(len(message.content.split()) for message in messages)

    model = SummaryModel("Short checkpoint")
    older = [UserMessage(content="old " * 100), AssistantMessage(content="answer")]
    recent = [UserMessage(content="recent " * 20), AssistantMessage(content="reply"), UserMessage(content="latest")]
    state = context([*older, *recent])
    await compact(CompactionExtension(model, max_tokens=100, keep_recent_tokens=15, count_tokens=count_tokens), state)
    assert state.state.messages[1:] == recent
    assert model.requests[0].messages[1:-1] == tuple(older)


@pytest.mark.parametrize("max_tokens,keep_recent_tokens", [(0, 1), (1, 0)])
def test_compaction_rejects_nonpositive_token_limits(max_tokens, keep_recent_tokens):
    with pytest.raises(ValueError, match="positive"):
        CompactionExtension(
            SummaryModel(),
            max_tokens=max_tokens,
            keep_recent_tokens=keep_recent_tokens,
        )


async def test_compaction_skips_a_checkpoint_without_new_completed_turns():
    model = SummaryModel()
    state = context([CompactedMessage(content="Existing checkpoint"), UserMessage(content="Current request")])
    events = await compact(
        CompactionExtension(model, max_tokens=1, keep_recent_tokens=1, count_tokens=len),
        state,
    )
    assert events == []
    assert model.requests == []


async def test_compaction_streams_reasoning_before_text_and_completion():
    class ReasoningSummaryModel:
        async def stream(self, request):
            yield ModelEvent.reasoning("Reviewing history.")
            yield ModelEvent.text("Short checkpoint")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Short checkpoint")))

    state = context(
        [UserMessage(content="Old request"), AssistantMessage(content="Old response"), UserMessage(content="Current")]
    )
    events = await compact(
        CompactionExtension(ReasoningSummaryModel(), max_tokens=2, keep_recent_tokens=1, count_tokens=len),
        state,
    )
    assert [event.type for event in events] == [
        AgentEventType.COMPACTION_STARTED,
        AgentEventType.COMPACTION_REASONING_DELTA,
        AgentEventType.COMPACTION_TEXT_DELTA,
        AgentEventType.COMPACTION_COMPLETED,
    ]


@pytest.mark.parametrize(
    "mode",
    ["after_response", "missing_response", "tool_delta", "no_response", "tool_response"],
)
async def test_compaction_rejects_invalid_model_protocol_without_rewriting_context(mode):
    class InvalidSummaryModel:
        async def stream(self, request):
            match mode:
                case "after_response":
                    yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Summary")))
                    yield ModelEvent.text("late")
                case "missing_response":
                    yield ModelEvent(ModelEventType.RESPONSE)
                case "tool_delta":
                    yield ModelEvent.tool_call(ToolCallDelta(0, name_delta="read_file"))
                case "no_response":
                    return
                case "tool_response":
                    yield ModelEvent.completed(
                        ModelResponse(AssistantMessage(tool_calls=(ToolCall("call", "read_file"),)))
                    )

    messages = [
        UserMessage(content="Old request"),
        AssistantMessage(content="Old response"),
        UserMessage(content="Current"),
    ]
    state = context(messages.copy())
    with pytest.raises(AgentProtocolError):
        await compact(
            CompactionExtension(InvalidSummaryModel(), max_tokens=2, keep_recent_tokens=1, count_tokens=len), state
        )
    assert state.state.messages == messages
    assert state.extensions[0].events == []
