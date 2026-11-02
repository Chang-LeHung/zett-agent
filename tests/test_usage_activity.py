from datetime import date, datetime

from zett_agent._compat import UTC
from zett_agent.agent import (
    Agent,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
)
from zett_agent.extensions.usage_activity import UsageActivityExtension
from zett_agent.messages import AssistantMessage
from zett_agent.model import (
    ModelEvent,
    ModelResponse,
    ModelUsage,
)


class Storage:
    def __init__(self):
        self.records = []

    async def record(self, record):
        self.records.append(record)

    async def activity(self, *, start, end):
        return []


async def test_usage_activity_extension_records_each_model_response():
    storage = Storage()
    occurred_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    extension = UsageActivityExtension(storage, clock=lambda: occurred_at)
    context = AgentRunContext(
        AgentRunConfig(session_id="session", request_id="request"),
        AgentState(),
        {},
    )

    await extension.after_model(
        context,
        ModelResponse(
            AssistantMessage(content="Done", provider="deepseek", model="deepseek-chat"),
            usage=ModelUsage(
                input_tokens=120,
                output_tokens=30,
                cache_read_tokens=80,
                cache_write_tokens=5,
                reasoning_tokens=10,
                ttft_any_ns=180_000_000,
                ttft_content_ns=940_000_000,
                first_chunk="Thinking",
                first_content_chunk="Done",
            ),
        ),
    )

    assert [record.model_dump() for record in storage.records] == [
        {
            "session_id": "session",
            "request_id": "request",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "occurred_at": occurred_at,
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_tokens": 80,
            "cache_write_tokens": 5,
            "reasoning_tokens": 10,
            "ttft_any_ns": 180_000_000,
            "ttft_content_ns": 940_000_000,
            "first_chunk": "Thinking",
            "first_content_chunk": "Done",
        }
    ]
    assert storage.records[0].total_tokens == 150
    assert await extension.activity(start=date(2026, 9, 1), end=date(2026, 9, 30)) == []


async def test_the_extension_receives_the_ttft_the_runtime_measured():
    class StreamingModel:
        async def stream(self, request):
            yield ModelEvent.reasoning("Plan")
            yield ModelEvent.text("Done")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Done", provider="deepseek")))

    storage = Storage()
    agent = await Agent.create(
        StreamingModel(),
        config=AgentRunConfig("streamed"),
        extensions=[UsageActivityExtension(storage)],
    )
    await agent.run("hi")

    record = storage.records[-1]
    assert record.first_chunk == "Plan"
    assert record.first_content_chunk == "Done"
    assert record.ttft_any_ns is not None
    assert record.ttft_content_ns is not None
    assert 0 <= record.ttft_any_ns <= record.ttft_content_ns


async def test_usage_activity_leaves_stream_timings_unknown_for_a_silent_response():
    storage = Storage()
    extension = UsageActivityExtension(storage)
    context = AgentRunContext(
        AgentRunConfig(session_id="session", request_id="request"),
        AgentState(),
        {},
    )

    await extension.after_model(
        context,
        ModelResponse(AssistantMessage(content="Done", provider="deepseek", model="deepseek-chat")),
    )

    record = storage.records[0]
    assert record.ttft_any_ns is None
    assert record.ttft_content_ns is None
    assert record.first_chunk is None
    assert record.first_content_chunk is None
