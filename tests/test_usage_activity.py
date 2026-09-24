from datetime import date, datetime

from zett_agent import (
    AgentRunConfig,
    AgentRunContext,
    AgentState,
    AssistantMessage,
    ModelResponse,
    ModelUsage,
    UsageActivityExtension,
)
from zett_agent._compat import UTC


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
        }
    ]
    assert storage.records[0].total_tokens == 150
    assert await extension.activity(start=date(2026, 9, 1), end=date(2026, 9, 30)) == []
