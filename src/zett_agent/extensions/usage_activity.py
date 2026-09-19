"""Record provider token usage as daily activity without coupling storage backends."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..agent import AgentRunContext
from ..model import ModelResponse
from .base import AgentExtension


class ModelUsageActivityRecord(BaseModel):
    """One completed model request recorded for activity visualization."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    request_id: str | None
    provider: str | None
    model: str | None
    occurred_at: datetime
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        """Return the tokens represented by this request."""
        return self.input_tokens + self.output_tokens


class ModelUsageActivityDay(BaseModel):
    """Aggregated model usage for one UTC calendar day."""

    model_config = ConfigDict(frozen=True)

    date: date
    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        """Return input plus output tokens for this day."""
        return self.input_tokens + self.output_tokens


class ModelUsageActivityStorage(Protocol):
    """Persistence boundary consumed by :class:`UsageActivityExtension`."""

    async def record(self, record: ModelUsageActivityRecord) -> None:
        """Persist one completed model request."""

    async def activity(self, *, start: date, end: date) -> list[ModelUsageActivityDay]:
        """Return daily aggregates in the inclusive date range."""


class UsageActivityExtension(AgentExtension):
    """Record every completed model request through an injected storage adapter."""

    def __init__(
        self,
        storage: ModelUsageActivityStorage,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage = storage
        self.clock = clock or (lambda: datetime.now(UTC))

    async def after_model(self, context: AgentRunContext, response: ModelResponse) -> None:
        """Persist provider-reported usage for one completed primary model call."""
        usage = response.usage
        await self.storage.record(
            ModelUsageActivityRecord(
                session_id=context.config.session_id,
                request_id=context.config.request_id,
                provider=response.message.provider,
                model=response.message.model,
                occurred_at=self.clock(),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
        )

    async def activity(self, *, start: date, end: date) -> list[ModelUsageActivityDay]:
        """Read daily activity through the configured storage."""
        return await self.storage.activity(start=start, end=end)
