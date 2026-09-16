"""Validated construction settings for the ZettCode runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from zett_agent import ReasoningEffort


class ProviderName(StrEnum):
    """Provider adapters exposed by the initial CLI."""

    DEEPSEEK = "deepseek"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class ZettCodeConfig:
    """All process-level settings needed to construct one coding runtime."""

    workspace: Path
    provider: ProviderName
    model: str
    api_key: str
    database: Path
    session_id: str | None = None
    base_url: str | None = None
    responses_api: bool = False
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    parallel_tool_call: bool = True
    max_iterations: int = 36
    compaction_max_tokens: int = 128_000
    compaction_keep_tokens: int = 32_000

    def __post_init__(self) -> None:
        workspace = self.workspace.expanduser().resolve()
        database = self.database.expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {workspace}")
        if not self.model.strip():
            raise ValueError("Model cannot be empty")
        if not self.api_key.strip():
            raise ValueError(f"Missing API key for {self.provider.value}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.compaction_keep_tokens < 1:
            raise ValueError("compaction_keep_tokens must be positive")
        if self.compaction_max_tokens <= self.compaction_keep_tokens:
            raise ValueError("compaction_max_tokens must be greater than compaction_keep_tokens")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "database", database)


def provider_api_key(provider: ProviderName) -> str:
    """Read the selected provider credential without logging it."""
    match provider:
        case ProviderName.DEEPSEEK:
            return os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API", "")
        case ProviderName.OPENAI:
            return os.getenv("OPENAI_API_KEY", "")


def default_model(provider: ProviderName) -> str:
    """Return a conservative default model for one provider."""
    match provider:
        case ProviderName.DEEPSEEK:
            return "deepseek-chat"
        case ProviderName.OPENAI:
            return "gpt-5-mini"
