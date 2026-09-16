"""Application runtime that composes AgentClient with coding extensions."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime

from zett_agent import (
    AgentClient,
    AgentRunConfig,
    CodingExtension,
    CompactionExtension,
    DeepSeekProvider,
    OpenAIProvider,
    SQLiteSessionExtension,
    ToolGuidelinesExtension,
    create_agent,
    new_uuid7,
)

from .config import ProviderName, ZettCodeConfig


def build_system_prompt(config: ZettCodeConfig) -> str:
    """Build request instructions with an explicit workspace boundary."""
    now = datetime.now().astimezone()
    return f"""You are ZettCode, a focused coding agent working with the user in one local workspace.

Inspect relevant files before editing. Make focused changes, preserve unrelated work,
and run checks proportional to risk. Explain blockers concretely. Do not perform
destructive actions unless the user explicitly requests them.

Runtime environment:
- Workspace: {config.workspace}
- Local time: {now.isoformat(timespec="seconds")}
- Operating system: {platform.platform()}
- Architecture: {platform.machine() or "unknown"}
- Python: {platform.python_implementation()} {platform.python_version()}
- Executable: {sys.executable}
- Shell: {os.getenv("SHELL", "unknown")}

Treat the workspace as the current working directory. Prefer relative paths in tool
calls. Verify mutable facts with tools instead of assuming this startup snapshot is current."""


@dataclass(slots=True)
class ZettCodeRuntime:
    """Own the model, persistence extension, AgentClient, and active session."""

    config: ZettCodeConfig
    client: AgentClient
    persistence: SQLiteSessionExtension
    model: OpenAIProvider | DeepSeekProvider
    session_id: str

    @classmethod
    async def create(cls, config: ZettCodeConfig) -> ZettCodeRuntime:
        """Create all owned resources after changing into the chosen workspace."""
        os.chdir(config.workspace)
        config.database.parent.mkdir(parents=True, exist_ok=True)
        persistence = SQLiteSessionExtension(config.database)
        recent = persistence.list_sessions(limit=1)
        session_id = config.session_id or (recent[0].session_id if recent else new_uuid7())
        match config.provider:
            case ProviderName.DEEPSEEK:
                model = DeepSeekProvider(
                    config.model,
                    config.api_key,
                    base_url=config.base_url,
                    response=config.responses_api,
                )
            case ProviderName.OPENAI:
                model = OpenAIProvider(
                    config.model,
                    config.api_key,
                    base_url=config.base_url,
                    response=config.responses_api,
                )
        try:
            client = await create_agent(
                model,
                config=AgentRunConfig(session_id=session_id),
                system_prompt=build_system_prompt(config),
                extensions=[
                    CodingExtension(),
                    persistence,
                    ToolGuidelinesExtension(),
                    CompactionExtension(
                        model,
                        max_tokens=config.compaction_max_tokens,
                        keep_recent_tokens=config.compaction_keep_tokens,
                    ),
                ],
                reasoning_effort=config.reasoning_effort,
                parallel_tool_call=config.parallel_tool_call,
                max_iterations=config.max_iterations,
            )
        except BaseException:
            await model.aclose()
            persistence.close()
            raise
        return cls(config, client, persistence, model, session_id)

    def new_session(self) -> str:
        """Switch future requests to a fresh session identity."""
        self.session_id = new_uuid7()
        return self.session_id

    def use_session(self, session_id: str) -> None:
        """Select a persisted or new explicit session for the next request."""
        if not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        self.session_id = session_id.strip()

    async def aclose(self) -> None:
        """Release all resources owned by this runtime."""
        await self.model.aclose()
        self.persistence.close()
