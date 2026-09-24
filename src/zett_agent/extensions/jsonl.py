"""Append one self-contained JSON object per Agent turn."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from weakref import WeakKeyDictionary

from .._compat import UTC
from ..agent import AgentRunContext
from ..messages import AnyMessage, AssistantMessage, UserMessage
from .base import AgentExtension
from .events import ExtensionEvent, RunCancelledEvent


@dataclass(frozen=True, slots=True)
class _PendingTurn:
    input_message: UserMessage
    started_at: datetime


class JSONLExtension(AgentExtension):
    """Append each complete Agent turn as one line in a session JSONL file.

    A turn begins with the request's finalized UserMessage and includes every
    assistant, tool, internal-agent, and steering message appended before the
    terminal outcome. Model retries and tool loops remain inside that one line;
    history from earlier turns is not copied again.

    Each record also contains AgentRunContext metadata and tags. Message attributes
    remain attached to their messages. Completed, failed, and cancelled turns
    are recorded, while raw exception objects and provider credentials are not.

    Args:
        directory: Directory that owns the per-session ``.jsonl`` files. Session
            IDs are URL-safe Base64 encoded for canonical, traversal-safe names.

    Examples:
        Usage::

            history = JSONLExtension(".agent-turns")
            agent = await Agent.create(
                model,
                config=AgentRunConfig("session-42"),
                extensions=[history],
            )

            await agent.run(
                "Inspect the failing test",
                metadata={"source": "editor"},
                tags={"intent": "debug"},
            )

            print(history.session_path("session-42"))

    .. note::
        Writes from one extension instance are serialized with a lock. The
        extension does not coordinate independent processes writing the same
        directory.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._pending: WeakKeyDictionary[AgentRunContext, _PendingTurn] = WeakKeyDictionary()
        self._write_lock = Lock()

    def session_path(self, session_id: str) -> Path:
        """Return the canonical JSONL path for one session without creating it."""
        encoded = base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii").rstrip("=")
        return self.directory / f"{encoded}.jsonl"

    async def before_run(self, context: AgentRunContext) -> None:
        """Remember the finalized input that marks the start of this turn."""
        message = context.input_message
        if message is None or not any(item is message for item in context.state.messages):
            raise RuntimeError("JSONL turn input must be appended before before_run")
        self._pending[context] = _PendingTurn(message, datetime.now(UTC))

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Append one completed turn after the final assistant response exists."""
        await self._finalize(context, status="completed")

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Append one failed turn with a safe error summary."""
        await self._finalize(
            context,
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Append a cancelled turn from its dedicated terminal notification."""
        if isinstance(event, RunCancelledEvent):
            await self._finalize(context, status="cancelled", completed_at=event.occurred_at)

    async def _finalize(
        self,
        context: AgentRunContext,
        *,
        status: str,
        error: dict[str, str] | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        pending = self._pending.pop(context, None)
        if pending is None:
            return
        messages = self._turn_messages(context, pending.input_message)
        from ..storage import encode_messages

        record = {
            "version": 1,
            "session_id": context.config.session_id,
            "request_id": context.config.request_id,
            "parent_session_id": context.config.parent_session_id,
            "started_at": pending.started_at.isoformat(),
            "completed_at": (completed_at or datetime.now(UTC)).isoformat(),
            "status": status,
            "metadata": context.metadata,
            "tags": context.tags,
            "messages": json.loads(encode_messages(messages)),
        }
        if error is not None:
            record["error"] = error
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        await asyncio.to_thread(self._append_line, self.session_path(context.config.session_id), line)

    @staticmethod
    def _turn_messages(context: AgentRunContext, input_message: UserMessage) -> list[AnyMessage]:
        for index, message in enumerate(context.state.messages):
            if message is input_message:
                return list(context.state.messages[index:])
        return [input_message]

    def _append_line(self, path: Path, line: str) -> None:
        with self._write_lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.write("\n")
