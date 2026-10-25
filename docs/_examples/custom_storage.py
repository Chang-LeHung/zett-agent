"""Implement the storage contract from scratch: one JSON file, no database.

The base persistence extension already knows when to load, append, and snapshot.
An adapter only answers those three questions, which this file does with a dictionary
and a JSON file so the mechanics stay visible.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast

from zett_agent.agent import AgentRunConfig
from zett_agent.client import create_agent
from zett_agent.extensions.compaction import CompactedMessage
from zett_agent.extensions.events import MessageTiming
from zett_agent.extensions.persistence import (
    BaseSessionPersistenceExtension,
    ContextSnapshot,
    RawMessageRecord,
    SessionView,
)
from zett_agent.ids import new_uuid7
from zett_agent.json_types import JsonValue
from zett_agent.messages import AnyMessage, AssistantMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, ModelUsage, RetryOptions
from zett_agent.storage import decode_messages, encode_messages


class _StoredMessage(TypedDict):
    """One stored message: the fields a replay needs."""

    id: str
    request_id: str
    sequence: int
    message: str
    metadata: dict[str, JsonValue]
    tags: dict[str, JsonValue]
    started_at: str
    completed_at: str
    duration_ns: int


class _StoredSession(TypedDict):
    """One stored session: its messages and latest checkpoint."""

    messages: list[_StoredMessage]
    snapshot: dict[str, JsonValue] | None
    parent_session_id: str | None
    session_type: int
    title: str | None
    agent_name: str | None


class JsonFileStorage:
    """Store every session in one JSON file, with the three required operations.

    The file holds, per session, the immutable messages, the latest checkpoint, and
    the few display attributes. Appends allocate a per-session sequence; a snapshot
    records the inclusive sequence it replaces and advances the version.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        text = path.read_text() if path.exists() else "{}"
        self._data = cast("dict[str, _StoredSession]", json.loads(text))

    # -- helpers ---------------------------------------------------------------------------

    def _session(self, session_id: str) -> _StoredSession:
        return self._data.setdefault(
            session_id,
            {
                "messages": [],
                "snapshot": None,
                "parent_session_id": None,
                "session_type": 0,
                "title": None,
                "agent_name": None,
            },
        )

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False))

    # -- the contract ----------------------------------------------------------------------

    async def load(self, session_id: str) -> SessionView:
        """Return the latest checkpoint and every message after its boundary."""
        data = self._session(session_id)
        snapshot = ContextSnapshot.model_validate(data["snapshot"]) if data["snapshot"] else None
        boundary = snapshot.compacted_through_sequence if snapshot is not None else 0
        tail = [
            RawMessageRecord(
                id=item["id"],
                session_id=session_id,
                request_id=item["request_id"],
                sequence=item["sequence"],
                message=decode_messages(item["message"])[0],
                metadata=item["metadata"],
                tags=item["tags"],
                started_at=datetime.fromisoformat(item["started_at"]),
                completed_at=datetime.fromisoformat(item["completed_at"]),
                duration_ns=item["duration_ns"],
                reasoning_started_at=None,
                reasoning_completed_at=None,
                reasoning_duration_ns=None,
                content_started_at=None,
                content_completed_at=None,
                content_duration_ns=None,
                input_tokens=None,
                output_tokens=None,
                cache_read_tokens=None,
                cache_write_tokens=None,
                reasoning_tokens=None,
                created_at=datetime.fromisoformat(item["completed_at"]),
                updated_at=datetime.fromisoformat(item["completed_at"]),
            )
            for item in data["messages"]
            if item["sequence"] > boundary
        ]
        return SessionView(
            parent_session_id=data["parent_session_id"],
            session_type=data["session_type"],
            title=data["title"],
            agent_name=data["agent_name"],
            snapshot=snapshot,
            raw_tail=tail,
        )

    async def append(
        self,
        session_id: str,
        request_id: str,
        message: AnyMessage,
        timing: MessageTiming | None = None,
        parent_session_id: str | None = None,
        title: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tags: Mapping[str, JsonValue] | None = None,
        usage: ModelUsage | None = None,
    ) -> int:
        """Append one immutable message and return its new sequence number."""
        data = self._session(session_id)
        if data["parent_session_id"] is None:
            data["parent_session_id"] = parent_session_id
            data["title"] = title
            data["agent_name"] = agent_name
        moment = datetime.now(timezone.utc)
        started = timing.started_at if timing is not None else moment
        completed = timing.completed_at if timing is not None else moment
        sequence = len(data["messages"]) + 1
        data["messages"].append(
            {
                "id": new_uuid7(),
                "request_id": request_id,
                "sequence": sequence,
                "message": encode_messages([message]),
                "metadata": dict(metadata or {}),
                "tags": dict(tags or {}),
                "started_at": started.isoformat(),
                "completed_at": completed.isoformat(),
                "duration_ns": timing.duration_ns if timing is not None else 0,
            }
        )
        self._save()
        return sequence

    async def snapshot(
        self,
        session_id: str,
        compacted_message: CompactedMessage,
        compacted_through_sequence: int,
        expected_version: int,
    ) -> ContextSnapshot:
        """Create one checkpoint, refusing a stale version."""
        data = self._session(session_id)
        current = ContextSnapshot.model_validate(data["snapshot"]).version if data["snapshot"] else 0
        if expected_version != current:
            raise ValueError(f"Snapshot version changed: expected {expected_version}, stored {current}")
        sequences = [item["sequence"] for item in data["messages"]]
        if compacted_through_sequence > (max(sequences) if sequences else 0):
            raise ValueError("compacted_through_sequence is past the end of the Raw Log")
        moment = datetime.now(timezone.utc)
        snapshot = ContextSnapshot(
            id=new_uuid7(),
            session_id=session_id,
            version=current + 1,
            compacted_message=compacted_message,
            compacted_through_sequence=compacted_through_sequence,
            created_at=moment,
            updated_at=moment,
        )
        data["snapshot"] = snapshot.model_dump(mode="json")
        self._save()
        return snapshot


class JsonPersistence(BaseSessionPersistenceExtension[JsonFileStorage]):
    """Only choose the storage; the base owns every lifecycle hook."""

    def __init__(self, path: Path) -> None:
        super().__init__(JsonFileStorage(path))


class FixedModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Saved in JSON.")))


async def main() -> None:
    with TemporaryDirectory(prefix="zett-custom-storage-") as directory:
        path = Path(directory) / "sessions.json"
        extension = JsonPersistence(path)
        client = await create_agent(FixedModel(), config=AgentRunConfig(session_id="json"), extensions=[extension])

        await client.run("Remember this")
        # Reopen from disk: a new instance only has the file.
        reopened = JsonFileStorage(path)
        view = await reopened.load("json")
        print(f"Stored {len(view.raw_tail)} messages; reopened {len(view.raw_tail)} from a new instance")
        assert [record.message.role.value for record in view.raw_tail] == ["system", "user", "assistant"]


if __name__ == "__main__":
    asyncio.run(main())
