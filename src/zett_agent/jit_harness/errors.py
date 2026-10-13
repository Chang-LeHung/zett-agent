"""Bounded recent tool-error diagnostics, separate from payload-free profiling."""

from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from hashlib import sha256
from threading import RLock
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from .._compat import UTC
from .enums import ToolCallStatus

if TYPE_CHECKING:
    from .profiling import ToolCallProfile


class ToolErrorRecord(BaseModel):
    """One distinct recent error, with the latest invocation's context.

    Errors deduplicate by tool name, exception type, and complete message.
    Reoccurrence refreshes recency and last_seen_at; reading does not.
    occurrences covers only the current cache residence, restarting if an
    evicted error reappears. message is truncated to the configured character
    limit and message_truncated says whether the full text was longer; None
    means the exception could not be formatted. No exception object, traceback,
    arguments, or result object is retained. Exception messages themselves may
    contain sensitive application data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    error_type: str
    message: str | None
    message_truncated: bool
    occurrences: int
    last_seen_at: datetime
    last_call_id: str
    status: ToolCallStatus
    #: Tool whose handler composed the latest occurrence; null at the top level.
    caller_tool_name: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: ToolCallStatus) -> ToolCallStatus:
        """Keep the cache limited to the two outcomes it exists to explain."""
        if value not in (ToolCallStatus.FAILED, ToolCallStatus.REJECTED):
            raise ValueError("Error records can only be failed or rejected")
        return value


class ToolErrorReport(BaseModel):
    """A read-only page of retained diagnostics, newest occurrence first.

    capacity bounds distinct errors across the whole harness, not per tool.
    retained_errors is the current global cache size; total_matches counts
    retained entries matching the optional tool-name filter. Pagination is a
    live view: export after a request for stable pages. Empty pages do not prove
    that a tool never failed; older diagnostics may have been evicted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    errors: tuple[ToolErrorRecord, ...]
    capacity: int
    max_message_chars: int
    retained_errors: int
    total_matches: int
    offset: int
    next_offset: int | None


class ToolErrorCache:
    """Internal: LRU-bounded recent-error cache owned by one profiler."""

    def __init__(self, capacity: int, max_message_chars: int) -> None:
        for value in (capacity, max_message_chars):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Error capacity and message character limit must be positive integers")
        self.capacity = capacity
        self.max_message_chars = max_message_chars
        self._lock = RLock()
        self._entries: OrderedDict[str, ToolErrorRecord] = OrderedDict()

    def remember(self, call: ToolCallProfile, message: str | None) -> None:
        if call.status not in (ToolCallStatus.FAILED, ToolCallStatus.REJECTED) or call.error_type is None:
            return
        # Hash full text, not its prefix: different long errors must not merge.
        key = sha256(json.dumps([call.tool_name, call.error_type, message], ensure_ascii=True).encode()).hexdigest()
        size = len(message) if message is not None else None
        now = datetime.now(UTC)
        with self._lock:
            previous = self._entries.get(key)
            self._entries[key] = ToolErrorRecord(
                tool_name=call.tool_name,
                error_type=call.error_type,
                message=message[: self.max_message_chars] if message is not None else None,
                message_truncated=size is not None and size > self.max_message_chars,
                occurrences=previous.occurrences + 1 if previous is not None else 1,
                last_seen_at=now,
                last_call_id=call.call_id,
                status=call.status,
                caller_tool_name=call.caller_tool_name,
            )
            self._entries.move_to_end(key)
            if len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def report(self, name: str | None, offset: int, limit: int) -> ToolErrorReport:
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("Error offset must be nonnegative and limit must be between 1 and 1000")
        with self._lock:
            records = [
                record for record in reversed(self._entries.values()) if name is None or record.tool_name == name
            ]
            page = records[offset : offset + limit]
            end = offset + len(page)
            return ToolErrorReport(
                errors=tuple(page),
                capacity=self.capacity,
                max_message_chars=self.max_message_chars,
                retained_errors=len(self._entries),
                total_matches=len(records),
                offset=offset,
                next_offset=end if end < len(records) else None,
            )
