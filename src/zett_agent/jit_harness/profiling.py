"""Payload-free session profiling for the harness's shared dispatch boundary."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter_ns
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..messages import ImageContent, TextContent
from .enums import ToolCallStatus, ToolOrigin
from .errors import ToolErrorCache, ToolErrorReport
from .pages import MAX_MEASURED_CHARS

#: Finished call records kept per harness, oldest dropped first. Totals survive
#: eviction, so frequent tools still report lifetime aggregates.
MAX_RETAINED_CALLS = 2_000

_JSON: TypeAdapter[Any] = TypeAdapter(Any)


def _measured(size: int) -> int:
    """Clamp one measurement to the harness's measurement budget."""
    return size if size <= MAX_MEASURED_CHARS else MAX_MEASURED_CHARS


def _lower_bound(value: Any) -> int | None:
    """Sum characters without serializing, stopping once the budget is exceeded.

    Serializing a large payload costs milliseconds on the dispatching thread, and
    the exact length above the measurement budget cannot change a decision. This
    walk is O(items), allocates nothing, and returns None for values whose length
    only the exact serializer can know. MAX_MEASURED_CHARS means "at least this
    many", so the caller can stop without serializing.
    """
    total = 0
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
        elif item is None or isinstance(item, bool | int | float):
            total += 1
        elif isinstance(item, bytes | bytearray):
            total += len(item)
        elif isinstance(item, dict):
            # Every key and value costs at least one JSON character, so a
            # container larger than the remaining budget is already over it.
            if len(item) > MAX_MEASURED_CHARS - total:
                return MAX_MEASURED_CHARS
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            if len(item) > MAX_MEASURED_CHARS - total:
                return MAX_MEASURED_CHARS
            stack.extend(item)
        else:
            return None
        if total > MAX_MEASURED_CHARS:
            return MAX_MEASURED_CHARS
    return total


def _characters(value: Any, *, output: bool = False) -> int | None:
    """Measure characters without retaining data or failing a tool invocation.

    Measurements stop at the harness measurement budget, so a reported
    ``MAX_MEASURED_CHARS`` means "at least this many": beyond it extra precision
    costs time and cannot change a decision. Values whose length no serializer
    can know stay unknown instead of being reported as zero.
    """
    try:
        if output:
            if isinstance(value, str):
                return _measured(len(value))
            if isinstance(value, TextContent):
                return _measured(len(value.text))
            if isinstance(value, ImageContent):
                return 0
            if isinstance(value, list) and all(isinstance(part, (TextContent, ImageContent)) for part in value):
                text = sum(len(part.text) for part in value if isinstance(part, TextContent))
                return _measured(text) if value else 2
        if (cheap := _lower_bound(value)) is not None and cheap >= MAX_MEASURED_CHARS:
            return MAX_MEASURED_CHARS
        encoded = _JSON.dump_python(value, mode="json")
        serialized = json.dumps(encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return _measured(len(serialized))
    except Exception:
        # Arbitrary manually supplied tools can return non-JSON Python objects.
        # Unknown length is not zero, and telemetry must not break execution.
        return None


class ToolCallProfile(BaseModel):
    """One finished dispatch attempt, without argument or result contents.

    ``call_id`` is the join key: a model-requested call keeps the conversation's
    ToolCall id, a nested call carries the id the harness assigned, and the same
    value appears in the JIT CUSTOM events and recent-error records.

    Character counts use compact Unicode JSON for arguments/structured results,
    plain string length for text results, and text-block lengths for multimodal
    results. Image data and URLs are excluded. None means unavailable, not zero.
    Counts stop at the harness measurement budget, so a value equal to it means
    "at least this many". Duration includes validation and nested execution, but
    excludes surrounding middleware, lifecycle hooks, and telemetry measurement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Invocation id, shared with the conversation ToolCall and the JIT events.
    call_id: str
    #: Registered name that was dispatched.
    tool_name: str
    #: Registry that served the call; see ToolOrigin.
    origin: ToolOrigin
    #: How the attempt ended; see ToolCallStatus.
    status: ToolCallStatus
    #: Elapsed dispatch time in milliseconds, measured as this class documents.
    duration_ms: float = Field(ge=0)
    #: Measured argument length, or null when the arguments are not JSON-representable.
    input_chars: int | None = Field(default=None, ge=0)
    #: Result length, or the error-message length after a failure; null when unknown or cancelled.
    output_chars: int | None = Field(default=None, ge=0)
    #: Tool whose handler composed this call; null at the top level.
    caller_tool_name: str | None = None
    #: Exception class name for failures and rejections; the message lives in tool_errors().
    error_type: str | None = None


class ToolProfileStats(BaseModel):
    """Lifetime totals for one tool in one harness, counting finished attempts.

    Averages use measured samples only; unknown lengths never count as zero.
    Parent duration includes child duration, so summing parent and child times
    double-counts nested work. Tool updates preserve these lifetime totals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    #: Finished attempts observed for this tool.
    call_count: int = 0
    #: Finished attempts, one field per ToolCallStatus outcome.
    completed_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    rejected_count: int = 0
    #: Mean finished-attempt timing in milliseconds; null until one finishes.
    average_duration_ms: float | None = None
    #: Mean measured lengths; unmeasurable attempts contribute no sample, so an
    #: average is null when nothing could be measured.
    average_input_chars: float | None = None
    average_output_chars: float | None = None


class JITHarnessProfile(BaseModel):
    """An immutable, JSON-exportable view of session-local profiling data.

    Stats cover finished attempts; in-flight calls are not published. Calls
    contain a page of finished records ordered by dispatch. offset/next_offset
    paginate them. Retained records are bounded per harness, oldest dropped
    first, while lifetime aggregates stay complete. Do not paginate while calls
    are finishing if a stable export is required; export after the request.
    Profiles are not part of executable JITHarnessSnapshot data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: One entry per requested tool name, sorted by name when no filter was given.
    stats: tuple[ToolProfileStats, ...]
    #: This page of finished records, oldest retained dispatch first.
    calls: tuple[ToolCallProfile, ...]
    #: Retained records matching the filter, across every page.
    total_calls: int
    #: Offset this page started at, and the offset to pass next when one exists.
    offset: int
    next_offset: int | None


@dataclass(frozen=True)
class ActiveToolCall:
    """One in-flight dispatch; internal to the profiler."""

    #: Dispatch order, starting at 1; assigned when the attempt starts.
    sequence: int
    #: Invocation identity, later reported as ToolCallProfile.call_id.
    call_id: str
    #: Registered name being dispatched.
    tool_name: str
    #: Registry that resolved the name; see ToolOrigin.
    origin: ToolOrigin
    #: Tool whose handler composed this call, or None at the top level.
    caller_tool_name: str | None
    #: Measured argument length, or None when it is not JSON-representable.
    input_chars: int | None
    #: Monotonic start reading used for the duration; not a wall-clock value.
    started_ns: int


@dataclass(frozen=True)
class RecordedCall:
    """One finished call retained in dispatch order; internal to the profiler."""

    sequence: int
    profile: ToolCallProfile


@dataclass
class ToolProfileTotals:
    """Mutable running totals for one tool, converted by ``report()``.

    Internal to the profiler. Kept separate from ToolProfileStats so
    accumulation stays cheap and the public view is built only when someone
    reads it.
    """

    #: Finished attempts for this tool.
    call_count: int = 0
    #: Finished attempts per status, missing keys meaning zero.
    statuses: dict[ToolCallStatus, int] = field(default_factory=dict)
    #: Sum of finished durations in nanoseconds.
    duration_ns: int = 0
    #: Sum of measured input lengths, and the attempts behind them.
    input_chars: int = 0
    inputs: int = 0
    #: Sum of measured output lengths, and the attempts behind them.
    output_chars: int = 0
    outputs: int = 0

    def report(self, name: str) -> ToolProfileStats:
        finished = self.call_count
        return ToolProfileStats(
            tool_name=name,
            call_count=self.call_count,
            **{f"{status}_count": self.statuses.get(status, 0) for status in ToolCallStatus},
            average_duration_ms=self.duration_ns / finished / 1_000_000 if finished else None,
            average_input_chars=self.input_chars / self.inputs if self.inputs else None,
            average_output_chars=self.output_chars / self.outputs if self.outputs else None,
        )


class ToolProfiler:
    """Session-local measurements; internal to the harness.

    ``call_capacity`` bounds retained finished records exactly as the error cache
    bounds diagnostics: oldest dispatch first, while lifetime aggregates stay
    complete. Nested and concurrent calls finish out of dispatch order, so the
    bounded window is sorted when a report is read.
    """

    def __init__(
        self, *, error_capacity: int = 100, error_message_chars: int = 4000, call_capacity: int = MAX_RETAINED_CALLS
    ) -> None:
        if isinstance(call_capacity, bool) or not isinstance(call_capacity, int) or call_capacity < 1:
            raise ValueError("Call capacity must be a positive integer")
        self._lock = RLock()
        self.call_capacity = call_capacity
        self._sequence = 0
        self._totals: dict[str, ToolProfileTotals] = {}
        self._calls: deque[RecordedCall] = deque(maxlen=call_capacity)
        self._errors = ToolErrorCache(error_capacity, error_message_chars)

    def start(
        self,
        name: str,
        arguments: Any,
        *,
        call_id: str,
        parent: ActiveToolCall | None,
        origin: ToolOrigin,
    ) -> ActiveToolCall:
        with self._lock:
            self._sequence += 1
            return ActiveToolCall(
                self._sequence,
                call_id,
                name,
                origin,
                parent.tool_name if parent else None,
                _characters(arguments),
                perf_counter_ns(),
            )

    def finish(
        self,
        active: ActiveToolCall,
        status: ToolCallStatus,
        output: Any,
        error: BaseException | None,
    ) -> None:
        elapsed = max(0, perf_counter_ns() - active.started_ns)
        message = None
        if status is ToolCallStatus.CANCELLED:
            size = None
        elif error is not None:
            try:
                message = str(error)
                size = len(message)
            except Exception:
                size = None
        else:
            size = _characters(output, output=True)
        profile = ToolCallProfile(
            call_id=active.call_id,
            tool_name=active.tool_name,
            origin=active.origin,
            status=status,
            duration_ms=elapsed / 1_000_000,
            input_chars=active.input_chars,
            output_chars=size,
            caller_tool_name=active.caller_tool_name,
            error_type=type(error).__name__ if error else None,
        )
        with self._lock:
            totals = self._totals.setdefault(active.tool_name, ToolProfileTotals())
            totals.call_count += 1
            totals.statuses[status] = totals.statuses.get(status, 0) + 1
            totals.duration_ns += elapsed
            if active.input_chars is not None:
                totals.input_chars += active.input_chars
                totals.inputs += 1
            if size is not None:
                totals.output_chars += size
                totals.outputs += 1
            self._calls.append(RecordedCall(active.sequence, profile))
            self._errors.remember(profile, message)

    def errors(self, name: str | None, offset: int, limit: int) -> ToolErrorReport:
        return self._errors.report(name, offset, limit)

    def summary(self, name: str) -> ToolProfileStats:
        with self._lock:
            return self._totals.get(name, ToolProfileTotals()).report(name)

    def report(self, name: str | None, offset: int, limit: int) -> JITHarnessProfile:
        if offset < 0 or limit < 1 or limit > 1000:
            raise ValueError("Profiling offset must be nonnegative and limit must be between 1 and 1000")
        with self._lock:
            names = [name] if name is not None else sorted(self._totals)
            # Bounded by call_capacity, so ordering a read stays cheap.
            records = sorted(
                (record for record in self._calls if name is None or record.profile.tool_name == name),
                key=lambda record: record.sequence,
            )
            page = records[offset : offset + limit]
            end = offset + len(page)
            return JITHarnessProfile(
                stats=tuple(self.summary(item) for item in names),
                calls=tuple(record.profile for record in page),
                total_calls=len(records),
                offset=offset,
                next_offset=end if end < len(records) else None,
            )
