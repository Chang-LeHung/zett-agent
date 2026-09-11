from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic_ns
from typing import Self

from ..events import AgentPhase
from ..messages import AgentMessage, AnyMessage, AssistantMessage, UserMessage
from ..model import ModelUsage


@dataclass(frozen=True, slots=True)
class ExtensionEvent:
    """Immutable notification broadcast or routed to a named extension."""


@dataclass(frozen=True, slots=True)
class MessageTiming:
    """Persistable timing measurements for one immutable Raw Log message.

    UTC timestamps describe when work happened and can be displayed or compared
    across processes. Durations are measured with a monotonic clock so wall-clock
    corrections cannot produce negative or inaccurate elapsed times. Monotonic
    clock readings themselves are intentionally not persisted.
    """

    #: UTC time when processing began. For assistant and tool messages this is
    #: the transition into GENERATING and RUNNING_TOOL, respectively. A directly
    #: appended or user-authored message uses its append time.
    started_at: datetime
    #: UTC time when processing ended. For assistant and tool messages this is
    #: the transition back to READY after the operation finished.
    completed_at: datetime
    #: Total elapsed processing time in nanoseconds, calculated from monotonic
    #: clock readings rather than by subtracting the UTC timestamps.
    duration_ns: int
    #: UTC arrival time of the first non-empty reasoning delta. None when the
    #: model did not stream reasoning for this message.
    reasoning_started_at: datetime | None = None
    #: UTC time when reasoning gave way to content, a tool call, or the final
    #: response. None when no reasoning segment was observed.
    reasoning_completed_at: datetime | None = None
    #: Elapsed reasoning time in nanoseconds measured with a monotonic clock.
    #: None when no complete reasoning segment was observed.
    reasoning_duration_ns: int | None = None
    #: UTC arrival time of the first non-empty answer-content delta. None for
    #: messages without streamed answer content, including tool messages.
    content_started_at: datetime | None = None
    #: UTC time of the final model response after streamed content. None when no
    #: content segment was observed.
    content_completed_at: datetime | None = None
    #: Elapsed answer-content streaming time in nanoseconds measured with a
    #: monotonic clock. None when no complete content segment was observed.
    content_duration_ns: int | None = None

    def __post_init__(self) -> None:
        if self.duration_ns < 0:
            raise ValueError("duration_ns cannot be negative")
        self._validate_optional_span(
            "reasoning",
            self.reasoning_started_at,
            self.reasoning_completed_at,
            self.reasoning_duration_ns,
        )
        self._validate_optional_span(
            "content",
            self.content_started_at,
            self.content_completed_at,
            self.content_duration_ns,
        )

    @staticmethod
    def _validate_optional_span(
        name: str,
        started_at: datetime | None,
        completed_at: datetime | None,
        duration_ns: int | None,
    ) -> None:
        values = (started_at, completed_at, duration_ns)
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError(f"{name} timing fields must be supplied together")
        if duration_ns is not None and duration_ns < 0:
            raise ValueError(f"{name}_duration_ns cannot be negative")

    @classmethod
    def instant(cls) -> "MessageTiming":
        """Represent a message that required no measured processing interval."""
        now = datetime.now(UTC)
        return cls(started_at=now, completed_at=now, duration_ns=0)


@dataclass(frozen=True, slots=True)
class MessageAppendedEvent(ExtensionEvent):
    """One newly produced raw message; restoration and compaction do not emit it."""

    message: AnyMessage
    timing: MessageTiming
    #: Provider-reported usage for an assistant model response. User, system,
    #: tool, and directly imported messages do not consume a model response.
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        if self.usage is not None and not isinstance(self.message, AssistantMessage):
            raise ValueError("Model usage belongs only to assistant messages")


@dataclass(frozen=True, slots=True)
class InternalMessageEvent(ExtensionEvent):
    """Request that the active Agent process one internal instruction.

    Extensions publish this event through ``AgentContext.publish()``. The
    built-in InternalMessageExtension receives it and queues the message without
    coupling the producing extension to the Agent loop.
    """

    message: AgentMessage

    def __post_init__(self) -> None:
        if not isinstance(self.message, AgentMessage):
            raise TypeError("InternalMessageEvent requires an AgentMessage")


@dataclass(frozen=True, slots=True)
class SteeringMessageEvent(ExtensionEvent):
    """Urgent user input replacing continuation at the next model/tool boundary."""

    message: UserMessage

    def __post_init__(self) -> None:
        if not isinstance(self.message, UserMessage):
            raise TypeError("SteeringMessageEvent requires a UserMessage")


@dataclass(frozen=True, slots=True)
class PhaseTransitionEvent(ExtensionEvent):
    """One validated request-state transition published after the phase changes.

    ``context.state.phase`` already equals ``current_phase`` while subscribers
    process this event. Rejected transitions never publish an event.
    """

    previous_phase: AgentPhase
    current_phase: AgentPhase
    #: UTC wall-clock time suitable for persistence, logs, and user interfaces.
    occurred_at: datetime
    #: Process-local monotonic reading used only to calculate elapsed durations.
    monotonic_ns: int


@dataclass(frozen=True, slots=True)
class RunCancelledEvent(ExtensionEvent):
    """A request entered CANCELLED; delivered after its PhaseTransitionEvent.

    This internal notification also works when the stream consumer has closed.
    It does not suppress CancelledError or GeneratorExit.
    """

    #: Active operation interrupted by cancellation.
    previous_phase: AgentPhase
    #: UTC cancellation time, shared with the corresponding phase transition.
    occurred_at: datetime
    #: Monotonic cancellation boundary for measuring elapsed time.
    monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ModelOutputLifecycleEvent(ExtensionEvent):
    """A precise boundary within one streamed model response."""

    #: UTC wall-clock time suitable for persistence, logs, and user interfaces.
    occurred_at: datetime
    #: Process-local monotonic reading used only to calculate elapsed durations.
    monotonic_ns: int

    @classmethod
    def now(cls) -> Self:
        """Capture wall and monotonic clocks at the same lifecycle boundary."""
        return cls(occurred_at=datetime.now(UTC), monotonic_ns=monotonic_ns())


@dataclass(frozen=True, slots=True)
class ReasoningStartedEvent(ModelOutputLifecycleEvent):
    """The first non-empty reasoning delta of one model call arrived."""


@dataclass(frozen=True, slots=True)
class ReasoningCompletedEvent(ModelOutputLifecycleEvent):
    """The reasoning segment of one model call ended."""


@dataclass(frozen=True, slots=True)
class ContentStartedEvent(ModelOutputLifecycleEvent):
    """The first non-empty answer-content delta of one model call arrived."""


@dataclass(frozen=True, slots=True)
class ContentCompletedEvent(ModelOutputLifecycleEvent):
    """The answer-content segment of one model call ended."""


@dataclass(frozen=True, slots=True)
class CompactionEvent(ExtensionEvent):
    """A completed context compaction.

    Ranges are one-based and inclusive in the non-system message list immediately
    before this compaction. A previous checkpoint counts as one message. These
    are context positions, not durable raw-log IDs or user-turn numbers.
    """

    #: First and last messages folded into the checkpoint.
    compressed_from: int
    compressed_to: int
    #: First and last messages retained without modification.
    kept_from: int
    kept_to: int
    #: Exact checkpoint content now present in the active context.
    summary: str
