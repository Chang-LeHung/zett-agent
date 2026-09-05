from dataclasses import dataclass

from .messages import AnyMessage


@dataclass(frozen=True, slots=True)
class ExtensionEvent:
    """Immutable notification broadcast to registered extensions."""


@dataclass(frozen=True, slots=True)
class MessageAppendedEvent(ExtensionEvent):
    """One newly produced raw message; restoration and compaction do not emit it."""

    message: AnyMessage


@dataclass(frozen=True, slots=True)
class CompactionEvent(ExtensionEvent):
    """A completed context compaction.

    Ranges are one-based and inclusive in the non-system message list immediately
    before this compaction. A previous checkpoint counts as one message. These
    are context positions, not durable raw-log IDs or user-turn numbers.
    """

    # First and last messages folded into the checkpoint.
    compressed_from: int
    compressed_to: int
    # First and last messages retained without modification.
    kept_from: int
    kept_to: int
    # Exact checkpoint content now present in the active context.
    summary: str
