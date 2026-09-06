"""In-memory conversation accumulation exposed as an Agent extension."""

from ..agent import AgentContext
from ..messages import AnyMessage, SystemMessage
from .base import AgentExtension
from .events import CompactionEvent, ExtensionEvent, MessageAppendedEvent


class InMemoryMessageAccumulator(AgentExtension):
    """Retain non-system conversation messages for reuse by later requests.

    System instructions belong to the current Agent configuration and are rebuilt
    for every request. Keeping them here would accumulate stale prompts whenever
    an Agent or its tool guidance changes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[AnyMessage]] = {}

    async def on_message(self, context: AgentContext) -> None:
        """Combine current instructions with an independent copy of remembered dialogue."""
        state = context.state
        instructions = [message for message in state.messages if isinstance(message, SystemMessage)]
        current_dialogue = [message for message in state.messages if not isinstance(message, SystemMessage)]
        remembered = self._sessions.get(context.config.session_id)
        if remembered is None:
            remembered = list(current_dialogue)
            self._sessions[context.config.session_id] = remembered
        state.messages[:] = [*instructions, *remembered]

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Accumulate raw messages and replace dialogue after successful compaction."""
        session_id = context.config.session_id
        match event:
            case MessageAppendedEvent(message=SystemMessage()):
                return
            case MessageAppendedEvent(message=message):
                self._sessions.setdefault(session_id, []).append(message)
            case CompactionEvent():
                self._sessions[session_id] = [
                    message for message in context.state.messages if not isinstance(message, SystemMessage)
                ]

    def messages(self, session_id: str) -> tuple[AnyMessage, ...]:
        """Return remembered dialogue without request-specific system instructions."""
        return tuple(self._sessions.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        """Forget one session without changing other accumulated sessions."""
        self._sessions.pop(session_id, None)
