"""Priority inbox for extension-generated instructions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from ..exceptions import AgentProtocolError
from ..messages import AgentMessage
from .base import AgentExtension
from .events import ExtensionEvent, InternalMessageEvent
from .external import ExternalEvent

if TYPE_CHECKING:
    from ..agent import AgentConfig, AgentContext

INTERNAL_MESSAGE_EVENT_NAME = "internal_message"


@dataclass(slots=True)
class _Inbox:
    context: AgentContext
    messages: deque[AgentMessage] = field(default_factory=deque)
    accepting: bool = True


class InternalMessageExtension(AgentExtension):
    """Receive internal instructions from publish() or external event delivery.

    Examples:
        Usage::

            await context.publish(InternalMessageEvent(AgentMessage(content="Check the result.")))

    External callers use name="internal_message" and a payload containing
    session_id, content, and optionally request_id. The Agent owns the per-run
    processing limit. Each new user input starts a separate run or stream.
    """

    def __init__(self) -> None:
        self._inboxes: dict[AgentContext, _Inbox] = {}
        self._lock = Lock()

    def open(self, context: AgentContext) -> None:
        """Open a request inbox before lifecycle hooks run."""
        with self._lock:
            if context in self._inboxes:
                raise AgentProtocolError("Internal inbox is already active")
            self._inboxes[context] = _Inbox(context)

    def close(self, context: AgentContext) -> None:
        """Release queued input on every exit, including broken cleanup hooks."""
        with self._lock:
            self._inboxes.pop(context, None)

    def accept(self, config: AgentConfig | None, event: ExternalEvent) -> bool:
        """Accept internal text for an unambiguously identified active request."""
        if event.name != INTERNAL_MESSAGE_EVENT_NAME:
            return False
        if config is None:
            return False
        session_id = config.session_id
        request_id = config.request_id
        content = event.payload.get("content")
        if not isinstance(session_id, str) or not session_id.strip():
            return False
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
            return False
        if not isinstance(content, str) or not content.strip():
            return False
        with self._lock:
            matches = [
                inbox
                for inbox in self._inboxes.values()
                if inbox.accepting
                and inbox.context.config.session_id == session_id
                and (request_id is None or inbox.context.config.request_id == request_id)
            ]
            if len(matches) != 1:
                return False
            matches[0].messages.append(AgentMessage(content=content))
            return True

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Append a typed internal instruction to its originating request."""
        if isinstance(event, InternalMessageEvent):
            with self._lock:
                inbox = self._inboxes.get(context)
                if inbox is None or not inbox.accepting:
                    raise AgentProtocolError("Internal inbox is closed")
                inbox.messages.append(event.message)

    def next_message(self, context: AgentContext) -> AgentMessage | None:
        """Take the next internal instruction, or atomically close an empty inbox."""
        with self._lock:
            inbox = self._inboxes[context]
            if inbox.messages:
                return inbox.messages.popleft()
            inbox.accepting = False
            return None
