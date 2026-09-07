"""Urgent user input consumed at model/tool completion boundaries."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from ..exceptions import AgentProtocolError
from ..messages import UserMessage
from .base import AgentExtension
from .events import ExtensionEvent, SteeringMessageEvent
from .external import ExternalEvent

if TYPE_CHECKING:
    from ..agent import AgentConfig, AgentContext

STEERING_MESSAGE_EVENT_NAME = "steering_message"


@dataclass(slots=True)
class _Inbox:
    context: AgentContext
    messages: deque[UserMessage] = field(default_factory=deque)
    accepting: bool = True


class SteeringExtension(AgentExtension):
    """Agent-owned urgent user inbox, checked after each model and tool call.

    Steering replaces the continuation of the current task. It does not cancel
    an in-flight model or tool. Unexecuted tool calls receive explicit skipped
    results before the new user message is appended. Internal messages remain
    queued until steering has been answered. No internal-message quota applies.

    Example:
        agent.emit_external_event(
            ExternalEvent("steering_message", {"content": "Stop editing; explain first."}),
            config=AgentConfig(session_id="s1"),
        )

        await context.publish(SteeringMessageEvent(UserMessage(content="Inspect another file.")))
    """

    def __init__(self) -> None:
        self._inboxes: dict[AgentContext, _Inbox] = {}
        self._lock = Lock()

    def open(self, context: AgentContext) -> None:
        """Initialize request-local routing before extension hooks run."""
        with self._lock:
            if context in self._inboxes:
                raise AgentProtocolError("Steering inbox is already active")
            self._inboxes[context] = _Inbox(context)

    def close(self, context: AgentContext) -> None:
        """Clear routing and messages on every request exit."""
        with self._lock:
            self._inboxes.pop(context, None)

    def accept(self, config: AgentConfig | None, event: ExternalEvent) -> bool:
        """Route external text by session and optional request ID, from any thread."""
        if event.name != STEERING_MESSAGE_EVENT_NAME:
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
            matches[0].messages.append(UserMessage(content=content))
            return True

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Receive typed user input, including multimodal UserMessages."""
        if isinstance(event, SteeringMessageEvent):
            with self._lock:
                inbox = self._inboxes.get(context)
                if inbox is None or not inbox.accepting:
                    raise AgentProtocolError("Steering inbox is closed")
                inbox.messages.append(event.message)

    def take(self, context: AgentContext) -> UserMessage | None:
        """Poll after a model/tool call without closing an empty inbox."""
        with self._lock:
            inbox = self._inboxes[context]
            return inbox.messages.popleft() if inbox.messages else None

    def select_next(self, context: AgentContext, reserve_other_input: Callable[[], bool]) -> UserMessage | None:
        """Return only steering input; atomically close when no input remains.

        The caller owns other input and reserves it through a synchronous
        callback, returning True when it has work to process. No other message
        type enters or leaves this extension. The callback must not call back
        into steering or invoke hooks: the steering lock is held throughout
        selection so an accepted message cannot be lost at request completion.
        """
        with self._lock:
            inbox = self._inboxes[context]
            if inbox.messages:
                return inbox.messages.popleft()
            if not reserve_other_input():
                inbox.accepting = False
            return None
