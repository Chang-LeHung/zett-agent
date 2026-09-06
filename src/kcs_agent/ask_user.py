"""Pause an ask_user tool call until an external UI response is accepted."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent import AgentContext
from .events import AgentEvent, AgentEventType
from .external_events import ExternalEventExtension
from .messages import ToolCall
from .tools import AgentTool, tool

ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_EVENT_NAME = "ask_user"
ASK_USER_RESPONSE_EVENT_NAME = "ask_user_response"

Question = Annotated[str, Field(min_length=1, max_length=4_000)]
Option = Annotated[str, Field(min_length=1, max_length=500)]
Options = Annotated[list[Option], Field(max_length=20)]


class AskUserRequest(BaseModel):
    """Validated question rendered by a UI for one ask_user tool call."""

    model_config = ConfigDict(extra="forbid")

    question: Question
    options: Options | None = None
    allow_multiple: bool = False


class AskUserResult(BaseModel):
    """Accepted external response returned to the model as a tool result."""

    name: str
    payload: dict[str, Any]


class AskUserEvent(AgentEvent):
    """CUSTOM AgentEvent requesting user input from a streaming client."""

    __slots__ = ()

    def __init__(self, session_id: str, call: ToolCall, request: AskUserRequest) -> None:
        super().__init__(
            AgentEventType.CUSTOM,
            session_id=session_id,
            tool_calls=[call],
            name=ASK_USER_EVENT_NAME,
            payload={
                "session_id": session_id,
                "tool_call_id": call.id,
                "question": request.question,
                "options": request.options or [],
                "allow_multiple": request.allow_multiple,
                "response_event": ASK_USER_RESPONSE_EVENT_NAME,
            },
        )


class AskUserExtension(ExternalEventExtension):
    """Register ask_user and suspend its execution until accept() receives a reply.

    The normal tool lifecycle remains intact; waiting happens before TOOL_STARTED::

        model emits ask_user ToolCall
                    |
                    v
        before_tool_events()
                    |
                    +--> yield AskUserEvent --> UI
                    |                           |
                    |       WAITING             | ExternalEvent
                    |                           v
                    +<--------------------- accept()
                    |
                    v
        TOOL_STARTED -> ask_user() -> ToolMessage -> next model step

        While WAITING, RunCancelledEvent cancels the Future and on_error()
        completes it with that error. Both paths wake the suspended coroutine.

    Closing or cancelling the stream removes the pending route, so a late UI
    response cannot resume an abandoned request.

    Outbound protocol::

        AgentEvent(
            type=CUSTOM,
            name="ask_user",
            session_id="session-42",
            payload={
                "session_id": "session-42",
                "tool_call_id": "call-7",
                "question": "Which format?",
                "options": ["Markdown", "Plain text"],
                "allow_multiple": False,
                "response_event": "ask_user_response",
            },
        )

    Inbound protocol::

        agent.emit_external_event(ExternalEvent(
            name="ask_user_response",
            payload={
                "session_id": "session-42",
                "tool_call_id": "call-7",
                "answer": "Markdown",
            },
        ))

    ExternalEventExtension owns routing, synchronization, wake-up, and cleanup.
    The response payload is application-defined and returned unchanged to the
    model. Agent.emit_external_event() returns False when no extension accepts
    an unrelated, malformed, duplicate, stale, or cancelled event.
    """

    def __init__(self) -> None:
        super().__init__(
            response_event_name=ASK_USER_RESPONSE_EVENT_NAME,
            correlation_field="tool_call_id",
        )

    async def on_tool(self, context: AgentContext) -> None:
        """Register a request-scoped ask_user tool bound to this context."""
        context.register_tool(self._build_tool(context))

    async def before_tool_events(self, context: AgentContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Emit AskUserEvent, then wait for the matching external response."""
        if call.name != ASK_USER_TOOL_NAME:
            return
        try:
            request = AskUserRequest.model_validate(call.arguments)
        except ValidationError:
            # Let normal tool argument validation produce a failed ToolMessage.
            return

        async with self._wait_for_external_event(context, call.id):
            yield AskUserEvent(context.config.session_id, call, request)

    def _build_tool(self, context: AgentContext) -> AgentTool:
        """Create a validated tool whose result belongs to this request context."""

        @tool(name=ASK_USER_TOOL_NAME)
        async def ask_user(
            question: Question,
            options: Options | None = None,
            allow_multiple: bool = False,
        ) -> AskUserResult:
            """Ask the user a question and wait for an external response.

            Args:
                question: Clear question that the user can answer directly.
                options: Optional choices displayed in their given order.
                allow_multiple: Whether the user may select more than one option.

            Snippet:
                ask_user(question="Which format?", options=["Markdown", "Plain text"])

            Guidelines:
                - Use only when user input is required to continue correctly.
                - Provide short, mutually distinct options when choices are known.
                - Do not ask for information already present in the conversation.
            """
            _ = question, options, allow_multiple
            event = self._take_external_event(context)
            return AskUserResult(name=event.name, payload=event.payload)

        return ask_user
