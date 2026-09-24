"""Require external approval before executing shell tool calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from .._compat import StrEnum
from ..agent import AgentRunContext
from ..events import AgentEvent, AgentEventType
from ..messages import ToolCall
from ..tools.base import ToolResult
from .external import ExternalEventExtension

SHELL_APPROVAL_TOOL_NAME = "run_shell"
SHELL_APPROVAL_EVENT_NAME = "shell_approval_requested"
SHELL_APPROVAL_RESPONSE_EVENT_NAME = "shell_approval_response"

ShellCommandNext = Callable[[], Awaitable[ToolResult]]


class ShellApprovalMode(StrEnum):
    """Persisted approval policy for one Agent session."""

    REVIEW = "review"
    ALLOW_ALL = "allow_all"


class ShellApprovalStorage(Protocol):
    """Persistence boundary for session policy and remembered commands."""

    async def get_session_mode(self, session_id: str) -> ShellApprovalMode:
        """Return the persisted approval policy for one session."""

    async def set_session_mode(self, session_id: str, mode: ShellApprovalMode) -> None:
        """Persist the approval policy for one session."""

    async def clear_session_mode(self, session_id: str) -> None:
        """Remove the persisted approval policy for one session."""

    async def is_allowed(self, session_id: str, command: str) -> bool:
        """Return whether one session permits execution of one exact command."""

    async def allow_command(self, command: str) -> None:
        """Persist one command in the allowlist."""


class ShellApprovalResponse(BaseModel):
    """External decision for one pending shell command."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["execute", "abort"]
    remember: bool = False


class ShellCommandAborted(RuntimeError):
    """Raised into normal tool failure handling when the user aborts a command."""


class ShellApprovalRequestedEvent(AgentEvent):
    """CUSTOM event asking a UI to approve or abort one shell command."""

    __slots__ = ()

    def __init__(
        self,
        session_id: str,
        call: ToolCall,
        *,
        command: str,
        timeout_seconds: int,
        remember_supported: bool,
    ) -> None:
        super().__init__(
            AgentEventType.CUSTOM,
            session_id=session_id,
            tool_calls=[call],
            name=SHELL_APPROVAL_EVENT_NAME,
            payload={
                "session_id": session_id,
                "tool_call_id": call.id,
                "command": command,
                "timeout_seconds": timeout_seconds,
                "remember_supported": remember_supported,
                "response_event": SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            },
        )


class ShellApprovalExtension(ExternalEventExtension):
    """Intercept run_shell and pause until a trusted external decision arrives.

    Multiple shell calls may wait concurrently. Each response is correlated by
    its own tool-call ID, so approving one call releases that call immediately
    without waiting for the others.
    """

    def __init__(self, storage: ShellApprovalStorage | None = None, *, enabled: bool = True) -> None:
        super().__init__(
            response_event_name=SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            correlation_field="tool_call_id",
        )
        self.storage = storage
        self.enabled = enabled

    async def on_tool_call(
        self,
        context: AgentRunContext,
        call: ToolCall,
        call_next: ShellCommandNext,
    ) -> ToolResult:
        """Approve one shell call, then execute it or raise a normal tool failure."""
        if not self.enabled or call.name != SHELL_APPROVAL_TOOL_NAME:
            return await call_next()
        command = call.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return await call_next()
        command = command.strip()
        if self.storage is not None and await self.storage.is_allowed(context.config.session_id, command):
            return await call_next()

        timeout_seconds = call.arguments.get("timeout_seconds", 30)
        if not isinstance(timeout_seconds, int):
            timeout_seconds = 30
        async with self._wait_for_external_event(context, call.id):
            await context.emit(
                ShellApprovalRequestedEvent(
                    context.config.session_id,
                    call,
                    command=command,
                    timeout_seconds=timeout_seconds,
                    remember_supported=self.storage is not None,
                )
            )
        event = self._take_external_event(context)
        try:
            response = ShellApprovalResponse.model_validate(event.payload)
        except ValidationError as error:
            raise ValueError("Invalid shell approval response") from error
        if response.decision == "abort":
            raise ShellCommandAborted(f"Shell command aborted by user: {command}")
        if response.remember:
            if self.storage is None:
                raise ValueError("Shell approval storage is not configured")
            await self.storage.allow_command(command)
        return await call_next()
