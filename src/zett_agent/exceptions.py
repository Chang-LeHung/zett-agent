from __future__ import annotations

from typing import TYPE_CHECKING

from .stop import StopInfo

if TYPE_CHECKING:
    from .messages import AssistantMessage


class AgentError(Exception):
    """Base exception for runtime and provider errors."""


class AgentProtocolError(AgentError):
    """A model stream did not produce exactly one final response."""


class AgentIterationLimitError(AgentError):
    """The model/tool loop exceeded its configured limit."""


class AgentStopError(AgentError):
    """A model step ended for a reason other than finishing, and the caller asked to be told.

    Raised by :meth:`~zett_agent.messages.AssistantMessage.raise_for_stop`. The runtime
    itself never raises it: a refusal or a truncated answer is a normal outcome that
    arrives as a message and an event, so a host that does not look still gets the
    conversation back.

    Attributes:
        stop: The :class:`~zett_agent.stop.StopInfo` that explains it.
        message: The assistant message that carried it, with whatever text it had.
    """

    def __init__(self, stop: StopInfo, message: AssistantMessage | None = None) -> None:
        super().__init__(f"The model stopped: {stop.describe()}")
        self.stop = stop
        self.message = message


class AgentResumeError(AgentError):
    """A session has nothing that can be resumed, or what it has cannot be picked up again.

    Raised by :meth:`~zett_agent.agent.Agent.resume`. :meth:`~zett_agent.agent.Agent.resume_point`
    reports the same cases as ``None``, so a host can decide whether to offer a resume
    before it asks for one.

    Attributes:
        reason: Why, as one of ``"empty"`` (no messages), ``"completed"`` (the last request
            finished), ``"refused"`` (the model declined, and the same input would be declined
            again), ``"context_full"`` (the context window was full), ``"unknown_stop"`` (the
            last response stopped for a reason this library does not know) or ``"malformed"``
            (the history ends in a tool result that answers nothing).
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
