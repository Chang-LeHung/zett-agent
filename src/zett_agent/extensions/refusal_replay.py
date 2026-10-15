"""Choose whether a refused answer is shown to the model again on later turns."""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection
from contextlib import aclosing
from dataclasses import replace

from .._compat import StrEnum
from ..agent import AgentRunContext
from ..messages import AssistantMessage
from ..model import ModelEvent, ModelRequest
from ..stop import StopReason
from .base import AgentExtension, ModelRequestNext

__all__ = ["RefusalReplay", "RefusalReplayExtension"]


class RefusalReplay(StrEnum):
    """What later requests do with an assistant message that was refused."""

    #: Send it again as it was stored: whatever text came before the refusal, and the
    #: refusal text itself. This is what happens without the extension.
    KEEP = "keep"
    #: Leave it out of the request. It stays in storage and on screen; the model just
    #: no longer reads its own refused output on the next turn.
    DROP = "drop"


class RefusalReplayExtension(AgentExtension):
    """Decide whether the model reads its own refused answers on later turns.

    A refused step can leave text behind: whatever was written before the refusal arrived,
    or the refusal itself. By default that message stays in the history and is sent again
    with every later request, as any other assistant message is. Anthropic's guidance is to
    treat output produced before a refusal as incomplete and discard it, and a history that
    keeps repeating a refusal can make later turns refuse too. This extension is the switch
    for that choice.

    Only the request changes. The message is not removed from stored history, from the
    transcript a user reads, or from ``AgentState.messages``; it is left out of what is
    sent to the model, so changing the setting never rewrites a session.

    Args:
        replay: :attr:`RefusalReplay.KEEP` (the default, which changes nothing) or
            :attr:`RefusalReplay.DROP`.
        reasons: The stop reasons to treat this way. Refusals only by default; add
            :attr:`~zett_agent.stop.StopReason.INTERRUPTED` to drop output a failed stream
            left half written.

    Examples:
        Stop sending refused answers back to the model::

            agent = await create_agent(
                model,
                extensions=[RefusalReplayExtension(replay=RefusalReplay.DROP)],
            )

    .. note::
        Dropping a message can leave two user messages in a row. Providers differ in
        whether they accept that, so this is worth a test against the one you use.
    """

    def __init__(
        self,
        replay: RefusalReplay | str = RefusalReplay.KEEP,
        *,
        reasons: Collection[StopReason] = (StopReason.REFUSAL,),
    ) -> None:
        self.replay = RefusalReplay(replay)
        self.reasons = frozenset(StopReason(reason) for reason in reasons)

    async def on_model_request(
        self,
        context: AgentRunContext,
        request: ModelRequest,
        call_next: ModelRequestNext,
    ) -> AsyncIterator[ModelEvent]:
        """Remove the refused assistant messages from the request when dropping is selected."""
        if self.replay is RefusalReplay.DROP:
            kept = tuple(message for message in request.messages if not self._dropped(message))
            if len(kept) != len(request.messages):
                request = replace(request, messages=kept)
        async with aclosing(call_next(request)) as events:
            async for event in events:
                yield event

    def _dropped(self, message: object) -> bool:
        return (
            isinstance(message, AssistantMessage) and message.stop is not None and message.stop.reason in self.reasons
        )
