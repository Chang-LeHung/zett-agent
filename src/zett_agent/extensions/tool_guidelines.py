"""Tool prompt guidance exposed as an Agent extension."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace

from .._compat import StrEnum
from ..agent import AgentRunContext
from ..messages import SystemMessage
from ..model import ModelEvent, ModelRequest, ToolDefinition
from ..tools.base import render_tool_description, render_tool_guidance
from .base import AgentExtension, ModelRequestNext


class ToolGuidancePlacement(StrEnum):
    """Where a tool's snippet and guidelines are shown to the model."""

    #: Two sections of the system instructions: ``# Tool snippets`` and ``# Tool guidelines``.
    SYSTEM_PROMPT = "system_prompt"
    #: Appended to each tool's own ``description`` in the request's tool definitions.
    DESCRIPTION = "description"


class ToolGuidelinesExtension(AgentExtension):
    """Show registered tool snippets and guidelines to the model.

    By default the guidance becomes system instructions, grouped so the model
    can scan every call shape first. Set ``placement`` to
    :attr:`ToolGuidancePlacement.DESCRIPTION` to attach each tool's snippet and
    guidelines to that tool's own description instead; the system prompt then
    carries nothing. Use it for endpoints and gateways that surface only tool
    definitions, or when the guidance should travel with the tool wherever the
    definition goes.

    Tools marked ``deferred=True`` are skipped in both modes: their definitions
    reach the model through tool search, so rendering their guidance would
    announce a tool that is not callable yet. Tool search appends their
    guidelines to the definition it returns.

    Args:
        placement: Where the guidance goes; the system prompt by default.
        snippets: Include each tool's ``Snippet`` section.
        guidelines: Include each tool's ``Guidelines`` section. Placement into
            descriptions renders only the parts enabled here.

    Examples:
        Keep guidance in the tool definitions::

            agent = await Agent.create(
                model,
                extensions=[
                    CodingExtension(),
                    ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION),
                ],
            )

    .. note::
        Description placement edits only the request sent to the provider. The
        registered tools, stored history, and the persisted system messages are
        unchanged, so switching placement between runs never rewrites a session.
    """

    def __init__(
        self,
        *,
        placement: ToolGuidancePlacement = ToolGuidancePlacement.SYSTEM_PROMPT,
        snippets: bool = True,
        guidelines: bool = True,
    ) -> None:
        self.placement = ToolGuidancePlacement(placement)
        self.snippets = snippets
        self.guidelines = guidelines

    @property
    def in_system_prompt(self) -> bool:
        """Whether this extension writes guidance into the system instructions."""
        return self.placement is ToolGuidancePlacement.SYSTEM_PROMPT

    async def on_state(self, context: AgentRunContext) -> None:
        """Place guidance after existing system instructions and before dialogue."""
        if not self.in_system_prompt:
            return
        state = context.state
        visible = tuple(candidate for candidate in context.tools.values() if not candidate.deferred)
        message = SystemMessage(
            content=render_tool_guidance(visible, snippets=self.snippets, guidelines=self.guidelines),
            attributes={"tool_guidelines": True},
        )
        if message.content and message not in state.messages:
            instructions = [message for message in state.messages if isinstance(message, SystemMessage)]
            context.add_message(message, index=len(instructions))

    async def on_model_request(
        self,
        context: AgentRunContext,
        request: ModelRequest,
        call_next: ModelRequestNext,
    ) -> AsyncIterator[ModelEvent]:
        """Fold guidance into tool descriptions when that placement is selected."""
        if not self.in_system_prompt:
            request = replace(
                request, tools=tuple(self._described(context, definition) for definition in request.tools)
            )
        async with aclosing(call_next(request)) as events:
            async for event in events:
                yield event

    def _described(self, context: AgentRunContext, definition: ToolDefinition) -> ToolDefinition:
        """Return one tool definition whose description includes its guidance."""
        registered = context.tools.get(definition.name)
        if registered is None or registered.deferred:
            return definition
        # Build on the description the request already carries: another
        # extension (tool search adds a catalog to its own tool) may have
        # appended to it, and replacing it would discard that text.
        description = render_tool_description(
            registered, snippet=self.snippets, guidelines=self.guidelines, base=definition.description
        )
        return replace(definition, description=description)
