"""Tool prompt guidance exposed as an Agent extension."""

from ..agent import AgentRunContext
from ..messages import SystemMessage
from ..tools.base import render_tool_guidance
from .base import AgentExtension


class ToolGuidelinesExtension(AgentExtension):
    """Inject registered tool snippets and guidelines into system instructions.

    Tools marked ``deferred=True`` are skipped: their definitions reach the model
    through tool search, so rendering their guidance would announce a tool that
    is not callable yet.
    """

    async def on_state(self, context: AgentRunContext) -> None:
        """Place guidance after existing system instructions and before dialogue."""
        state = context.state
        visible = tuple(candidate for candidate in context.tools.values() if not candidate.deferred)
        message = SystemMessage(content=render_tool_guidance(visible))
        if message.content and message not in state.messages:
            instructions = [message for message in state.messages if isinstance(message, SystemMessage)]
            context.add_message(message, index=len(instructions))
