"""Tool prompt guidance exposed as an Agent extension."""

from ..agent import AgentRunContext
from ..messages import SystemMessage
from ..tools import render_tool_guidance
from .base import AgentExtension


class ToolGuidelinesExtension(AgentExtension):
    """Inject registered tool snippets and guidelines into system instructions."""

    async def on_state(self, context: AgentRunContext) -> None:
        """Place guidance after existing system instructions and before dialogue."""
        state = context.state
        message = SystemMessage(content=render_tool_guidance(tuple(context.tools.values())))
        if message.content and message not in state.messages:
            instructions = [message for message in state.messages if isinstance(message, SystemMessage)]
            context.add_message(message, index=len(instructions))
