"""Register one tool as an Agent extension."""

from __future__ import annotations

from ..agent import AgentRunContext
from ..tools.base import AgentTool
from .base import AgentExtension


class ToolExtension(AgentExtension):
    """Register exactly one tool for every request.

    This is the unit of tool composition: one extension, one tool. An
    application that wants only part of a bundle such as
    :class:`~zett_agent.extensions.file_system.FileSystemExtension` adds the
    matching single-tool extensions instead, and a subclass fixes the tool so a
    caller does not pass it. Overriding :meth:`on_tool` is the place to veto or
    wrap the registration, for example to hide the tool in some sessions.

    Args:
        registered: The tool to expose to the model.

    Examples:
        Offer a custom tool without writing an extension::

            agent = await Agent.create(model, extensions=[ToolExtension(lookup_order)])

        Fix the tool in a subclass::

            class LookupOrderExtension(ToolExtension):
                def __init__(self) -> None:
                    super().__init__(lookup_order)

    .. note::
        Registration copies the tool for each request, so the shared
        ``AgentTool`` is never mutated. Prompt guidance for it still needs
        :class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension`.
    """

    def __init__(self, registered: AgentTool) -> None:
        if not isinstance(registered, AgentTool):
            raise TypeError("ToolExtension needs an AgentTool; decorate the function with @tool")
        self._tool = registered

    @property
    def tool(self) -> AgentTool:
        """The tool this extension registers."""
        return self._tool

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """The immutable registration set, always exactly one tool."""
        return (self._tool,)

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register the tool for this request."""
        context.register_tool(self._tool)
