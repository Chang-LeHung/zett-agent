"""Shared handler middleware for model calls and Python tool composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AgentTool, ToolResult

if TYPE_CHECKING:
    from ..agent import AgentRunContext
    from ..extensions.base import AgentExtension, ToolCallNext
    from ..messages import ToolCall


async def invoke_tool(
    context: AgentRunContext, call: ToolCall, registered: AgentTool, *, refresh_registration: bool = True
) -> ToolResult:
    """Execute handler middleware without emitting events or serializing results.

    Lifecycle hooks, invocation identity, scheduling, and result persistence
    belong to the caller. Composed tools receive the raw Python result.
    Set refresh_registration=False when the supplied invocation boundary already
    resolves the live registry itself; this preserves nested harness checks even
    when base registrations are intentionally left unwrapped.
    """

    async def invoke_handler() -> ToolResult:
        # Hooks/middleware can replace a registration before dispatch. Prefer
        # its current wrapper. If removed, keep the original invocation boundary:
        # harness wrappers report unloaded/policy-removed tools explicitly.
        current = context.tools.get(call.name, registered) if refresh_registration else registered
        return await current(call.arguments)

    call_next: ToolCallNext = invoke_handler
    for extension in reversed(context.extensions):
        inner = call_next

        async def wrapped(*, hook: AgentExtension = extension, next_handler: ToolCallNext = inner) -> ToolResult:
            return await hook.on_tool_call(context, call, next_handler)

        call_next = wrapped
    return await call_next()
