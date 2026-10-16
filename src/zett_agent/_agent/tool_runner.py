"""Running the tools a model asked for: in parallel or in turn, one call at a time, and recording each result."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from time import monotonic_ns
from typing import TYPE_CHECKING, Any

from .._compat import UTC
from ..events import AgentEventType
from ..messages import AgentMessage, ImageContent, TextContent, ToolCall, ToolMessage, UserMessage
from ..tools.base import (
    AgentTool,
    ToolExecutionMode,
    _bind_tool_call,
    _reset_tool_call,
)
from ..tools.invocation import invoke_tool

if TYPE_CHECKING:
    from ..agent import AgentRunContext
    from ..extensions.events import MessageTiming
    from ..extensions.steering import SteeringExtension


@dataclass(frozen=True, slots=True)
class _ToolInvocation:
    """Completed handler output retained until ordered lifecycle finalization."""

    call: ToolCall
    result: ToolMessage
    error: Exception | None
    timing: MessageTiming


class ToolRunnerMixin:
    """Execute one batch of tool calls and append their results to the context."""

    _steering_extension: SteeringExtension

    async def _execute_tools(
        self,
        context: AgentRunContext,
        calls: Sequence[ToolCall],
        parallel_tool_call: bool,
        active_input: AgentMessage | UserMessage | None = None,
    ) -> UserMessage | None:
        """Run a model's tool calls, parallel ones together first, and stop at a steering message.

        Returns the steering message that interrupted the batch, or None when every call ran.
        """
        parallel_calls: list[ToolCall] = []
        serial_calls: list[ToolCall] = []
        for call in calls:
            registered = context.tools.get(call.name)
            if (
                parallel_tool_call
                and registered is not None
                and registered.execution_mode is ToolExecutionMode.PARALLEL
            ):
                parallel_calls.append(call)
            else:
                serial_calls.append(call)

        if parallel_calls:
            await self._execute_parallel_tools(context, parallel_calls)
            steering = self._steering_extension.take(context)
            if steering is not None:
                await self._start_steering(context, steering, serial_calls, active_input)
                return steering

        return await self._execute_serial_tools(context, serial_calls, active_input)

    async def _begin_tool_execution(self, context: AgentRunContext, calls: Sequence[ToolCall]) -> None:
        """Announce a batch of calls: their hooks, the phase change, and the started event."""
        for call in calls:
            await self._broadcast("before_tool", context, call)
        await self._start_tool_execution(context)
        await context._emit_event(AgentEventType.TOOL_STARTED, tool_calls=list(calls))

    async def _execute_parallel_tools(
        self,
        context: AgentRunContext,
        calls: Sequence[ToolCall],
    ) -> None:
        """Run one parallel batch and emit each result as its handler finishes."""
        await self._begin_tool_execution(context, calls)
        tasks = [asyncio.create_task(self._invoke_tool(context, call)) for call in calls]
        remaining = len(tasks)
        try:
            for completed in asyncio.as_completed(tasks):
                invocation = await completed
                remaining -= 1
                if remaining == 0:
                    await self._finish_tool_execution(context)
                await self._finalize_tool(context, invocation)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _execute_serial_tools(
        self,
        context: AgentRunContext,
        calls: Sequence[ToolCall],
        active_input: AgentMessage | UserMessage | None,
    ) -> UserMessage | None:
        """Run serial calls one at a time and stop at a steering boundary."""
        for index, call in enumerate(calls):
            await self._begin_tool_execution(context, [call])
            invocation = await self._invoke_tool(context, call)
            await self._finish_tool_execution(context)
            await self._finalize_tool(context, invocation)
            steering = self._steering_extension.take(context)
            if steering is not None:
                await self._start_steering(context, steering, calls[index + 1 :], active_input)
                return steering
        return None

    async def _invoke_tool(self, context: AgentRunContext, call: ToolCall) -> _ToolInvocation:
        """Run only a handler concurrently; lifecycle hooks remain serialized."""
        from ..extensions.events import MessageTiming

        started_at = datetime.now(UTC)
        started_ns = monotonic_ns()
        error = None
        token = _bind_tool_call(call.id)
        try:
            try:
                if call.error is not None:
                    raise ValueError(call.error)
                registered = context.tools.get(call.name)
                if registered is None:
                    raise ValueError(f"Unknown tool: {call.name}")
                content = self._tool_content(await invoke_tool(context, call, registered), registered)
            except Exception as tool_error:
                error = tool_error
                content = str(tool_error)
        finally:
            _reset_tool_call(token)
        completed_ns = monotonic_ns()
        completed_at = datetime.now(UTC)
        return _ToolInvocation(
            call=call,
            result=ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                success=error is None,
            ),
            error=error,
            timing=MessageTiming(
                started_at=started_at,
                completed_at=completed_at,
                duration_ns=max(0, completed_ns - started_ns),
            ),
        )

    @staticmethod
    def _tool_content(output: Any, registered: AgentTool) -> Any:
        """Turn what a handler returned into the content of its result message.

        Text and image parts, alone or as a non-empty list of them, are passed through as they are;
        anything else is serialized by the tool.
        """
        if isinstance(output, (ImageContent, TextContent)):
            return [output]
        if (
            isinstance(output, list)
            and output
            and all(isinstance(part, (ImageContent, TextContent)) for part in output)
        ):
            return output
        return registered.serialize_result(output)

    async def _finalize_tool(
        self,
        context: AgentRunContext,
        invocation: _ToolInvocation,
    ) -> None:
        """Publish one completed invocation and append its result to context."""
        call, result, error = invocation.call, invocation.result, invocation.error
        await self._broadcast("after_tool", context, call, result, error)
        await context.append_message(result, invocation.timing)
        await context._emit_event(
            AgentEventType.TOOL_FAILED if error else AgentEventType.TOOL_COMPLETED,
            tool_calls=[call],
            message=result,
            error=error,
        )
