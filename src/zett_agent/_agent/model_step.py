"""One model call: building its request, relaying what it streams, and settling how it stopped."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .._compat import aclosing_stream
from ..events import AgentEventType, ModelOutputTracker
from ..exceptions import AgentProtocolError
from ..messages import AssistantMessage
from ..model import (
    NOT_REPORTED_STATUS,
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelResponse,
    ReasoningEffort,
    ServerToolResult,
)
from ..stop import StopReason, continues_pause
from .contract import AgentContract, mixin_api

if TYPE_CHECKING:
    from ..agent import AgentRunContext

#: How each streamed model event reaches the caller. Every kind but ``RESPONSE`` is shown under the
#: ``AgentEventType`` of the same name, with its payload in the named field. A payload that must
#: be there carries the message to raise when it is missing; a delta may be an empty string.
_RELAYED: dict[ModelEventType, tuple[str, str | None]] = {
    ModelEventType.TEXT_DELTA: ("delta", None),
    ModelEventType.REASONING_DELTA: ("delta", None),
    ModelEventType.TOOL_CALL_DELTA: ("tool_call_delta", "Missing tool-call delta"),
    ModelEventType.SERVER_TOOL_STARTED: ("server_tool_call", "Missing server-tool call"),
    ModelEventType.SERVER_TOOL_INPUT_DELTA: ("server_tool_input_delta", "Missing server-tool input delta"),
    ModelEventType.SERVER_TOOL_COMPLETED: ("server_tool_result", "Missing server-tool result"),
    ModelEventType.SERVER_TOOL_FAILED: ("server_tool_result", "Missing server-tool result"),
}


class ModelStepMixin(AgentContract):
    """Run one provider call inside a request and persist what it produced."""

    @mixin_api
    async def _run_model_step(
        self,
        context: AgentRunContext,
        reasoning_effort: ReasoningEffort,
        parallel_tool_call: bool,
        *,
        may_pause: bool = False,
    ) -> ModelResponse:
        """Run preprocessing and one model call, validate its stream, and persist output.

        ``may_pause`` says a paused step will be continued, so its pause is not recorded on the
        stored message: it would otherwise be reported again whenever the session is restored.
        """
        request = ModelRequest(
            messages=tuple(context.state.messages),
            tools=tuple(tool.definition for tool in context.tools.values()),
            server_tools=tuple(context.server_tools.values()),
            reasoning_effort=reasoning_effort,
            parallel_tool_call=parallel_tool_call,
            cache_key=context.config.cache_key or context.session_id,
            cache_mode=context.config.cache_mode,
        )
        await self._notify_before_model(context, request)
        # before_model hooks may add or replace instructions; publish those
        # changes before rebuilding the request sent to the provider.
        await context.flush_message_events()

        request = self._refresh_model_request(context, request)
        model_started = await self._start_model_generation(context)
        output_tracker = ModelOutputTracker()
        output_tracker.seed_server_tools(context._paused_server_tools)
        await context._emit_event(AgentEventType.MODEL_STARTED)
        response = None
        closed_server_calls: set[str] = set()
        model = context.model
        if model is None:
            raise AgentProtocolError("The request context has no model")
        async with aclosing_stream(self._stream_model_with_middleware(context, request, model)) as events:
            async for event in events:
                if response is not None:
                    raise AgentProtocolError("Model emitted events after its final response")
                if (
                    event.type is ModelEventType.SERVER_TOOL_STARTED
                    and event.server_tool_call is not None
                    and any(call.id == event.server_tool_call.id for call in context._paused_server_tools)
                ):
                    # The paused step already announced this call; do not show a second start card.
                    continue
                for boundary in await output_tracker.observe(context, event):
                    await context._emit_event(boundary)
                if event.type is ModelEventType.RESPONSE:
                    if event.response is None:
                        raise AgentProtocolError("Missing model response")
                    response = event.response
                else:
                    await self._relay_model_event(context, event)
                    if event.type in (ModelEventType.SERVER_TOOL_COMPLETED, ModelEventType.SERVER_TOOL_FAILED):
                        if event.server_tool_result is not None:
                            closed_server_calls.add(event.server_tool_result.call_id)
        if response is None:
            raise AgentProtocolError("Model stream ended without a response")

        response = self._settle_stop(response, may_pause=may_pause)
        response = await self._settle_paused_server_tools(
            context, response, output_tracker, may_pause, closed_server_calls
        )
        model_completed = await self._finish_model_generation(context)
        await context.append_message(
            response.message,
            output_tracker.message_timing(model_started, model_completed),
            response.usage,
        )
        await self._broadcast("after_model", context, response)
        await context._emit_event(AgentEventType.MODEL_COMPLETED, response=response)
        return response

    @staticmethod
    async def _relay_model_event(context: AgentRunContext, event: ModelEvent) -> None:
        """Show one streamed delta or hosted-tool event to the caller as the event of the same name."""
        relayed = _RELAYED.get(event.type)
        if relayed is None:
            return
        field_name, missing = relayed
        payload = getattr(event, field_name)
        if missing is not None and payload is None:
            raise AgentProtocolError(missing)
        await context._emit_event(AgentEventType(event.type.value), **{field_name: payload})

    async def _notify_before_model(self, context: AgentRunContext, request: ModelRequest) -> None:
        """Let every extension see the request, rebuilt from the context each time it is asked."""
        for extension in self.extensions:
            await extension.before_model(context, self._refresh_model_request(context, request))

    @staticmethod
    def _refresh_model_request(context: AgentRunContext, request: ModelRequest) -> ModelRequest:
        """Capture current context while retaining per-call model options.

        Preprocessing can replace history or tool registrations. Refresh before
        each observer and the provider so neither sees stale pre-compaction input.
        """
        return replace(
            request,
            messages=tuple(context.state.messages),
            tools=tuple(tool.definition for tool in context.tools.values()),
            server_tools=tuple(context.server_tools.values()),
        )

    @staticmethod
    def _settle_stop(response: ModelResponse, *, may_pause: bool = False) -> ModelResponse:
        """Put the stop on the message, and withhold tool calls that came with a bad stop.

        A refused or interrupted step cannot be trusted to have produced a whole tool call:
        a refusal can arrive after the call was streamed, and an interrupted stream may end
        in the middle of one. Running it would act on output the model or the provider
        disowned, so the calls are removed from the message that is stored and replayed, and
        their names are kept in ``stop.detail`` for the host to show.
        """
        stop = response.stop
        message = response.message
        if stop is None:  # pragma: no cover - ModelResponse always fills it in
            return response
        if stop.reason in (StopReason.REFUSAL, StopReason.INTERRUPTED) and message.tool_calls:
            withheld = [call.name for call in message.tool_calls]
            stop = replace(stop, detail={**stop.detail, "withheld_tool_calls": withheld})
            message = replace(message, tool_calls=())
        # Only an abnormal stop is written to the message. A message that ended normally is
        # left exactly as the model produced it, so stored history and every comparison
        # against it are unchanged; the stop of a normal step is still on the response.
        continued = continues_pause(stop, may_pause=may_pause, has_tool_calls=bool(message.tool_calls))
        if not stop.ok and not continued:
            message = replace(message, stop=stop)
        return replace(response, message=message, stop=stop)

    async def _settle_paused_server_tools(
        self,
        context: AgentRunContext,
        response: ModelResponse,
        tracker: ModelOutputTracker,
        may_pause: bool,
        closed: set[str],
    ) -> ModelResponse:
        """Carry hosted calls through a pause, or close the ones the pause chain abandoned.

        A paused step keeps its open calls so the next request can resume them. When the chain
        ends without a result, the blocks are removed from history and the host gets a terminal
        completion; otherwise the next turn would send a ``server_tool_use`` with no result.
        """
        opened = tracker.take_paused_server_tools()
        dropped = tracker.take_abandoned_server_tools()
        continued = continues_pause(
            response.stop, may_pause=may_pause, has_tool_calls=bool(response.message.tool_calls)
        )
        carried = tuple(context._paused_server_tools)
        if continued:
            merged = {call.id: call for call in carried}
            for call in opened:
                merged.setdefault(call.id, call)
            context._paused_server_tools[:] = list(merged.values())
            return response

        abandoned = {call.id: call for call in carried if call.id not in closed}
        for call in (*dropped, *opened):
            if call.id not in closed:
                abandoned.setdefault(call.id, call)
        context._paused_server_tools.clear()
        if not abandoned:
            return response

        ids = set(abandoned)
        response = replace(response, message=_without_server_tool_blocks(response.message, ids))
        _drop_server_tool_blocks(context, ids)
        for call in abandoned.values():
            await context._emit_event(
                AgentEventType.SERVER_TOOL_COMPLETED,
                server_tool_result=ServerToolResult(
                    call_id=call.id,
                    name=call.name,
                    output={"status": NOT_REPORTED_STATUS},
                ),
            )
        return response

    @mixin_api
    async def _announce_stop(self, context: AgentRunContext, response: ModelResponse) -> None:
        """Tell the host a step ended for a reason other than finishing or asking for tools."""
        stop = response.stop
        if stop is None or stop.ok:
            return
        await context._emit_event(
            AgentEventType.MODEL_STOPPED,
            response=response,
            stop=stop,
            message=response.message,
        )


def _replay_block_id(block: Mapping[str, Any]) -> str | None:
    """Return the provider item id a replay block carries, when it has one.

    Hosted items name it ``id`` in the protocols seen so far; ``call_id`` covers an
    adapter that uses that spelling instead.
    """
    value = block.get("id", block.get("call_id"))
    return value if isinstance(value, str) else None


def _without_server_tool_blocks(message: AssistantMessage, ids: set[str]) -> AssistantMessage:
    """Return ``message`` without the replay blocks of the given hosted calls."""
    blocks = tuple(block for block in message.replay_blocks if _replay_block_id(block) not in ids)
    return replace(message, replay_blocks=blocks) if len(blocks) != len(message.replay_blocks) else message


def _drop_server_tool_blocks(context: AgentRunContext, ids: set[str]) -> None:
    """Remove abandoned hosted calls from stored assistant history."""
    messages = list(context.state.messages)
    changed = False
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            updated = _without_server_tool_blocks(message, ids)
            if updated is not message:
                messages[index] = updated
                changed = True
    if changed:
        context.replace_messages(messages, emit_new=False)
