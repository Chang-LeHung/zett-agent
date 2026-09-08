"""Application-side callbacks for events emitted by Agent.stream()."""

from .events import AgentEvent, AgentEventType


class AgentEventDispatcher:
    """Dispatch streamed events to optional asynchronous callbacks.

    Override only the callbacks your UI or application needs. Each callback
    receives the original event, including its relevant message, tool calls,
    or custom payload. Awaiting dispatch preserves stream order and applies
    backpressure; callback failures and cancellation propagate to the caller.
    This class does not change the Agent lifecycle or consume its stream.

    Examples:
        Usage::

            from contextlib import aclosing

            class ConsoleEvents(AgentEventDispatcher):
                async def on_text_delta_event(self, event: AgentEvent) -> None:
                    print(event.delta, end="", flush=True)

                async def on_custom_event(self, event: AgentEvent) -> None:
                    print(event.name, event.payload)

            handler = ConsoleEvents()
            async with aclosing(agent.stream("Inspect this project")) as events:
                async for event in events:
                    await handler.dispatch(event)

    Stream errors and cancellation are raised by the stream itself, not
    synthesized as callbacks. Internal extension events are a separate API.
    """

    async def dispatch(self, event: AgentEvent) -> None:
        """Await the callback named on_<event type>_event exactly once.

        Routing uses the finite AgentEventType enum, never a custom event name.
        All enum values have explicit methods for completion and discovery in
        editors. Custom names remain data handled by on_custom_event().
        """
        event_type = AgentEventType(event.type)
        callback = getattr(self, f"on_{event_type.value}_event")
        await callback(event)

    async def on_compaction_started_event(self, event: AgentEvent) -> None:
        """Handle the start of context compaction."""

    async def on_compaction_text_delta_event(self, event: AgentEvent) -> None:
        """Handle a streamed checkpoint text fragment."""

    async def on_compaction_reasoning_delta_event(self, event: AgentEvent) -> None:
        """Handle a streamed compaction reasoning fragment."""

    async def on_compaction_completed_event(self, event: AgentEvent) -> None:
        """Handle completion of context compaction."""

    async def on_model_started_event(self, event: AgentEvent) -> None:
        """Handle the start of a primary model call."""

    async def on_text_delta_event(self, event: AgentEvent) -> None:
        """Handle an answer-content fragment in event.delta."""

    async def on_reasoning_delta_event(self, event: AgentEvent) -> None:
        """Handle a reasoning fragment in event.delta."""

    async def on_tool_call_delta_event(self, event: AgentEvent) -> None:
        """Handle a streamed tool-call fragment."""

    async def on_model_completed_event(self, event: AgentEvent) -> None:
        """Handle the complete model response."""

    async def on_tool_started_event(self, event: AgentEvent) -> None:
        """Handle the start of tool execution."""

    async def on_tool_completed_event(self, event: AgentEvent) -> None:
        """Handle a successful tool result."""

    async def on_tool_failed_event(self, event: AgentEvent) -> None:
        """Handle a failed tool result."""

    async def on_tool_skipped_event(self, event: AgentEvent) -> None:
        """Handle a tool call skipped because steering took over."""

    async def on_steering_started_event(self, event: AgentEvent) -> None:
        """Handle the start of steering-message processing."""

    async def on_steering_completed_event(self, event: AgentEvent) -> None:
        """Handle completion of steering-message processing."""

    async def on_steering_interrupted_event(self, event: AgentEvent) -> None:
        """Handle interrupted steering-message processing."""

    async def on_internal_message_started_event(self, event: AgentEvent) -> None:
        """Handle the start of internal-message processing."""

    async def on_internal_message_completed_event(self, event: AgentEvent) -> None:
        """Handle completion of internal-message processing."""

    async def on_internal_message_interrupted_event(self, event: AgentEvent) -> None:
        """Handle interrupted internal-message processing."""

    async def on_run_completed_event(self, event: AgentEvent) -> None:
        """Handle the final successful answer in event.message."""

    async def on_custom_event(self, event: AgentEvent) -> None:
        """Handle extension-defined events using event.name and event.payload."""
