"""Round-trip an external approval through the framework's pending-event base."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from zett_agent import (
    AgentEvent,
    AgentEventType,
    AgentRunConfig,
    AgentRunContext,
    AssistantMessage,
    ExternalEvent,
    ExternalEventExtension,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
    create_agent,
    tool,
)


class ApprovalExtension(ExternalEventExtension):
    def __init__(self) -> None:
        super().__init__(response_event_name="approval.response", correlation_field="tool_call_id")

    async def on_tool(self, context: AgentRunContext) -> None:
        @tool
        def request_approval(action: str) -> str:
            """Ask the user whether a proposed action should proceed.

            Args:
                action: Human-readable proposed action; this tool does not execute it.

            Snippet:
                request_approval(action="Export notes")

            Guidelines:
                - Ask before performing an action that requires user approval.
            """
            response = self._take_external_event(context)
            approved = response.payload.get("approved")
            if not isinstance(approved, bool):
                raise ValueError("approved must be a boolean")
            return "approved" if approved else "rejected"

        context.register_tool(request_approval)

    async def before_tool_events(self, context: AgentRunContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        if call.name != "request_approval":
            return
        async with self._wait_for_external_event(context, call.id):
            yield AgentEvent(
                AgentEventType.CUSTOM,
                session_id=context.config.session_id,
                name="approval.requested",
                payload={"tool_call_id": call.id, "action": call.arguments["action"]},
            )
        # Leaving the async-with body waits for the routed external response.


class ApprovalModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            assert request.messages[-1].success
            message = AssistantMessage(content="Approval received; no export was executed in this demo.")
        else:
            message = AssistantMessage(
                tool_calls=(ToolCall("approval-1", "request_approval", {"action": "Export notes"}),)
            )
        yield ModelEvent.completed(ModelResponse(message))


async def main() -> None:
    config = AgentRunConfig(session_id="approval-demo", request_id="request-1")
    client = await create_agent(
        ApprovalModel(), config=config, extensions=[ApprovalExtension(), ToolGuidelinesExtension()]
    )
    async with aclosing(client.stream("Export my notes")) as events:
        async for event in events:
            if event.type == AgentEventType.CUSTOM and event.name == "approval.requested":
                print(f"Question: {event.payload['action']}?")
                # A real application sends this from a UI/API callback, not automatically.
                response = ExternalEvent(
                    name="approval.response",
                    payload={"tool_call_id": event.payload["tool_call_id"], "approved": True},
                )
                accepted = client.agent.emit_external_event(response, config=config)
                assert accepted == ["ApprovalExtension"]
                assert client.agent.emit_external_event(response, config=config) == []
                print("Response accepted; duplicate rejected")
            elif event.type == AgentEventType.RUN_COMPLETED:
                print(event.message.content)


if __name__ == "__main__":
    asyncio.run(main())
