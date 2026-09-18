"""An extension with tools, input rewriting, UI events, and request cleanup."""

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy

from zett_agent import (
    AgentEvent,
    AgentEventDispatcher,
    AgentEventType,
    AgentExtension,
    AgentRunContext,
    AssistantMessage,
    ExtensionEvent,
    InMemoryMessageAccumulator,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RetryOptions,
    RunCancelledEvent,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
    UserMessage,
    create_agent,
    tool,
)


class NoteExtension(AgentExtension):
    """Expose a request-bound note tool and translate an explicit /note command."""

    priority = 50

    def __init__(self) -> None:
        self.active: dict[AgentRunContext, int] = {}

    async def on_tool(self, context: AgentRunContext) -> None:
        @tool
        def lookup_note(topic: str) -> str:
            """Look up a small demonstration note.

            Args:
                topic: Exact note topic to look up.

            Snippet:
                lookup_note(topic="extensions")

            Guidelines:
                - Use when a user requests a project note.
            """
            return {"extensions": "Extensions customize lifecycle hooks."}.get(topic, "No note found.")

        context.register_tool(lookup_note)

    async def on_state(self, context: AgentRunContext) -> None:
        context.add_message(SystemMessage(content="Use the note tool for project facts."), index=0)

    async def on_message(self, context: AgentRunContext) -> None:
        message = context.input_message
        if message is not None and message.text.startswith("/note "):
            attributes = deepcopy(message.attributes)
            attributes["original_content"] = deepcopy(message.content)
            context.input_message = UserMessage(
                content=f"Look up this note: {message.text.removeprefix('/note ')}",
                attributes=attributes,
            )
        # The Agent appends input after all on_message hooks. Do not append here.
        self.active[context] = 0

    async def before_model(self, context: AgentRunContext, request: ModelRequest) -> None:
        self.active[context] += 1
        await context.emit(
            AgentEvent(
                AgentEventType.CUSTOM,
                session_id=context.config.session_id,
                name="note.model_step",
                payload={"step": self.active[context]},
            )
        )

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        self.active.pop(context, None)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        self.active.pop(context, None)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        if isinstance(event, RunCancelledEvent):
            self.active.pop(context, None)


class NoteModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(request.messages[-1], ToolMessage):
            message = AssistantMessage(content="Extensions customize lifecycle hooks.")
        else:
            assert request.messages[-1].text == "Look up this note: extensions"
            assert request.messages[-1].attributes["original_content"] == "/note extensions"
            assert any(isinstance(item, SystemMessage) and "note tool" in item.content for item in request.messages)
            message = AssistantMessage(tool_calls=(ToolCall("note-1", "lookup_note", {"topic": "extensions"}),))
        yield ModelEvent.completed(ModelResponse(message))


class NoteEvents(AgentEventDispatcher):
    async def on_custom_event(self, event: AgentEvent) -> None:
        if event.name == "note.model_step":
            print(f"Model step: {event.payload['step']}")


async def main() -> None:
    extension = NoteExtension()
    client = await create_agent(
        NoteModel(),
        extensions=[InMemoryMessageAccumulator(), extension, ToolGuidelinesExtension()],
        event_dispatcher=NoteEvents(),
    )
    reply = await client.run("/note extensions")
    assert not extension.active
    print(reply.content)
    print("Request state cleared")


if __name__ == "__main__":
    asyncio.run(main())
