"""Run, stream, and persist a conversation without an asynchronous entry point."""

from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent import (
    AgentEventDispatcher,
    AgentEventType,
    AgentRunConfig,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    SQLiteSessionExtension,
    SyncModelAdapter,
    UserMessage,
    create_agent_sync,
)


class Echo:
    """A synchronous model used only to make this example runnable offline."""

    def stream(self, request):
        users = [message for message in request.messages if isinstance(message, UserMessage)]
        answer = f"Turn {len(users)}: {users[-1].text}"
        yield ModelEvent.text(answer)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=answer)))


class Console(AgentEventDispatcher):
    async def on_text_delta_event(self, event):
        print(event.delta)


def main() -> None:
    with TemporaryDirectory() as directory:
        history = SQLiteSessionExtension(Path(directory) / "sessions.sqlite3")
        try:
            with create_agent_sync(
                SyncModelAdapter(Echo()),
                config=AgentRunConfig("sync-example"),
                extensions=[history],
                event_dispatcher=Console(),
            ) as agent:
                assert agent.run("Hello").content == "Turn 1: Hello"
                with agent.stream("Remember this") as events:
                    for event in events:
                        if event.type is AgentEventType.RUN_COMPLETED:
                            assert event.message.content == "Turn 2: Remember this"
            records = history.list_raw_messages("sync-example")
            assert [record.message.role for record in records] == ["user", "assistant", "user", "assistant"]
            print("Synchronous history persisted")
        finally:
            history.close()


if __name__ == "__main__":
    main()
