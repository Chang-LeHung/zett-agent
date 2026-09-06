"""Built-in extensions for history, tool guidance, and filesystem access."""

from .agent import AgentContext
from .extension_events import CompactionEvent, ExtensionEvent, MessageAppendedEvent
from .extension_hooks import AgentExtension
from .messages import AnyMessage, SystemMessage
from .tools import AgentTool, glob, grep, read_file, render_tool_guidance, replace_in_file, run_shell, write_file


class FileSystemExtension(AgentExtension):
    """Register working-directory file tools with an optional read-only boundary.

    Read-only mode exposes read_file, glob, and grep. Writable mode additionally
    exposes write_file and replace_in_file. Shell execution is deliberately not
    a filesystem capability because arbitrary commands cannot guarantee that
    they will leave the workspace unchanged.

    Example:
        extension = FileSystemExtension(read_only=True)
        agent = await Agent.create(model, config=config, extensions=[extension])
    """

    def __init__(self, *, read_only: bool = False) -> None:
        self.read_only = read_only

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """Return the exact immutable registration set for the configured mode."""
        read_tools = (read_file, glob, grep)
        return read_tools if self.read_only else (*read_tools, write_file, replace_in_file)

    async def on_tool(self, context: AgentContext) -> None:
        """Register the selected filesystem tools for this request."""
        for registered in self.tools:
            context.register_tool(registered)


class CodingExtension(FileSystemExtension):
    """Register local coding tools using the process working directory.

    Provides read_file, write_file, replace_in_file, glob, grep, and run_shell.
    File tools retain their existing schemas and path validation. Shell commands
    execute with the host process permissions; the working directory is not a
    sandbox. Add ToolGuidelinesExtension to include
    their snippets and guidelines in the model's system instructions.

    Example:
        agent = await Agent.create(
            model,
            config=AgentConfig(session_id="coding"),
            extensions=[CodingExtension(), ToolGuidelinesExtension()],
        )
        await agent.run("Read README.md and find Python files.")
    """

    def __init__(self) -> None:
        super().__init__(read_only=False)

    async def on_tool(self, context: AgentContext) -> None:
        """Register writable filesystem tools followed by shell execution."""
        await super().on_tool(context)
        context.register_tool(run_shell)


class ToolGuidelinesExtension(AgentExtension):
    """Inject persistent tool instructions once when an Agent starts."""

    async def on_message(self, context: AgentContext) -> None:
        """Place guidance after existing system instructions and before dialogue."""
        state = context.state
        message = SystemMessage(content=render_tool_guidance(tuple(context.tools.values())))
        if message.content and message not in state.messages:
            instructions = [message for message in state.messages if isinstance(message, SystemMessage)]
            dialogue = [message for message in state.messages if not isinstance(message, SystemMessage)]
            state.messages[:] = [*instructions, message, *dialogue]


class InMemoryMessageAccumulator(AgentExtension):
    """Retain non-system conversation messages for reuse by later requests.

    System instructions belong to the current Agent configuration and are rebuilt
    for every request. Keeping them here would accumulate stale prompts whenever
    an Agent or its tool guidance changes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[AnyMessage]] = {}

    async def on_message(self, context: AgentContext) -> None:
        """Combine current instructions with an independent copy of remembered dialogue."""
        state = context.state
        instructions = [message for message in state.messages if isinstance(message, SystemMessage)]
        current_dialogue = [message for message in state.messages if not isinstance(message, SystemMessage)]
        remembered = self._sessions.get(context.config.session_id)
        if remembered is None:
            remembered = list(current_dialogue)
            self._sessions[context.config.session_id] = remembered
        state.messages[:] = [*instructions, *remembered]

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Accumulate raw messages and replace dialogue after successful compaction."""
        session_id = context.config.session_id
        match event:
            case MessageAppendedEvent(message=SystemMessage()):
                return
            case MessageAppendedEvent(message=message):
                self._sessions.setdefault(session_id, []).append(message)
            case CompactionEvent():
                self._sessions[session_id] = [
                    message for message in context.state.messages if not isinstance(message, SystemMessage)
                ]

    def messages(self, session_id: str) -> tuple[AnyMessage, ...]:
        """Return remembered dialogue without request-specific system instructions."""
        return tuple(self._sessions.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        """Forget one session without changing other accumulated sessions."""
        self._sessions.pop(session_id, None)
