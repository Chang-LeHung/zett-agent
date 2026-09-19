"""Shell approval event, allowlist, execution, and abort behavior."""

import asyncio

from zett_agent import (
    SHELL_APPROVAL_EVENT_NAME,
    SHELL_APPROVAL_RESPONSE_EVENT_NAME,
    Agent,
    AgentExtension,
    AgentRunConfig,
    AgentRunContext,
    AssistantMessage,
    ExternalEvent,
    ModelEvent,
    ModelResponse,
    ShellApprovalExtension,
    ShellApprovalMode,
    ToolCall,
    ToolMessage,
    tool,
)


class ShellModel:
    def __init__(self, command: str = "printf approved") -> None:
        self.command = command
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        message = (
            AssistantMessage(tool_calls=(ToolCall("shell-1", "run_shell", {"command": self.command}),))
            if len(self.requests) == 1
            else AssistantMessage(content="finished")
        )
        yield ModelEvent.completed(ModelResponse(message))


class FakeShellTool(AgentExtension):
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def on_tool(self, context: AgentRunContext) -> None:
        @tool(name="run_shell")
        async def run_shell(command: str, timeout_seconds: int = 30) -> str:
            """Run a test shell command.

            Guidelines:
                - Test only.
            """
            _ = timeout_seconds
            self.commands.append(command)
            return f"ran: {command}"

        context.register_tool(run_shell)


class ParallelShellModel:
    def __init__(self) -> None:
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        message = (
            AssistantMessage(
                tool_calls=(
                    ToolCall("shell-1", "run_shell", {"command": "first"}),
                    ToolCall("shell-2", "run_shell", {"command": "second"}),
                )
            )
            if len(self.requests) == 1
            else AssistantMessage(content="finished")
        )
        yield ModelEvent.completed(ModelResponse(message))


class SequentialShellModel:
    def __init__(self) -> None:
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            message = AssistantMessage(tool_calls=(ToolCall("shell-1", "run_shell", {"command": "first"}),))
        elif len(self.requests) == 2:
            message = AssistantMessage(tool_calls=(ToolCall("shell-2", "run_shell", {"command": "second"}),))
        else:
            message = AssistantMessage(content="finished")
        yield ModelEvent.completed(ModelResponse(message))


class MemoryShellApprovalStorage:
    def __init__(self, allowed: set[str] | None = None) -> None:
        self.allowed = set(allowed or ())
        self.modes: dict[str, ShellApprovalMode] = {}

    async def get_session_mode(self, session_id: str) -> ShellApprovalMode:
        return self.modes.get(session_id, ShellApprovalMode.REVIEW)

    async def set_session_mode(self, session_id: str, mode: ShellApprovalMode) -> None:
        self.modes[session_id] = mode

    async def clear_session_mode(self, session_id: str) -> None:
        self.modes.pop(session_id, None)

    async def is_allowed(self, session_id: str, command: str) -> bool:
        if self.modes.get(session_id) is ShellApprovalMode.ALLOW_ALL:
            return True
        return command in self.allowed

    async def allow_command(self, command: str) -> None:
        self.allowed.add(command)


async def test_shell_approval_storage_allowlist_bypasses_ui() -> None:
    storage = MemoryShellApprovalStorage({"printf approved"})
    shell = FakeShellTool()
    model = ShellModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("allowed"),
        extensions=[ShellApprovalExtension(storage), shell],
    )

    events = [event async for event in agent.stream("run")]

    assert shell.commands == ["printf approved"]
    assert all(event.name != SHELL_APPROVAL_EVENT_NAME for event in events)


async def test_disabled_shell_approval_allows_commands_without_events() -> None:
    storage = MemoryShellApprovalStorage()
    shell = FakeShellTool()
    model = ShellModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("allow-all"),
        extensions=[ShellApprovalExtension(storage, enabled=False), shell],
    )

    events = [event async for event in agent.stream("run")]

    assert shell.commands == ["printf approved"]
    assert all(event.name != SHELL_APPROVAL_EVENT_NAME for event in events)


async def test_session_allow_all_bypasses_shell_approval() -> None:
    storage = MemoryShellApprovalStorage()
    await storage.set_session_mode("allow-all-session", ShellApprovalMode.ALLOW_ALL)
    shell = FakeShellTool()
    model = ShellModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("allow-all-session"),
        extensions=[ShellApprovalExtension(storage), shell],
    )

    events = [event async for event in agent.stream("run")]

    assert shell.commands == ["printf approved"]
    assert all(event.name != SHELL_APPROVAL_EVENT_NAME for event in events)


async def test_shell_approval_execute_and_remember() -> None:
    storage = MemoryShellApprovalStorage()
    shell = FakeShellTool()
    model = ShellModel()
    config = AgentRunConfig("execute")
    agent = await Agent.create(
        model,
        config=config,
        extensions=[ShellApprovalExtension(storage), shell],
    )
    events = []
    approved = asyncio.Event()

    async def consume() -> None:
        async for event in agent.stream("run"):
            events.append(event)
            if event.name == SHELL_APPROVAL_EVENT_NAME:
                approved.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(approved.wait(), timeout=1)
    request = next(event for event in events if event.name == SHELL_APPROVAL_EVENT_NAME)
    assert request.payload["command"] == "printf approved"
    assert request.payload["remember_supported"] is True
    assert agent.emit_external_event(
        ExternalEvent(
            SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            {"tool_call_id": "shell-1", "decision": "execute", "remember": True},
        ),
        config=config,
    ) == ["ShellApprovalExtension"]
    await asyncio.wait_for(task, timeout=1)

    assert shell.commands == ["printf approved"]
    assert storage.allowed == {"printf approved"}
    result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert result.success is True


async def test_shell_approval_abort_becomes_failed_tool_result() -> None:
    storage = MemoryShellApprovalStorage()
    shell = FakeShellTool()
    model = ShellModel("rm -rf /tmp/example")
    config = AgentRunConfig("abort")
    agent = await Agent.create(
        model,
        config=config,
        extensions=[ShellApprovalExtension(storage), shell],
    )
    events = []
    ready = asyncio.Event()

    async def consume() -> None:
        async for event in agent.stream("run"):
            events.append(event)
            if event.name == SHELL_APPROVAL_EVENT_NAME:
                ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), timeout=1)
    agent.emit_external_event(
        ExternalEvent(
            SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            {"tool_call_id": "shell-1", "decision": "abort"},
        ),
        config=config,
    )
    await asyncio.wait_for(task, timeout=1)

    assert shell.commands == []
    result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert result.success is False
    assert "aborted by user" in str(result.content)


async def test_parallel_shell_calls_execute_as_each_approval_arrives() -> None:
    storage = MemoryShellApprovalStorage()
    shell = FakeShellTool()
    model = ParallelShellModel()
    config = AgentRunConfig("parallel-approvals")
    agent = await Agent.create(
        model,
        config=config,
        extensions=[ShellApprovalExtension(storage), shell],
    )
    events = []
    both_ready = asyncio.Event()

    async def consume() -> None:
        async for event in agent.stream("run"):
            events.append(event)
            approvals = [item for item in events if item.name == SHELL_APPROVAL_EVENT_NAME]
            if len(approvals) == 2:
                both_ready.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(both_ready.wait(), timeout=1)
    approvals = [item for item in events if item.name == SHELL_APPROVAL_EVENT_NAME]
    first = next(item for item in approvals if item.payload["command"] == "first")
    second = next(item for item in approvals if item.payload["command"] == "second")

    agent.emit_external_event(
        ExternalEvent(
            SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            {"tool_call_id": second.payload["tool_call_id"], "decision": "execute"},
        ),
        config=config,
    )
    await wait_until(lambda: shell.commands == ["second"])
    assert shell.commands == ["second"]

    agent.emit_external_event(
        ExternalEvent(
            SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            {"tool_call_id": first.payload["tool_call_id"], "decision": "execute"},
        ),
        config=config,
    )
    await asyncio.wait_for(task, timeout=1)

    assert shell.commands == ["second", "first"]


async def test_session_mode_change_applies_during_active_run() -> None:
    storage = MemoryShellApprovalStorage()
    shell = FakeShellTool()
    model = SequentialShellModel()
    config = AgentRunConfig("dynamic-mode")
    agent = await Agent.create(
        model,
        config=config,
        extensions=[ShellApprovalExtension(storage), shell],
    )
    events = []
    first_waiting = asyncio.Event()

    async def consume() -> None:
        async for event in agent.stream("run"):
            events.append(event)
            if event.name == SHELL_APPROVAL_EVENT_NAME:
                first_waiting.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_waiting.wait(), timeout=1)
    await storage.set_session_mode("dynamic-mode", ShellApprovalMode.ALLOW_ALL)
    agent.emit_external_event(
        ExternalEvent(
            SHELL_APPROVAL_RESPONSE_EVENT_NAME,
            {"tool_call_id": "shell-1", "decision": "execute"},
        ),
        config=config,
    )
    await asyncio.wait_for(task, timeout=1)

    assert shell.commands == ["first", "second"]
    assert [event.name for event in events].count(SHELL_APPROVAL_EVENT_NAME) == 1


async def wait_until(predicate, *, timeout: float = 1) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)
