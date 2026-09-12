"""Parallel and serial local tool scheduling."""

import asyncio

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentEventType,
    AgentPhase,
    AssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolExecutionMode,
    ToolMessage,
    tool,
)


class ToolBatchModel:
    def __init__(self, calls: tuple[ToolCall, ...]) -> None:
        self.calls = calls
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        message = (
            AssistantMessage(tool_calls=self.calls) if len(self.requests) == 1 else AssistantMessage(content="done")
        )
        yield ModelEvent.completed(ModelResponse(message))


async def test_parallel_tools_run_first_together_then_serial_tools() -> None:
    running = 0
    peak_running = 0
    completed: list[str] = []
    both_started = asyncio.Event()

    @tool(execution_mode=ToolExecutionMode.PARALLEL)
    async def inspect(name: str) -> str:
        """Inspect one independent input.

        Args:
            name: Input name.

        Snippet:
            inspect(name="first")

        Guidelines:
            - Use for independent reads.
        """
        nonlocal running, peak_running
        running += 1
        peak_running = max(peak_running, running)
        if running == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        await asyncio.sleep(0)
        completed.append(name)
        running -= 1
        return name

    @tool(execution_mode=ToolExecutionMode.SERIAL)
    async def mutate(name: str) -> str:
        """Mutate one ordered input.

        Args:
            name: Input name.

        Snippet:
            mutate(name="last")

        Guidelines:
            - Run after independent reads.
        """
        assert running == 0
        assert set(completed) == {"first", "second"}
        completed.append(name)
        return name

    model = ToolBatchModel(
        (
            ToolCall("serial", "mutate", {"name": "last"}),
            ToolCall("parallel-1", "inspect", {"name": "first"}),
            ToolCall("parallel-2", "inspect", {"name": "second"}),
        )
    )
    agent = await Agent.create(
        model,
        config=AgentConfig("parallel"),
        tools=[inspect, mutate],
        extensions=[],
    )

    events = [event async for event in agent.stream("run")]

    assert peak_running == 2
    assert completed[-1] == "last"
    started = [event for event in events if event.type is AgentEventType.TOOL_STARTED]
    assert [[call.id for call in event.tool_calls] for event in started] == [
        ["parallel-1", "parallel-2"],
        ["serial"],
    ]
    results = [message for message in model.requests[1].messages if isinstance(message, ToolMessage)]
    assert {message.tool_call_id for message in results[:2]} == {"parallel-1", "parallel-2"}
    assert results[-1].tool_call_id == "serial"
    assert all(request.parallel_tool_call for request in model.requests)


async def test_agent_parallel_switch_forces_parallel_marked_tools_to_run_serially() -> None:
    running = 0
    peak_running = 0

    @tool(execution_mode=ToolExecutionMode.PARALLEL)
    async def inspect(value: int) -> int:
        """Inspect one value.

        Args:
            value: Value to return.

        Snippet:
            inspect(value=1)

        Guidelines:
            - Use for independent reads.
        """
        nonlocal running, peak_running
        running += 1
        peak_running = max(peak_running, running)
        await asyncio.sleep(0.01)
        running -= 1
        return value

    model = ToolBatchModel((ToolCall("one", "inspect", {"value": 1}), ToolCall("two", "inspect", {"value": 2})))
    agent = await Agent.create(
        model,
        config=AgentConfig("serial-override"),
        tools=[inspect],
        extensions=[],
        parallel_tool_call=False,
    )

    events = [event async for event in agent.stream("run")]

    assert peak_running == 1
    assert not any(request.parallel_tool_call for request in model.requests)
    started = [event for event in events if event.type is AgentEventType.TOOL_STARTED]
    assert [[call.id for call in event.tool_calls] for event in started] == [["one"], ["two"]]


async def test_parallel_failure_is_returned_without_cancelling_sibling_or_serial_work() -> None:
    completed: list[str] = []

    @tool(execution_mode=ToolExecutionMode.PARALLEL)
    async def inspect(name: str, fail: bool = False) -> str:
        """Inspect one input and optionally fail.

        Args:
            name: Input name.
            fail: Whether to raise an expected tool error.

        Snippet:
            inspect(name="first")

        Guidelines:
            - Use for independent reads.
        """
        await asyncio.sleep(0)
        completed.append(name)
        if fail:
            raise RuntimeError(name)
        return name

    @tool(execution_mode=ToolExecutionMode.SERIAL)
    def finish() -> str:
        """Finish serial work.

        Snippet:
            finish()

        Guidelines:
            - Run after reads.
        """
        completed.append("finish")
        return "finish"

    model = ToolBatchModel(
        (
            ToolCall("bad", "inspect", {"name": "bad", "fail": True}),
            ToolCall("good", "inspect", {"name": "good"}),
            ToolCall("finish", "finish"),
        )
    )
    agent = await Agent.create(model, config=AgentConfig("failure"), tools=[inspect, finish], extensions=[])

    events = [event async for event in agent.stream("run")]

    assert set(completed[:2]) == {"bad", "good"}
    assert completed[-1] == "finish"
    assert [
        event.type for event in events if event.type in (AgentEventType.TOOL_FAILED, AgentEventType.TOOL_COMPLETED)
    ] == [
        AgentEventType.TOOL_FAILED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.TOOL_COMPLETED,
    ]


async def test_parallel_results_stream_in_completion_order_with_same_named_calls_and_failure() -> None:
    releases = {name: asyncio.Event() for name in ("slow", "fast", "failed")}
    all_started = asyncio.Event()
    started = 0

    @tool
    async def inspect(name: str, fail: bool = False) -> str:
        """Inspect one independently released input.

        Args:
            name: Release gate name.
            fail: Whether this invocation should fail.

        Snippet:
            inspect(name="fast")

        Guidelines:
            - Use for independent completion-order tests.
        """
        nonlocal started
        started += 1
        if started == 3:
            all_started.set()
        await releases[name].wait()
        if fail:
            raise RuntimeError("expected failure")
        return name

    model = ToolBatchModel(
        (
            ToolCall("slow-id", "inspect", {"name": "slow"}),
            ToolCall("fast-id", "inspect", {"name": "fast"}),
            ToolCall("failed-id", "inspect", {"name": "failed", "fail": True}),
        )
    )
    agent = await Agent.create(model, config=AgentConfig("completion-order"), tools=[inspect], extensions=[])
    stream = agent.stream("run")

    async def release_in_order() -> None:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        for name in ("fast", "failed", "slow"):
            releases[name].set()
            await asyncio.sleep(0.01)

    release_task = asyncio.create_task(release_in_order())
    events = [event async for event in stream]
    await release_task

    terminal = [event for event in events if event.type in (AgentEventType.TOOL_COMPLETED, AgentEventType.TOOL_FAILED)]
    assert [event.tool_calls[0].id for event in terminal] == ["fast-id", "failed-id", "slow-id"]
    assert [event.type for event in terminal] == [
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.TOOL_FAILED,
        AgentEventType.TOOL_COMPLETED,
    ]
    assert [event.phase for event in terminal] == [
        AgentPhase.RUNNING_TOOL,
        AgentPhase.RUNNING_TOOL,
        AgentPhase.READY,
    ]
    results = [message for message in model.requests[1].messages if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in results] == ["fast-id", "failed-id", "slow-id"]


def test_tools_default_to_parallel_and_reject_invalid_execution_mode() -> None:
    @tool
    def default_tool() -> str:
        """Return a value.

        Snippet:
            default_tool()

        Guidelines:
            - Use in tests.
        """
        return "ok"

    assert default_tool.execution_mode is ToolExecutionMode.PARALLEL
    with pytest.raises(ValueError, match="ToolExecutionMode"):
        default_tool.execution_mode = "parallel"  # type: ignore[assignment]
        default_tool.__post_init__()

    with pytest.raises(ValueError, match="parallel_tool_call"):
        ModelRequest(messages=(), parallel_tool_call=1)  # type: ignore[arg-type]


async def test_run_override_does_not_change_agent_parallel_default() -> None:
    model = ToolBatchModel(())
    agent = await Agent.create(
        model,
        config=AgentConfig("override"),
        extensions=[],
        parallel_tool_call=False,
    )

    await agent.run("override", parallel_tool_call=True)
    await agent.run("default")

    assert [request.parallel_tool_call for request in model.requests] == [True, False]
    assert agent.parallel_tool_call is False


async def test_cancelling_parallel_batch_cancels_every_running_handler() -> None:
    started = 0
    all_started = asyncio.Event()
    cancelled: list[int] = []

    @tool(execution_mode=ToolExecutionMode.PARALLEL)
    async def wait(index: int) -> None:
        """Wait until the request is cancelled.

        Args:
            index: Invocation identifier.

        Snippet:
            wait(index=1)

        Guidelines:
            - Use to verify cancellation.
        """
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(index)

    model = ToolBatchModel((ToolCall("one", "wait", {"index": 1}), ToolCall("two", "wait", {"index": 2})))
    agent = await Agent.create(model, config=AgentConfig("cancel"), tools=[wait], extensions=[])
    task = asyncio.create_task(agent.run("run"))
    await asyncio.wait_for(all_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert set(cancelled) == {1, 2}
    assert agent.state.phase.value == "cancelled"
