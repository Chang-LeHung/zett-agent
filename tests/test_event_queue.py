import asyncio
import inspect

import pytest

from zett_agent import Agent, AgentEvent, AgentEventQueue, AgentEventType, AgentProtocolError


async def test_event_queue_preserves_fifo_order_without_stream_acknowledgements():
    queue = AgentEventQueue()
    first = AgentEvent(AgentEventType.MODEL_STARTED, "session")
    second = AgentEvent(AgentEventType.MODEL_COMPLETED, "session")

    await queue.put(first)
    await queue.put(second)

    assert await queue.get() is first
    assert await queue.get() is second


async def test_stream_acknowledgement_applies_backpressure_to_the_producer():
    queue = AgentEventQueue()
    queue.enable_acknowledgements()
    event = AgentEvent(AgentEventType.MODEL_STARTED, "session")
    produced = asyncio.create_task(queue.put(event))

    assert await queue.get() is event
    await asyncio.sleep(0)
    assert not produced.done()

    queue.acknowledge()
    await produced


async def test_event_queue_propagates_the_original_failure():
    queue = AgentEventQueue()
    error = RuntimeError("producer failed")

    await queue.fail(error)

    with pytest.raises(RuntimeError) as caught:
        await queue.get()
    assert caught.value is error


async def test_event_queue_rejects_output_after_run_completed():
    queue = AgentEventQueue()
    await queue.put(AgentEvent(AgentEventType.RUN_COMPLETED, "session"))

    with pytest.raises(AgentProtocolError, match="after RUN_COMPLETED"):
        await queue.put(AgentEvent(AgentEventType.CUSTOM, "session", name="late"))


async def test_closing_without_run_completed_is_a_protocol_error():
    queue = AgentEventQueue()
    await queue.close()

    with pytest.raises(AgentProtocolError, match="without RUN_COMPLETED"):
        await queue.get()


def test_only_the_outer_agent_stream_is_an_agent_event_async_generator():
    assert inspect.isasyncgenfunction(Agent.stream)
    for method_name in (
        "_run_request",
        "_run_loop",
        "_run_model_step",
        "_execute_tools",
        "_execute_parallel_tools",
        "_execute_serial_tools",
        "_finalize_tool",
        "_start_steering",
    ):
        method = getattr(Agent, method_name)
        assert inspect.iscoroutinefunction(method), method_name
        assert not inspect.isasyncgenfunction(method), method_name
