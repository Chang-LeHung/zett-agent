"""Blocking entry points exercise real Agent loops, tools, hooks, and storage."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar
from threading import Event, get_ident

import pytest

from zett_agent import (
    Agent,
    AgentEvent,
    AgentEventDispatcher,
    AgentEventType,
    AgentExtension,
    AgentMessage,
    AgentPhase,
    AgentProtocolError,
    AgentRunConfig,
    AgentRunContext,
    AskUserExtension,
    AssistantMessage,
    ExternalEvent,
    ModelEvent,
    ModelOutputTracker,
    ModelResponse,
    SQLiteSessionExtension,
    SyncAgent,
    SyncModelAdapter,
    SyncRuntime,
    ToolCall,
    UserMessage,
    create_agent_sync,
    tool,
)


class EchoModel:
    def __init__(self):
        self.requests = []
        self.loops = []

    async def stream(self, request):
        self.requests.append(request)
        self.loops.append(asyncio.get_running_loop())
        users = [m for m in request.messages if isinstance(m, UserMessage)]
        text = f"{len(users)}:{users[-1].text}"
        yield ModelEvent.text(text)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=text)))


def test_repeated_requests_share_loop_and_restore_history():
    model = EchoModel()
    with create_agent_sync(model) as agent:
        assert agent.run("one").content == "1:one"
        assert agent.run("two").content == "2:two"
        assert list(agent.stream("three"))[-1].type is AgentEventType.RUN_COMPLETED
    assert len(set(model.loops)) == 1
    assert model.loops[0].is_closed()


def test_runtime_calls_contextvars_and_original_exceptions():
    variable = ContextVar("sync_test", default="default")
    variable.set("caller")
    error = ValueError("original")

    async def fail():
        raise error

    with SyncRuntime() as runtime:
        assert runtime.call(variable.get) == "caller"
        with pytest.raises(ValueError) as caught:
            runtime.call(fail)
        assert caught.value is error
        assert runtime.call(lambda: 42) == 42
        with pytest.raises(RuntimeError, match="own event loop"):
            runtime.call(lambda: runtime.call(lambda: 1))
    with pytest.raises(RuntimeError, match="closed"):
        runtime.call(lambda: 1)
    runtime.close()


def test_stream_is_lazy_and_applies_backpressure_and_task_affinity():
    steps = []
    tasks = []

    async def source():
        owner = asyncio.current_task()
        try:
            for index in range(3):
                tasks.append(asyncio.current_task())
                steps.append(index)
                yield index
        finally:
            assert asyncio.current_task() is owner
            steps.append("closed")

    with SyncRuntime() as runtime:
        with runtime.stream(source) as stream:
            assert steps == []
            assert next(stream) == 0
            assert steps == [0]
            assert next(stream) == 1
        assert steps == [0, 1, "closed"]
    assert len(set(tasks)) == 1


def test_partial_stream_error_is_not_replayed():
    async def source():
        yield "visible"
        raise ValueError("partial failure")

    with SyncRuntime() as runtime, runtime.stream(source) as stream:
        assert next(stream) == "visible"
        with pytest.raises(ValueError, match="partial failure"):
            next(stream)
        assert list(stream) == []


def test_close_interrupts_blocked_next_and_finishes_cleanup():
    started = Event()
    cleaned = Event()

    async def source():
        try:
            started.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    with SyncRuntime() as runtime, ThreadPoolExecutor() as pool:
        stream = runtime.stream(source)
        consumer = pool.submit(next, stream, None)
        assert started.wait(3)
        with pytest.raises(RuntimeError, match="one consumer"):
            next(stream)
        stream.close()
        assert consumer.result(timeout=3) is None
        assert cleaned.is_set()


def test_runtime_close_cleans_started_and_unstarted_streams():
    runtime = SyncRuntime()
    steps = []

    async def source():
        try:
            yield 1
        finally:
            steps.append("closed")

    untouched = runtime.stream(source)
    active = runtime.stream(source)
    assert next(active) == 1
    runtime.close()
    assert steps == ["closed"]
    assert list(untouched) == []
    assert list(active) == []
    with pytest.raises(RuntimeError, match="closed"):
        runtime.stream(source)


def test_sync_context_keeps_task_identity_and_suppresses_errors():
    tasks = []

    @asynccontextmanager
    async def resource():
        tasks.append(asyncio.current_task())
        try:
            yield 42
        except ValueError:
            pass
        finally:
            tasks.append(asyncio.current_task())

    with SyncRuntime() as runtime:
        with runtime.context(resource()) as value:
            assert value == 42
            raise ValueError("suppressed")
        assert tasks[0] is tasks[1]


def test_close_early_cancels_agent_then_same_session_can_run():
    with SyncAgent(EchoModel(), config=AgentRunConfig("cancel")) as agent:
        with agent.stream("first") as events:
            assert next(events).type is AgentEventType.MODEL_STARTED
        assert agent.get_state("cancel").phase is AgentPhase.CANCELLED
        assert agent.run("second").content.endswith("second")


def test_sessions_are_isolated_across_calling_threads():
    with SyncAgent(EchoModel()) as agent, ThreadPoolExecutor() as pool:
        jobs = [pool.submit(agent.run, str(i), config=AgentRunConfig(f"session-{i}")) for i in range(4)]
        assert [job.result(timeout=5).content for job in jobs] == [f"1:{i}" for i in range(4)]


def test_sync_model_tool_and_sqlite_history(tmp_path):
    @tool
    def double(value: int) -> int:
        """Double a number.

        Args:
            value: Integer to multiply by two.
        Snippet:
            double(value=3)
        Guidelines:
            - Use for exact integer doubling.
        """
        return value * 2

    class ToolModel(EchoModel):
        async def stream(self, request):
            if not self.requests:
                self.requests.append(request)
                yield ModelEvent.completed(
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("one", "double", {"value": 3}),)))
                )
            else:
                async for event in super().stream(request):
                    yield event

    storage = SQLiteSessionExtension(tmp_path / "sessions.sqlite3")
    try:
        with SyncAgent(ToolModel(), tools=[double], extensions=[storage], config=AgentRunConfig("stored")) as agent:
            assert agent.run("calculate").content.endswith("calculate")
        records = storage.list_raw_messages("stored")
        assert [r.message.role for r in records] == ["user", "assistant", "tool", "assistant"]
        with storage.storage.sync() as db:
            view = db.load("stored")
            assert [m.role for m in view.messages] == ["user", "assistant", "tool", "assistant"]
        with double.sync() as invoke:
            assert invoke({"value": 7}) == 14
    finally:
        storage.close()


def test_sync_agent_uses_async_hooks_with_original_context():
    identities = {}

    class Hooks(AgentExtension):
        async def on_message(self, context):
            assert isinstance(context, AgentRunContext)
            assert not hasattr(context, "sync")
            assert get_ident() != caller_thread
            context.input_message = UserMessage(content="transformed")
            identities[context] = "retained"

        async def before_model_events(self, context, request):
            assert identities[context] == "retained"
            yield AgentEvent(
                type=AgentEventType.CUSTOM, session_id=context.config.session_id, name="sync", payload={"ok": True}
            )

        async def on_success(self, context, result):
            assert identities.pop(context) == "retained"

    caller_thread = get_ident()
    assert not hasattr(Hooks(), "sync")
    with SyncAgent(EchoModel(), extensions=[Hooks()]) as agent:
        events = list(agent.stream("raw"))
    assert any(e.name == "sync" for e in events)
    assert events[-1].message.content == "1:transformed"
    assert identities == {}


def test_internal_output_tracker_has_no_synchronous_facade():
    assert not hasattr(ModelOutputTracker(), "sync")


def test_sync_views_have_independent_identity_and_forward_attributes():
    class Service:
        __hash__ = None
        value = 1

    source = Service()
    with SyncRuntime() as runtime:
        first = runtime.wrap(source)
        second = runtime.wrap(source)
        assert first != second
        assert len({first, second}) == 2
        first.value = 2
        assert source.value == second.value == 2


def test_async_dispatcher_receives_events_from_sync_agent():
    seen = []

    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            seen.append((event.delta, get_ident()))

    caller = get_ident()
    with SyncAgent(EchoModel(), event_dispatcher=Handler()) as agent:
        assert agent.run("hello").content == "1:hello"
    assert seen[0][0] == "1:hello"
    assert seen[0][1] != caller


def test_ask_user_reply_from_sync_iterator():
    class AskModel(EchoModel):
        async def stream(self, request):
            if not self.requests:
                self.requests.append(request)
                yield ModelEvent.completed(
                    ModelResponse(
                        AssistantMessage(
                            tool_calls=(ToolCall("ask-1", "ask_user", {"question": "Which?", "options": ["A", "B"]}),)
                        )
                    )
                )
            else:
                async for event in super().stream(request):
                    yield event

    with SyncAgent(AskModel(), extensions=[AskUserExtension()]) as agent:
        with agent.stream("Choose") as events:
            for event in events:
                if event.name == "ask_user":
                    accepted = agent.emit_external_event(
                        ExternalEvent("ask_user_response", {"tool_call_id": "ask-1", "answer": "A"})
                    )
                    assert accepted == ["AskUserExtension"]
            assert event.type is AgentEventType.RUN_COMPLETED


async def test_sync_calls_work_inside_an_existing_event_loop():
    with SyncAgent(EchoModel()) as agent:
        assert agent.run("notebook").content == "1:notebook"


def test_existing_agent_can_be_initialized_and_run_through_sync_view():
    agent = Agent(EchoModel())
    with agent.sync() as blocking:
        with pytest.raises(AgentProtocolError, match="not initialized"):
            blocking.run("early")
        blocking.initialize(config=AgentRunConfig("existing"))
        assert blocking.run("now").content == "1:now"


def test_invalid_constructor_releases_owned_runtime():
    with pytest.raises(ValueError, match="positive"):
        SyncAgent(EchoModel(), max_iterations=0)


def test_sync_custom_model_and_callbacks_can_run_in_async_agent():
    class Model:
        def stream(self, request):
            yield ModelEvent.text("blocking model")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="blocking model")))

    with SyncAgent(SyncModelAdapter(Model())) as agent:
        assert agent.run("hi").content == "blocking model"


def test_close_propagates_cleanup_failures():
    async def source():
        try:
            yield 1
        finally:
            raise ValueError("cleanup failed")

    with SyncRuntime() as runtime:
        stream = runtime.stream(source)
        assert next(stream) == 1
        with pytest.raises(ValueError, match="cleanup failed"):
            stream.close()
        stream.close()


def test_mcp_scope_is_entered_and_exited_by_the_same_task():
    import anyio
    from mcp_types import ListToolsResult

    from zett_agent import McpExtension, McpHttpServer

    scopes = []

    class Client:
        async def list_tools(self, **kwargs):
            return ListToolsResult(tools=[])

    @asynccontextmanager
    async def factory(server):
        owner = asyncio.current_task()
        async with anyio.create_task_group():
            try:
                yield Client()
            finally:
                assert asyncio.current_task() is owner
                scopes.append(server.name)

    extension = McpExtension([McpHttpServer("test", "http://unused.invalid")], client_factory=factory)
    with SyncAgent(EchoModel(), extensions=[extension]) as agent:
        assert agent.run("first").content == "1:first"
        assert list(agent.stream("second"))[-1].type is AgentEventType.RUN_COMPLETED
        with agent.stream("cancel") as stream:
            next(stream)
    assert scopes == ["test", "test", "test"]


def test_sync_agent_async_hook_can_publish_to_other_extensions():
    from zett_agent import InternalMessageEvent

    class Publisher(AgentExtension):
        async def on_message(self, context):
            await context.publish(InternalMessageEvent(message=AgentMessage(content="Check")))

    with SyncAgent(EchoModel(), extensions=[Publisher()]) as agent:
        agent.run("hello")


def test_callback_failure_closes_the_request():
    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            raise ValueError("handler error")

    with SyncAgent(EchoModel(), config=AgentRunConfig("callback"), event_dispatcher=Handler()) as agent:
        with pytest.raises(ValueError, match="handler error"):
            list(agent.stream("hi"))
        assert agent.get_state("callback").phase is AgentPhase.CANCELLED


@pytest.mark.parametrize("failure", [False, True])
def test_async_contextmanager_exception_paths(failure):
    @asynccontextmanager
    async def resource():
        if failure:
            raise ValueError("enter failure")
        yield 1

    with SyncRuntime() as runtime:
        with pytest.raises(ValueError, match="enter failure" if failure else "body failure"):
            with runtime.context(resource()):
                raise ValueError("body failure")


def test_generic_proxy_handles_coroutine_returned_iterators_and_contexts():
    class Service:
        async def iterator(self):
            async def source():
                yield 4

            return source()

        @asynccontextmanager
        async def context(self):
            yield 5

    with SyncRuntime() as runtime:
        service = runtime.wrap(Service())
        assert list(service.iterator()) == [4]
        with service.context() as value:
            assert value == 5


def test_generic_invoke_preserves_positional_and_keyword_arguments():
    def combine(value: int, /, *, suffix: str) -> str:
        return f"{value}{suffix}"

    async def async_combine(value: int, /, *, suffix: str) -> str:
        return combine(value, suffix=suffix)

    with SyncRuntime() as runtime:
        proxy = runtime.wrap(object())
        assert proxy._invoke(combine, 3, suffix=" items") == "3 items"
        assert proxy._invoke(async_combine, 4, suffix=" items") == "4 items"
        with pytest.raises(TypeError):
            proxy._invoke(combine, 3)


def test_stream_can_be_closed_before_its_producer_starts():
    async def source():
        yield 1

    with SyncRuntime() as runtime:
        stream = runtime.stream(source)
        stream.close()
        assert list(stream) == []


def test_plain_async_iterator_without_aclose():
    class Iterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with SyncRuntime() as runtime:
        assert list(runtime.stream(Iterator)) == []


def test_shutdown_closes_all_streams_even_when_one_cleanup_fails():
    closed = []

    async def source(name):
        try:
            yield 1
        finally:
            closed.append(name)
            raise ValueError(name)

    runtime = SyncRuntime()
    one = runtime.stream(source, "one")
    two = runtime.stream(source, "two")
    assert next(one) == next(two) == 1
    with pytest.raises(ExceptionGroup) as caught:
        runtime.close()
    assert len(caught.value.exceptions) == 2
    assert sorted(closed) == ["one", "two"]
    assert not runtime._thread.is_alive()


def test_sync_agent_cancellation_closes_async_generator_hook():
    started = Event()
    finished = Event()

    class SlowHook(AgentExtension):
        async def before_model_events(self, context, request):
            try:
                started.set()
                await asyncio.Event().wait()
                yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name="slow")
            finally:
                finished.set()

    with SyncAgent(EchoModel(), extensions=[SlowHook()], config=AgentRunConfig("slow")) as agent:
        with ThreadPoolExecutor() as pool:
            stream = agent.stream("hello")
            reader = pool.submit(next, stream, None)
            assert started.wait(3)
            closer = pool.submit(stream.close)
            closer.result(timeout=5)
            reader.result(timeout=5)
        assert finished.is_set()


def test_shared_runtime_survives_agent_scope():
    with SyncRuntime() as runtime:
        with SyncAgent(EchoModel(), runtime=runtime) as agent:
            agent.run("shared")
        agent.close()
        assert runtime.call(lambda: 123) == 123
    with pytest.raises(RuntimeError, match="closed"):
        runtime.__enter__()


def test_abandoned_async_context_is_closed_during_runtime_shutdown():
    closed = []

    @asynccontextmanager
    async def resource():
        try:
            yield 1
        finally:
            closed.append(True)

    with SyncRuntime() as runtime:
        context = runtime.context(resource())
        assert context.__enter__() == 1
    assert closed == [True]


def test_callback_cannot_deadlock_by_closing_its_own_runtime():
    class Handler(AgentEventDispatcher):
        async def on_text_delta_event(self, event):
            runtime.close()

    with SyncRuntime() as runtime:
        with SyncAgent(EchoModel(), runtime=runtime, event_dispatcher=Handler()) as agent:
            with pytest.raises(RuntimeError, match="cannot block its own event loop"):
                agent.run("hello")
