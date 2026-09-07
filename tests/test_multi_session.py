"""One Agent concurrently serves isolated sessions on a shared event loop."""

import asyncio

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentMessage,
    AgentPhase,
    AgentProtocolError,
    AgentState,
    AskUserExtension,
    AssistantMessage,
    ExternalEvent,
    FileSystemExtension,
    InMemoryMessageAccumulator,
    ModelEvent,
    ModelResponse,
    PlanModeExtension,
    SQLiteSessionExtension,
    TodoWriteExtension,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class GatedModel:
    def __init__(self):
        self.entered = {name: asyncio.Event() for name in ("a", "b")}
        self.release = {name: asyncio.Event() for name in ("a", "b")}

    async def stream(self, request):
        label = next(m.content for m in reversed(request.messages) if isinstance(m, (UserMessage, AgentMessage)))
        self.entered[label[0]].set()
        await self.release[label[0]].wait()
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content=label)))


@pytest.mark.parametrize("persistent", [False, True])
async def test_concurrent_sessions_restore_only_their_own_history(tmp_path, persistent):
    extension = SQLiteSessionExtension(tmp_path / "sessions.db") if persistent else InMemoryMessageAccumulator()
    model = GatedModel()
    agent = await Agent.create(model, config=AgentConfig("a"), extensions=[extension], system_prompt="")
    tasks = [
        asyncio.create_task(agent.run(name, config=AgentConfig(name), metadata={"owner": name})) for name in ("a", "b")
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in model.entered.values())), 2)
        assert agent.get_state("a") is not agent.get_state("b")
        assert agent.get_state("missing") is None
        with pytest.raises(AgentProtocolError):
            await agent.run("a", config=AgentConfig("a"))
        with pytest.raises(AgentProtocolError, match="require config"):
            agent.emit_external_event(ExternalEvent("unknown", {}))
        model.release["a"].set()
        assert (await asyncio.wait_for(tasks[0], 2)).content == "a"
        assert agent.get_state("a").phase is AgentPhase.COMPLETED
        assert agent.get_state("b").phase is AgentPhase.GENERATING
        model.release["b"].set()
        await asyncio.wait_for(tasks[1], 2)
        for name in ("a", "b"):
            await agent.run(name, config=AgentConfig(name), metadata={"owner": name})
            assert [m.content for m in agent.get_state(name).messages] == [name] * 4
            if persistent:
                records = extension.list_raw_messages(name)
                assert [r.message.role for r in records] == ["user", "assistant"] * 2
                assert all(r.session_id == name and r.metadata == {"owner": name} for r in records)
                view = await extension.storage.load(name)
                assert [m.content for m in view.messages] == [name] * 4
            else:
                assert [m.content for m in extension.messages(name)] == [name] * 4
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if persistent:
            extension.close()


class QuestionModel:
    async def stream(self, request):
        if isinstance(request.messages[-1], ToolMessage):
            reply = AssistantMessage(content=request.messages[-1].content)
        else:
            reply = AssistantMessage(tool_calls=(ToolCall("same-call", "ask_user", {"question": "Choose"}),))
        yield ModelEvent.completed(ModelResponse(reply))


async def test_cancel_one_waiting_session_does_not_cancel_other(tmp_path):
    ask = AskUserExtension()
    storage = SQLiteSessionExtension(tmp_path / "cancel.db")
    agent = await Agent.create(QuestionModel(), config=AgentConfig("a"), extensions=[ask, storage])
    streams = {name: agent.stream(name, config=AgentConfig(name, request_id=name)) for name in ("a", "b")}
    try:
        for stream in streams.values():
            async for event in stream:
                if event.name == "ask_user":
                    break
        await streams["a"].aclose()
        assert agent.get_state("a").phase is AgentPhase.CANCELLED
        response = ExternalEvent("ask_user_response", {"tool_call_id": "same-call", "answer": "b-only"})
        assert agent.emit_external_event(response, config=AgentConfig("a")) == []
        assert agent.emit_external_event(response, config=AgentConfig("b", request_id="stale")) == []
        assert agent.emit_external_event(response, config=AgentConfig("b", request_id="b")) == ["AskUserExtension"]
        remaining = [event async for event in streams["b"]]
        assert remaining[-1].message.content.endswith('"b-only"}}')
        assert storage._requests == {}
        assert ask._pending == {}
        assert all("b-only" not in r.message.content for r in storage.list_raw_messages("a"))
    finally:
        for stream in streams.values():
            await stream.aclose()
        storage.close()


async def test_todo_cleanup_and_plan_baselines_are_session_local():
    todo, plan = TodoWriteExtension(), PlanModeExtension()
    contexts = [AgentContext(AgentConfig(name), AgentState(), {}) for name in ("a", "b")]
    for context in contexts:
        await todo.on_tool(context)
        await plan.on_tool(context)
        await plan.on_message(context)
        await context.tools["todo_write"]({"todos": [{"content": context.config.session_id, "status": "processing"}]})
    await todo.on_error(contexts[0], RuntimeError("a failed"))
    await plan.on_error(contexts[0], RuntimeError("a failed"))
    assert todo.todos("a") is None
    assert todo.todos("b").todos[0].content == "b"
    assert contexts[0] not in plan._requests
    assert contexts[1] in plan._requests
    await todo.after_run(contexts[1], AssistantMessage(content="done"))
    await plan.after_run(contexts[1], AssistantMessage(content="done"))
    assert todo.todos("b") is None
    assert plan._requests == {}


async def test_shared_filesystem_extension_copies_mutable_tool_definitions():
    extension = FileSystemExtension(read_only=True)
    a, b = [AgentContext(AgentConfig(name), AgentState(), {}) for name in ("a", "b")]
    await extension.on_tool(a)
    await extension.on_tool(b)
    a.tools["read_file"].description = "only session a"
    a.tools["read_file"].parameters["properties"].clear()
    assert b.tools["read_file"].description != "only session a"
    assert b.tools["read_file"].parameters["properties"]
    c = AgentContext(AgentConfig("c"), AgentState(), {})
    await extension.on_tool(c)
    assert c.tools["read_file"].parameters == b.tools["read_file"].parameters


async def test_internal_and_steering_queues_are_isolated_on_one_agent():
    model = GatedModel()
    agent = await Agent.create(model, config=AgentConfig("a"), system_prompt="")
    tasks = [asyncio.create_task(agent.run(name, config=AgentConfig(name))) for name in ("a", "b")]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in model.entered.values())), 2)
        assert agent.emit_external_event(
            ExternalEvent("internal_message", {"content": "a-internal"}), config=AgentConfig("a")
        ) == ["InternalMessageExtension"]
        assert agent.emit_external_event(
            ExternalEvent("steering_message", {"content": "b-steering"}), config=AgentConfig("b")
        ) == ["SteeringExtension"]
        for event in model.release.values():
            event.set()
        answers = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert [answer.content for answer in answers] == ["a-internal", "b-steering"]
        assert all(message.content.startswith("a") for message in agent.get_state("a").messages)
        assert all(message.content.startswith("b") for message in agent.get_state("b").messages)
        assert agent._internal_message_extension._inboxes == {}
        assert agent._steering_extension._inboxes == {}
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
