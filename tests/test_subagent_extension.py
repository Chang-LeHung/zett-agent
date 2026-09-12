"""Subagent delegation, isolation, guidance, and parent-session persistence."""

import asyncio
import json
from pathlib import Path
from uuid import UUID

import pytest

from zett_agent import (
    Agent,
    AgentExtension,
    AgentPhase,
    AgentRunConfig,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    ReasoningEffort,
    SessionPersistenceExtension,
    SQLiteSessionExtension,
    SQLiteSessionStorage,
    SubAgentDefinition,
    SubAgentExtension,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
    default_subagents,
)


@pytest.fixture
def storage(tmp_path):
    storage = SQLiteSessionStorage(tmp_path / "subagents.sqlite3")
    yield storage
    storage.close()


@pytest.fixture
def builtins(tmp_path, monkeypatch):
    """Create built-in definitions without touching the user's data directory."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    created: list[tuple[SubAgentDefinition, ...]] = []

    def create(model):
        definitions = default_subagents(model)
        created.append(definitions)
        return definitions

    yield create
    for definitions in created:
        for definition in definitions:
            for extension in definition.extensions:
                if isinstance(extension, SQLiteSessionExtension):
                    extension.close()


def child_storage(definitions: tuple[SubAgentDefinition, ...]) -> SQLiteSessionStorage:
    """Return the SQLite storage explicitly owned by a built-in definition."""
    for extension in definitions[0].extensions:
        if isinstance(extension, SQLiteSessionExtension):
            return extension.storage
    raise AssertionError("Built-in subagent does not define SQLite persistence")


class DelegatingModel:
    def __init__(self, subagent_type: str = "explore") -> None:
        self.subagent_type = subagent_type
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        system = next(
            (message.content for message in request.messages if isinstance(message, SystemMessage)),
            "",
        )
        if system.startswith("You are a read-only code exploration subagent"):
            message = AssistantMessage(content="Found the storage boundary in src/zett_agent/storage.py")
        elif system.startswith("You are a reasoning subagent"):
            message = AssistantMessage(content="Prefer the smaller design because it has fewer states")
        elif system.startswith("You are a custom subagent"):
            message = AssistantMessage(content="Custom extension completed")
        elif any(isinstance(message, ToolMessage) for message in request.messages):
            message = AssistantMessage(content="I used the delegated report")
        else:
            message = AssistantMessage(
                tool_calls=(
                    ToolCall(
                        "task-1",
                        "task",
                        {
                            "description": "Inspect session storage",
                            "prompt": "Find the session storage boundary and return one relevant path.",
                            "subagent_type": self.subagent_type,
                        },
                    ),
                )
            )
        yield ModelEvent.completed(ModelResponse(message))


async def test_explore_subagent_runs_end_to_end_in_a_persisted_child_session(storage, builtins):
    model = DelegatingModel()
    definitions = builtins(model)
    extension = SubAgentExtension(definitions)
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent-session"),
        extensions=[
            SessionPersistenceExtension(storage),
            extension,
            ToolGuidelinesExtension(),
        ],
    )

    result = await agent.run("Inspect the storage implementation")

    assert result.content == "I used the delegated report"
    assert agent.state.phase == AgentPhase.COMPLETED
    child_request = next(
        request
        for request in model.requests
        if any(
            isinstance(message, SystemMessage)
            and message.content.startswith("You are a read-only code exploration subagent")
            for message in request.messages
        )
    )
    assert {definition.name for definition in child_request.tools} == {"read_file", "glob", "grep"}
    assert child_request.reasoning_effort == ReasoningEffort.LOW

    parent_follow_up = model.requests[-1]
    tool_message = next(message for message in parent_follow_up.messages if isinstance(message, ToolMessage))
    payload = json.loads(tool_message.content)
    child_session_id = payload["session_id"]
    assert UUID(child_session_id).version == 7
    assert payload == {
        "session_id": child_session_id,
        "parent_session_id": "parent-session",
        "subagent_type": "explore",
        "content": "Found the storage boundary in src/zett_agent/storage.py",
    }

    subagent_storage = child_storage(definitions)
    child_view = await subagent_storage.load(child_session_id)
    assert child_view.parent_session_id == "parent-session"
    assert child_view.title is None
    assert child_view.agent_name is None
    assert [record.message.content for record in child_view.raw_tail] == [
        "Find the session storage boundary and return one relevant path.",
        "Found the storage boundary in src/zett_agent/storage.py",
    ]
    parent_summary = storage.list_sessions()[0]
    assert parent_summary.session_id == "parent-session"
    assert parent_summary.parent_session_id is None
    assert parent_summary.message_count == 4
    child_summary = subagent_storage.list_sessions()[0]
    assert child_summary.session_id == child_session_id
    assert child_summary.parent_session_id == "parent-session"
    assert child_summary.title is None
    assert child_summary.agent_name is None
    assert child_summary.message_count == 2


async def test_task_schema_and_guidance_describe_available_subagents(storage, builtins):
    model = DelegatingModel(subagent_type="reasoning")
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent"),
        extensions=[
            SessionPersistenceExtension(storage),
            SubAgentExtension(builtins(model)),
            ToolGuidelinesExtension(),
        ],
    )

    await agent.run("Compare the designs")

    parent_request = model.requests[0]
    task = next(definition for definition in parent_request.tools if definition.name == "task")
    assert task.parameters["properties"]["subagent_type"]["enum"] == ["reasoning", "explore"]
    assert task.description == "Delegate a complex task to an isolated specialized subagent."
    guidance = "\n".join(message.content for message in parent_request.messages if isinstance(message, SystemMessage))
    assert 'task(description="Trace auth flow"' in guidance
    assert "Use 'reasoning'" in guidance
    assert "Use 'explore'" in guidance
    child_request = next(
        request
        for request in model.requests
        if any(
            isinstance(message, SystemMessage) and message.content.startswith("You are a reasoning subagent")
            for message in request.messages
        )
    )
    assert child_request.tools == ()
    assert child_request.reasoning_effort == ReasoningEffort.HIGH


async def test_unknown_subagent_becomes_a_failed_tool_result_without_a_child_session(storage, builtins):
    model = DelegatingModel(subagent_type="missing")
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent"),
        extensions=[SessionPersistenceExtension(storage), SubAgentExtension(builtins(model))],
    )

    await agent.run("Delegate this")

    failed = next(message for message in model.requests[-1].messages if isinstance(message, ToolMessage))
    assert failed.success is False
    assert "Unknown subagent type" in failed.content
    assert all(summary.parent_session_id is None for summary in storage.list_sessions())


async def test_cancelling_parent_propagates_into_a_running_subagent(storage, builtins):
    class BlockingChildModel:
        def __init__(self) -> None:
            self.child_started = asyncio.Event()
            self.never = asyncio.Event()

        async def stream(self, request):
            system = next(
                (message.content for message in request.messages if isinstance(message, SystemMessage)),
                "",
            )
            if system.startswith("You are a reasoning subagent"):
                self.child_started.set()
                await self.never.wait()
                return
            yield ModelEvent.completed(
                ModelResponse(
                    AssistantMessage(
                        tool_calls=(
                            ToolCall(
                                "task-1",
                                "task",
                                {
                                    "description": "Analyze cancellation",
                                    "prompt": "Wait until cancelled.",
                                    "subagent_type": "reasoning",
                                },
                            ),
                        )
                    )
                )
            )

    model = BlockingChildModel()
    definitions = builtins(model)
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent"),
        extensions=[
            SessionPersistenceExtension(storage),
            SubAgentExtension(definitions),
        ],
    )
    task = asyncio.create_task(agent.run("Delegate a blocking task"))
    await asyncio.wait_for(model.child_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.state.phase == AgentPhase.CANCELLED
    subagent_storage = child_storage(definitions)
    child = next(summary for summary in subagent_storage.list_sessions() if summary.parent_session_id == "parent")
    assert child.message_count == 1
    assert [record.message.content for record in subagent_storage.list_raw_messages(child.session_id)] == [
        "Wait until cancelled."
    ]


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        (
            {"name": "Bad Name", "description": "Description", "system_prompt": "Prompt"},
            "lowercase letters",
        ),
        ({"name": "valid", "description": "", "system_prompt": "Prompt"}, "description"),
        ({"name": "valid", "description": "Description", "system_prompt": ""}, "system_prompt"),
        (
            {"name": "valid", "description": "Description", "system_prompt": "Prompt", "max_iterations": 0},
            "max_iterations",
        ),
    ],
)
def test_subagent_definition_rejects_invalid_configuration(definition, message):
    with pytest.raises(ValueError, match=message):
        SubAgentDefinition(model=DelegatingModel(), **definition)


def test_subagent_extension_requires_unique_nonempty_definitions(storage):
    model = DelegatingModel()
    definition = SubAgentDefinition("custom", "Custom work", "Return a report", model)
    with pytest.raises(ValueError, match="At least one"):
        SubAgentExtension(())
    with pytest.raises(ValueError, match="names must be unique"):
        SubAgentExtension((definition, definition))


async def test_default_subagents_do_not_require_parent_persistence(builtins, monkeypatch):
    model = DelegatingModel()
    definitions = builtins(model)
    monkeypatch.setattr("zett_agent.extensions.subagent.default_subagents", lambda resolved_model: definitions)
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent"),
        extensions=[SubAgentExtension()],
    )

    result = await agent.run("Delegate this")

    assert result.content == "I used the delegated report"


async def test_custom_subagent_extensions_run_inside_the_child_lifecycle(storage):
    observed_sessions: list[str] = []

    class ChildExtension(AgentExtension):
        async def on_state(self, context):
            observed_sessions.append(context.config.session_id)
            context.state.messages.append(SystemMessage(content="Injected by child extension"))

    model = DelegatingModel(subagent_type="custom")
    persistence = SessionPersistenceExtension(storage)
    definition = SubAgentDefinition(
        "custom",
        "run work with a custom lifecycle extension.",
        "You are a custom subagent.",
        model,
        extensions=(persistence, ChildExtension()),
    )
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent"),
        extensions=[
            persistence,
            SubAgentExtension((definition,)),
        ],
    )

    result = await agent.run("Delegate with an extension")

    assert result.content == "I used the delegated report"
    assert len(observed_sessions) == 1
    child_request = next(
        request for request in model.requests if request.messages[0].content.startswith("You are a custom")
    )
    assert any(
        isinstance(message, SystemMessage) and message.content == "Injected by child extension"
        for message in child_request.messages
    )


async def test_subagent_uses_its_definition_model_instead_of_the_parent_model(storage):
    class ChildModel:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Answer from child model")))

    parent_model = DelegatingModel(subagent_type="custom")
    child_model = ChildModel()
    persistence = SessionPersistenceExtension(storage)
    definition = SubAgentDefinition(
        "custom",
        "answer with a separately configured model.",
        "You are a custom subagent.",
        child_model,
        extensions=(persistence,),
    )
    agent = await Agent.create(
        parent_model,
        config=AgentRunConfig("parent"),
        extensions=[
            persistence,
            SubAgentExtension((definition,)),
        ],
    )

    await agent.run("Delegate to another model")

    assert len(child_model.requests) == 1
    tool_result = next(message for message in parent_model.requests[-1].messages if isinstance(message, ToolMessage))
    assert json.loads(tool_result.content)["content"] == "Answer from child model"


async def test_custom_definition_does_not_receive_implicit_persistence(storage):
    model = DelegatingModel(subagent_type="custom")
    definition = SubAgentDefinition(
        "custom",
        "run an intentionally ephemeral child.",
        "You are a custom subagent.",
        model,
    )
    agent = await Agent.create(
        model,
        config=AgentRunConfig("parent"),
        extensions=[
            SessionPersistenceExtension(storage),
            SubAgentExtension((definition,)),
        ],
    )

    await agent.run("Delegate without child persistence")

    summaries = storage.list_sessions()
    assert [summary.session_id for summary in summaries] == ["parent"]
    assert summaries[0].message_count == 4


def test_agent_config_rejects_invalid_parent_session_ids():
    with pytest.raises(ValueError, match="parent_session_id cannot be empty"):
        AgentRunConfig("child", parent_session_id=" ")
    with pytest.raises(ValueError, match="must differ"):
        AgentRunConfig("same", parent_session_id="same")
