"""AGENTS.md discovery, ordering, and system-message injection."""

from pathlib import Path

import pytest

from zett_agent.agent import (
    Agent,
    AgentRunConfig,
    AgentRunContext,
    AgentState,
)
from zett_agent.events import AgentPhase
from zett_agent.extensions.agents_md import (
    AGENTS_FILE_NAME,
    AgentsMdExtension,
)
from zett_agent.messages import AssistantMessage, SystemMessage
from zett_agent.model import ModelEvent, ModelResponse


def write(directory: Path, content: str, *, name: str = AGENTS_FILE_NAME) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def context(*messages) -> AgentRunContext:
    return AgentRunContext(
        AgentRunConfig(session_id="agents-md"),
        AgentState(_messages=list(messages), phase=AgentPhase.READY),
        {},
        (),
    )


class RecordedModel:
    def __init__(self) -> None:
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))


async def test_missing_file_injects_nothing(tmp_path):
    state = context()

    await AgentsMdExtension(tmp_path).on_state(state)

    assert state.state.messages == ()


async def test_reads_working_directory_instructions(tmp_path):
    write(tmp_path, "# Rules\n\nRun the tests.")
    state = context()

    await AgentsMdExtension(tmp_path).on_state(state)

    message = state.state.messages[0]
    assert isinstance(message, SystemMessage)
    assert "# Project instructions" in message.content
    assert str(tmp_path / AGENTS_FILE_NAME) in message.content
    assert "Run the tests." in message.content


def test_parent_files_are_loaded_outermost_first(tmp_path):
    write(tmp_path, "root rules")
    nested = tmp_path / "packages" / "app"
    write(nested, "app rules")

    extension = AgentsMdExtension(nested)

    assert extension.sources() == (tmp_path / AGENTS_FILE_NAME, nested / AGENTS_FILE_NAME)
    content = extension.instructions()
    assert content.index("root rules") < content.index("app rules")


def test_include_parents_false_reads_only_the_directory(tmp_path):
    write(tmp_path, "root rules")
    nested = tmp_path / "app"
    write(nested, "app rules")

    extension = AgentsMdExtension(nested, include_parents=False)

    assert extension.sources() == (nested / AGENTS_FILE_NAME,)
    assert "app rules" in extension.instructions()
    assert "root rules" not in extension.instructions()


def test_oversized_file_is_truncated_with_a_marker(tmp_path):
    write(tmp_path, "q" * 100)

    content = AgentsMdExtension(tmp_path, max_chars=10).instructions()

    assert "q" * 10 in content
    assert "truncated at 10 characters" in content
    assert "q" * 11 not in content


def test_custom_file_name_is_respected(tmp_path):
    write(tmp_path, "custom rules", name="CLAUDE.md")

    content = AgentsMdExtension(tmp_path, file_name="CLAUDE.md").instructions()

    assert "custom rules" in content


async def test_instructions_follow_existing_system_messages(tmp_path):
    write(tmp_path, "rules")
    existing = SystemMessage(content="Base instructions")
    state = context(existing)

    await AgentsMdExtension(tmp_path).on_state(state)

    assert state.state.messages[0] is existing
    assert isinstance(state.state.messages[1], SystemMessage)
    assert "rules" in state.state.messages[1].content


@pytest.mark.parametrize("kwargs", [{"file_name": "  "}, {"file_name": None}, {"max_chars": 0}])
def test_rejects_invalid_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        AgentsMdExtension(tmp_path, **kwargs)


async def test_agent_sends_agents_md_instructions_to_the_model(tmp_path):
    write(tmp_path, "Always answer in French.")
    model = RecordedModel()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("agents-md-run"),
        extensions=[AgentsMdExtension(tmp_path)],
    )

    await agent.run("Hello")

    guidance = "\n".join(
        message.content for message in model.requests[0].messages if isinstance(message, SystemMessage)
    )
    assert "Always answer in French." in guidance
