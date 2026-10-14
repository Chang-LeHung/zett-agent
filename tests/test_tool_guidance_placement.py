"""Where a tool's snippet and guidelines are shown: system prompt (default) or description."""

import pytest

from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.extensions.coding import CodingExtension
from zett_agent.extensions.plan_mode import PlanModeExtension
from zett_agent.extensions.tool_guidelines import ToolGuidancePlacement, ToolGuidelinesExtension
from zett_agent.messages import AssistantMessage, SystemMessage
from zett_agent.model import ModelEvent, ModelResponse
from zett_agent.tools.base import render_tool_description, render_tool_guidance, tool
from zett_agent.tools.coding import edit, read_file


@tool(guidelines=("Use for lookups.", "Never guess."))
async def lookup(key: str) -> str:
    """Look a key up.

    Args:
        key: Key to find.

    Snippet:
        lookup(key="a")
        lookup(key="b")
    """
    return key


@tool(deferred=True, guidelines="Only after a search.")
async def hidden() -> str:
    """Hidden until searched."""
    return "x"


@tool(guidelines="Bare.")
async def bare() -> str:
    """Has guidelines but no snippet."""
    return "x"


class Model:
    def __init__(self) -> None:
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="ok")))


async def request_for(*extensions, tools=(lookup, hidden, bare)):
    model = Model()
    agent = await Agent.create(
        model, config=AgentRunConfig("placement"), tools=list(tools), extensions=list(extensions)
    )
    await agent.run("hi")
    return model.requests[0]


def system_text(request) -> str:
    return "\n".join(m.content for m in request.messages if isinstance(m, SystemMessage))


def descriptions(request) -> dict[str, str]:
    return {definition.name: definition.description for definition in request.tools}


async def test_the_default_still_writes_guidance_into_the_system_prompt():
    request = await request_for(ToolGuidelinesExtension())

    assert "# Tool snippets" in system_text(request) and "## lookup" in system_text(request)
    assert descriptions(request)["lookup"] == "Look a key up."


async def test_description_placement_moves_guidance_into_each_tools_description():
    request = await request_for(ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION))

    assert "Tool snippets" not in system_text(request) and "Tool guidelines" not in system_text(request)
    assert descriptions(request)["lookup"] == (
        "Look a key up.\n\n"
        'Snippet:\n- lookup(key="a")\n- lookup(key="b")\n\n'
        "Guidelines:\n- Use for lookups.\n- Never guess."
    )
    # A tool with no snippet only gets the section it has.
    assert descriptions(request)["bare"] == "Has guidelines but no snippet.\n\nGuidelines:\n- Bare."


async def test_description_placement_skips_deferred_tools():
    request = await request_for(ToolGuidelinesExtension(placement="description"))

    assert descriptions(request)["hidden"] == "Hidden until searched."


@pytest.mark.parametrize(
    ("options", "has_snippet", "has_guidelines"),
    [
        ({"snippets": False}, False, True),
        ({"guidelines": False}, True, False),
        ({"snippets": False, "guidelines": False}, False, False),
    ],
)
async def test_each_part_can_be_switched_off_in_either_placement(options, has_snippet, has_guidelines):
    in_description = descriptions(
        await request_for(ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION, **options))
    )["lookup"]
    in_system = system_text(await request_for(ToolGuidelinesExtension(**options)))

    assert ("Snippet:" in in_description) is has_snippet and ("Guidelines:" in in_description) is has_guidelines
    assert ("# Tool snippets" in in_system) is has_snippet and ("# Tool guidelines" in in_system) is has_guidelines


async def test_the_registered_tools_and_system_messages_are_not_changed_by_description_placement():
    model = Model()
    agent = await Agent.create(
        model,
        config=AgentRunConfig("untouched"),
        tools=[lookup],
        extensions=[ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION)],
    )

    await agent.run("hi")
    await agent.run("again")

    assert lookup.description == "Look a key up."
    assert [descriptions(request)["lookup"] for request in model.requests][0] == descriptions(model.requests[1])[
        "lookup"
    ]
    assert not any(isinstance(m, SystemMessage) and "lookup" in m.content for m in model.requests[1].messages)


async def test_builtin_tools_carry_their_guidance_in_description_mode():
    request = await request_for(
        CodingExtension(), ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION), tools=()
    )

    assert "Guidelines:" in descriptions(request)["edit"]
    assert "Read the target lines with read_file first" in descriptions(request)["edit"]
    assert "Tool guidelines" not in system_text(request)


async def test_plan_mode_keeps_guidance_out_of_the_system_prompt_in_description_mode():
    request = await request_for(
        CodingExtension(),
        PlanModeExtension(),
        ToolGuidelinesExtension(placement=ToolGuidancePlacement.DESCRIPTION),
        tools=(),
    )

    assert "Tool guidelines" not in system_text(request) and "Tool snippets" not in system_text(request)
    assert "Guidelines:" in descriptions(request)["edit"]


async def test_plan_mode_honors_the_snippet_switch_in_the_system_prompt():
    request = await request_for(
        CodingExtension(), PlanModeExtension(), ToolGuidelinesExtension(snippets=False), tools=()
    )

    assert "# Tool snippets" not in system_text(request)
    assert "# Tool guidelines" in system_text(request)


def test_render_tool_description_is_the_plain_description_when_there_is_nothing_to_add():
    @tool(guidelines="x")
    async def quiet() -> str:
        """Quiet."""
        return ""

    assert render_tool_description(quiet, guidelines=False, snippet=False) == "Quiet."
    assert render_tool_description(read_file, snippet=False).startswith(read_file.description + "\n\nGuidelines:\n- ")
    assert "Snippet:\n- read_file(" in render_tool_description(read_file)


def test_render_tool_guidance_switches_are_independent():
    both = render_tool_guidance([edit])
    assert "# Tool snippets" in both and "# Tool guidelines" in both
    assert "# Tool guidelines" not in render_tool_guidance([edit], guidelines=False)
    assert "# Tool snippets" not in render_tool_guidance([edit], snippets=False)
    assert render_tool_guidance([edit], snippets=False, guidelines=False) == ""


def test_an_unknown_placement_is_rejected():
    with pytest.raises(ValueError):
        ToolGuidelinesExtension(placement="somewhere")
