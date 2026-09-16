"""Skill discovery, prompt injection, and full instruction loading."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from zett_agent import (
    DEFAULT_SKILL_ROOTS,
    READ_SKILL_TOOL_NAME,
    Agent,
    AgentRunConfig,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    SkillExtension,
    SkillFileParser,
    SystemMessage,
    ToolCall,
    ToolGuidelinesExtension,
    ToolMessage,
)


def write_skill(root, directory: str, content: str):
    path = root / directory / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    return path.resolve()


def test_skill_extension_discovers_default_user_roots(tmp_path, monkeypatch):
    isolated_home = tmp_path / "home"
    isolated_home.mkdir()
    monkeypatch.setenv("HOME", str(isolated_home))
    monkeypatch.chdir(tmp_path)
    for root in DEFAULT_SKILL_ROOTS:
        client = Path(root).parts[-2].removeprefix(".")
        write_skill(
            isolated_home / Path(root).relative_to("~"),
            "example",
            f"---\nname: {client}\ndescription: Skill from {root}.\n---\nBody",
        )

    extension = SkillExtension()

    assert [skill.name for skill in extension.skills] == ["agent", "claude", "cursor", "zett"]


async def test_skill_extension_loads_user_skill_from_default_zett_directory_and_cleans_up(tmp_path, monkeypatch):
    complete = """---
name: home-review
description: Review code using the user-level workflow.
---
# Home review

Inspect every changed file and run focused checks.
"""

    class Model:
        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                message = AssistantMessage(
                    tool_calls=(ToolCall("home-skill-1", READ_SKILL_TOOL_NAME, {"name": "home-review"}),)
                )
            else:
                message = AssistantMessage(content="User skill loaded")
            yield ModelEvent.completed(ModelResponse(message))

    with TemporaryDirectory(prefix="zett-agent-skill-e2e-") as temporary_home:
        home = Path(temporary_home)
        skill_path = write_skill(home / ".zett" / "skills", "home-review", complete)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with monkeypatch.context() as isolated:
            isolated.setenv("HOME", str(home))
            isolated.chdir(workspace)
            model = Model()
            extension = SkillExtension()
            agent = await Agent.create(
                model,
                config=AgentRunConfig("home-skill-e2e"),
                extensions=[extension, ToolGuidelinesExtension()],
            )

            answer = await agent.run("Use my review skill")

            assert answer.content == "User skill loaded"
            assert extension.skills == (SkillFileParser().parse(skill_path),)
            assert [tool.name for tool in model.requests[0].tools] == [READ_SKILL_TOOL_NAME]
            catalog = next(
                message.content
                for message in model.requests[0].messages
                if isinstance(message, SystemMessage) and message.content.startswith("# Available skills")
            )
            assert f"home-review: Review code using the user-level workflow. (file: {skill_path})" in catalog
            result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
            assert result.content == complete

    assert not home.exists()


async def test_skill_extension_advertises_metadata_and_loads_complete_file(tmp_path):
    complete = """---
name: review
description: Review changes with focused checks.
---
# Review

Read every changed file before reporting findings.
"""
    path = write_skill(tmp_path, "review", complete)

    class Model:
        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                message = AssistantMessage(tool_calls=(ToolCall("skill-1", READ_SKILL_TOOL_NAME, {"name": "review"}),))
            else:
                message = AssistantMessage(content="Skill loaded")
            yield ModelEvent.completed(ModelResponse(message))

    model = Model()
    extension = SkillExtension([tmp_path])
    agent = await Agent.create(
        model,
        config=AgentRunConfig("skill-session"),
        extensions=[extension, ToolGuidelinesExtension()],
    )

    await agent.run("Review this change")

    assert extension.skills[0].name == "review"
    assert extension.skills[0].path == path
    assert [tool.name for tool in model.requests[0].tools] == [READ_SKILL_TOOL_NAME]
    catalog = next(
        message.content
        for message in model.requests[0].messages
        if isinstance(message, SystemMessage) and message.content.startswith("# Available skills")
    )
    assert "review: Review changes with focused checks." in catalog
    assert str(path) in catalog
    assert "Read every changed file" not in catalog
    result = next(message for message in model.requests[1].messages if isinstance(message, ToolMessage))
    assert result.content == complete


async def test_skill_extension_rejects_unknown_skill_without_reading_other_files(tmp_path):
    write_skill(tmp_path, "known", "---\nname: known\ndescription: Known skill.\n---\nSecret instructions")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("must not be read")
    extension = SkillExtension([tmp_path])

    with pytest.raises(ValueError, match="Unknown skill"):
        await extension._read_skill_tool({"name": str(outside)})


def test_skill_parser_accepts_folded_and_literal_descriptions(tmp_path):
    write_skill(
        tmp_path,
        "folded",
        "---\nname: folded\ndescription: >\n  A description split\n  over two lines.\n---\nBody",
    )
    write_skill(
        tmp_path,
        "literal",
        "---\nname: literal\ndescription: |\n  First line.\n  Second line.\n---\nBody",
    )

    skills = SkillExtension([tmp_path]).skills

    assert [(skill.name, skill.description) for skill in skills] == [
        ("folded", "A description split over two lines."),
        ("literal", "First line.\nSecond line."),
    ]


@pytest.mark.parametrize(
    "content",
    [
        "# Missing front matter\nBody",
        "---\nname: unclosed\ndescription: Missing delimiter.\nBody",
        "---\ndescription: Missing name.\n---\nBody",
        "---\nname: missing-description\n---\nBody",
        "---\nname: INVALID_NAME\ndescription: Invalid name.\n---\nBody",
        "---\nname: empty-body\ndescription: Empty body.\n---\n",
    ],
)
def test_skill_extension_skips_files_that_do_not_follow_the_skill_format(tmp_path, content):
    write_skill(tmp_path, "invalid", content)

    assert SkillExtension([tmp_path]).skills == ()


def test_skill_file_parser_rejects_non_skill_paths(tmp_path):
    parser = SkillFileParser()
    ordinary = tmp_path / "README.md"
    ordinary.write_text("---\nname: readme\ndescription: Not a skill.\n---\nBody")

    assert parser.parse(ordinary) is None
    assert parser.parse(tmp_path / "missing" / "SKILL.md") is None


def test_skill_extension_deduplicates_names_using_root_precedence(tmp_path):
    preferred = write_skill(
        tmp_path / "preferred",
        "skill",
        "---\nname: duplicate\ndescription: Preferred description.\n---\nPreferred body",
    )
    write_skill(
        tmp_path / "fallback",
        "skill",
        "---\nname: duplicate\ndescription: Fallback description.\n---\nFallback body",
    )

    extension = SkillExtension([tmp_path / "preferred", tmp_path / "fallback", tmp_path / "preferred"])

    assert len(extension.skills) == 1
    assert extension.skills[0].description == "Preferred description."
    assert extension.skills[0].path == preferred


async def test_skill_extension_is_inert_when_no_skills_exist(tmp_path):
    class Model:
        async def stream(self, request):
            assert request.tools == ()
            assert all("Available skills" not in message.content for message in request.messages)
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    extension = SkillExtension([tmp_path / "missing"])
    agent = await Agent.create(Model(), config=AgentRunConfig("empty-skills"), extensions=[extension])

    await agent.run("Hello")
    assert extension.skills == ()


def test_skill_extension_ignores_symlink_that_escapes_a_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = write_skill(tmp_path, "outside", "---\nname: outside\ndescription: Outside.\n---\nBody")
    (root / "linked").symlink_to(outside.parent, target_is_directory=True)

    assert SkillExtension([root]).skills == ()
