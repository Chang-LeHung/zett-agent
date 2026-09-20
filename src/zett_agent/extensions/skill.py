"""Discover local agent skills and expose their instructions on demand."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..agent import AgentRunContext
from ..messages import SystemMessage
from ..tools import AgentTool, tool
from .base import AgentExtension

SKILL_FILE_NAME = "SKILL.md"
READ_SKILL_TOOL_NAME = "read_skill"
DEFAULT_SKILL_ROOTS = (
    "~/.zett/skills",
    "~/.agent/skills",
    "~/.claude/skills",
    "~/.cursor/skills",
)


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """Validated metadata for one discovered SKILL.md file."""

    name: str
    description: str
    path: Path


class SkillFileParser:
    """Validate and parse one Agent Skills-compatible SKILL.md file.

    A valid file uses UTF-8, starts with closed YAML-style front matter, and
    declares ``name`` and ``description``. Names use lowercase letters, digits,
    and single hyphens. The Markdown body must contain instructions. Unsupported
    or malformed files return ``None`` so discovery can continue independently.
    """

    _NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    _MAX_NAME_LENGTH = 64
    _MAX_DESCRIPTION_LENGTH = 1024

    def parse(self, path: str | Path) -> SkillDefinition | None:
        """Return validated metadata, or None when the file is not a valid skill."""
        resolved = Path(path).resolve()
        if resolved.name != SKILL_FILE_NAME or not resolved.is_file():
            return None
        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        front_matter, body = self._split_front_matter(content)
        if front_matter is None or not body.strip():
            return None
        name = self._front_matter_value(front_matter, "name")
        description = self._front_matter_value(front_matter, "description")
        if not self._valid_name(name) or not self._valid_description(description):
            return None
        return SkillDefinition(name=name, description=description, path=resolved)

    @staticmethod
    def _split_front_matter(content: str) -> tuple[list[str] | None, str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, content
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return lines[1:index], "\n".join(lines[index + 1 :])
        return None, content

    @staticmethod
    def _front_matter_value(lines: Sequence[str], key: str) -> str | None:
        prefix = f"{key}:"
        for index, line in enumerate(lines):
            if not line.startswith(prefix):
                continue
            value = line.removeprefix(prefix).strip()
            if value in {"|", ">"}:
                continuation = []
                for following in lines[index + 1 :]:
                    if following and not following[0].isspace():
                        break
                    if following.strip():
                        continuation.append(following.strip())
                separator = "\n" if value == "|" else " "
                return separator.join(continuation)
            return value.strip("\"'")
        return None

    def _valid_name(self, value: str | None) -> bool:
        return bool(value and len(value) <= self._MAX_NAME_LENGTH and self._NAME_PATTERN.fullmatch(value))

    def _valid_description(self, value: str | None) -> bool:
        return bool(value and value.strip() and len(value) <= self._MAX_DESCRIPTION_LENGTH)


class SkillExtension(AgentExtension):
    """Advertise valid local skills and register a tool that reads one in full.

    With no roots, discovery searches the user-level ``~/.zett/skills``,
    ``~/.agent/skills``, ``~/.claude/skills``, and ``~/.cursor/skills``
    directories. The system message contains only each skill's name,
    description, and location. Complete instructions enter context only after
    the model explicitly calls ``read_skill``. Pass ``roots`` explicitly when
    an application also wants project-local skills.

    Examples:
        Usage::

            extension = SkillExtension([Path(".agent/skills"), Path("~/.skills")])
            agent = await Agent.create(model, config=config, extensions=[extension])
    """

    priority = 80

    def __init__(
        self,
        roots: Sequence[str | Path] | None = None,
        *,
        parser: SkillFileParser | None = None,
    ) -> None:
        configured_roots = DEFAULT_SKILL_ROOTS if roots is None else roots
        # Root order defines precedence when several locations declare the same
        # skill name. Repeated resolved directories are scanned only once.
        self._roots = tuple(dict.fromkeys(Path(root).expanduser().resolve() for root in configured_roots))
        self._parser = parser or SkillFileParser()
        self._skills = self._discover()
        self._skills_by_name = {skill.name: skill for skill in self._skills}
        self._read_skill_tool = self._build_read_skill_tool()

    @property
    def skills(self) -> tuple[SkillDefinition, ...]:
        """Return discovered skills in deterministic name order."""
        return self._skills

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register skill loading only when at least one valid skill exists."""
        if self._skills:
            context.register_tool(self._read_skill_tool)

    async def on_state(self, context: AgentRunContext) -> None:
        """Place the skill catalog and its storage roots with the system messages."""
        if not self._skills:
            return
        message = SystemMessage(content=self._catalog())
        instructions = [item for item in context.state.messages if isinstance(item, SystemMessage)]
        context.add_message(message, index=len(instructions))

    def _catalog(self) -> str:
        """Describe where skills are stored and which ones are installed."""
        entries = "\n".join(f"- {skill.name}: {skill.description} (file: {skill.path})" for skill in self._skills)
        roots = "\n".join(f"- {root}" for root in self._roots)
        return (
            "# Available skills\n"
            "Skills are optional instruction sets. Each skill is one directory holding a `SKILL.md` file, and "
            "Zett Agent searches these roots in order; the first root that declares a name wins:\n"
            f"{roots}\n\n"
            f"When a skill is relevant, call `{READ_SKILL_TOOL_NAME}` with its name and follow the complete "
            "returned instructions. Skill bodies stay out of context until that call.\n\n"
            f"{entries}"
        )

    def _discover(self) -> tuple[SkillDefinition, ...]:
        skills: dict[str, SkillDefinition] = {}
        discovered_paths: set[Path] = set()
        for root in self._roots:
            if not root.is_dir():
                continue
            for candidate in sorted(root.rglob(SKILL_FILE_NAME)):
                resolved = candidate.resolve()
                if not resolved.is_file() or not resolved.is_relative_to(root) or resolved in discovered_paths:
                    continue
                discovered_paths.add(resolved)
                skill = self._parser.parse(resolved)
                if skill is not None:
                    skills.setdefault(skill.name, skill)
        return tuple(skills[name] for name in sorted(skills))

    def _build_read_skill_tool(self) -> AgentTool:
        skills = self._skills_by_name

        @tool(name=READ_SKILL_TOOL_NAME)
        def read_skill(name: str) -> str:
            """Read the complete SKILL.md instructions for one advertised skill.

            Args:
                name: Exact skill name shown in the available-skills system message.

            Snippet:
                read_skill(name="code-review")

            Guidelines:
                - Read a relevant skill before applying its workflow.
                - Use only names from the available-skills catalog.
            """
            skill = skills.get(name)
            if skill is None:
                available = ", ".join(sorted(skills)) or "none"
                raise ValueError(f"Unknown skill {name!r}; available skills: {available}")
            return skill.path.read_text(encoding="utf-8")

        return read_skill
