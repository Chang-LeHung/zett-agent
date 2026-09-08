"""Typed tool definitions, docstring parsing, and prompt guidance."""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import ConfigDict, TypeAdapter, create_model

from ..model import ToolDefinition
from ..sync_runtime import SyncMethodsMixin


@dataclass(slots=True)
class AgentTool(SyncMethodsMixin):
    """A tool is a name, an input schema, and an async callable."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    guidelines: tuple[str, ...]
    snippet: str = ""

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("A tool description cannot be empty")
        if not self.guidelines or any(not guideline.strip() for guideline in self.guidelines):
            raise ValueError("A tool needs at least one non-empty guideline")

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, self.parameters)

    async def __call__(self, arguments: Mapping[str, Any]) -> Any:
        """Execute with a mapping of model-provided keyword arguments.

        Args:
            arguments: Input object validated by the decorated function's schema.

        Returns:
            The function's typed result before JSON serialization.

        Examples:
            Invoke a decorated tool from Python::

                result = await read_file({"path": "README.md"})

        Note:
            Decorated synchronous functions run in a worker thread; async
            functions are awaited directly. A manually supplied handler owns
            its validation policy.
        """
        return await self.handler(**arguments)

    def serialize_result(self, value: Any) -> str:
        """Convert Pydantic models, dataclasses, and plain values into JSON."""
        return json.dumps(TypeAdapter(Any).dump_python(value, mode="json"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class _ToolDocumentation:
    """Structured metadata parsed from one tool function docstring."""

    description: str
    parameter_descriptions: dict[str, str]
    snippet: str
    guidelines: tuple[str, ...]


def _parse_tool_docstring(function: Callable[..., Any]) -> _ToolDocumentation:
    """Parse Description, Args, Snippet, and Guidelines from a Google-style docstring."""
    document = inspect.getdoc(function) or ""
    lines = document.splitlines()
    description_lines = []
    for line in lines:
        if not line.strip():
            break
        description_lines.append(line.strip())
    description = " ".join(description_lines)
    if not description:
        raise ValueError("A tool needs a docstring description")

    sections: dict[str, list[str]] = {"args": [], "snippet": [], "guidelines": []}
    aliases = {"arguments": "args", "parameters": "args"}
    current: str | None = None
    for line in lines[len(description_lines) :]:
        heading = line.strip().removesuffix(":").lower()
        heading = aliases.get(heading, heading)
        if heading in sections and line.strip().endswith(":"):
            current = heading
            continue
        if current is not None:
            sections[current].append(line)

    parameter_descriptions: dict[str, str] = {}
    current_parameter: str | None = None
    for line in sections["args"]:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            parameter, description_part = stripped.split(":", 1)
            current_parameter = parameter.split("(", 1)[0].strip()
            parameter_descriptions[current_parameter] = description_part.strip()
        elif current_parameter is not None:
            parameter_descriptions[current_parameter] = (
                f"{parameter_descriptions[current_parameter]} {stripped}".strip()
            )

    snippet = "\n".join(line.strip() for line in sections["snippet"] if line.strip())
    guidelines = tuple(line.strip().removeprefix("-").strip() for line in sections["guidelines"] if line.strip())
    return _ToolDocumentation(description, parameter_descriptions, snippet, guidelines)


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    snippet: str | None = None,
    guidelines: str | Sequence[str] | None = None,
):
    """Turn a typed function and its structured docstring into an AgentTool.

    Args:
        function: Typed function for bare @tool use; omitted for @tool(...).
        name: Optional public name overriding the Python function name.
        snippet: Optional usage snippet overriding the docstring section.
        guidelines: Optional model guidance overriding docstring Guidelines.

    Returns:
        An AgentTool for direct decoration, or a decorator when configured first.

    Raises:
        ValueError: If annotations, guidance, or documented parameter names are invalid.

    Examples:
        Register a typed function::

            @tool
            def read_file(path: str) -> str:
                \"\"\"Read a text file.

                Args:
                    path: File path relative to the workspace.

                Snippet:
                    read_file(path=\"README.md\")

                Guidelines:
                    - Read a file before changing it.
                \"\"\"
                return path
    """

    def decorate(function: Callable[..., Any]) -> AgentTool:
        documentation = _parse_tool_docstring(function)
        hints = get_type_hints(function, include_extras=True)
        fields = {}
        for parameter in inspect.signature(function).parameters.values():
            if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise ValueError("Tools accept only named parameters")
            if parameter.name not in hints:
                raise ValueError(f"Missing annotation for {parameter.name}")
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (hints[parameter.name], default)
        inputs = create_model("ToolInput", __config__=ConfigDict(extra="forbid"), **fields)
        parameters = inputs.model_json_schema()
        unknown_parameters = documentation.parameter_descriptions.keys() - fields.keys()
        if unknown_parameters:
            unknown = ", ".join(sorted(unknown_parameters))
            raise ValueError(f"Docstring Args contains unknown parameters: {unknown}")
        for parameter, description in documentation.parameter_descriptions.items():
            parameters["properties"][parameter]["description"] = description

        async def invoke(**arguments: Any) -> Any:
            parsed = inputs.model_validate(arguments)
            values = {field: getattr(parsed, field) for field in fields}
            if inspect.iscoroutinefunction(function):
                return await function(**values)
            return await asyncio.to_thread(function, **values)

        if guidelines is None:
            resolved_guidelines = documentation.guidelines
        elif isinstance(guidelines, str):
            resolved_guidelines = (guidelines,)
        else:
            resolved_guidelines = tuple(guidelines)
        return AgentTool(
            name=name or function.__name__,
            description=documentation.description,
            parameters=parameters,
            handler=invoke,
            guidelines=resolved_guidelines,
            snippet=snippet if snippet is not None else documentation.snippet,
        )

    return decorate(function) if function is not None else decorate


def render_tool_guidance(tools: Sequence[AgentTool]) -> str:
    """Render all snippets first, followed by all usage guidelines."""
    snippets = [f"- {registered.name}: {registered.snippet}" for registered in tools if registered.snippet]
    guideline_groups = []
    for registered in tools:
        items = [f"- {guideline}" for guideline in registered.guidelines if guideline]
        if items:
            guideline_groups.append(f"## {registered.name}\n" + "\n".join(items))
    sections = []
    if snippets:
        sections.append("# Tool snippets\n" + "\n".join(snippets))
    if guideline_groups:
        sections.append("# Tool guidelines\n" + "\n\n".join(guideline_groups))
    return "\n\n".join(sections)
