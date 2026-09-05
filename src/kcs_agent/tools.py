import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import ConfigDict, TypeAdapter, create_model

from .model import ToolDefinition


@dataclass(slots=True)
class AgentTool:
    """A tool is a name, an input schema, and an async callable."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    guidelines: str = ""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, self.parameters)

    async def __call__(self, arguments: Mapping[str, Any]) -> Any:
        """Execute with model-provided keyword arguments."""
        return await self.handler(**arguments)

    def serialize_result(self, value: Any) -> str:
        """Convert Pydantic models, dataclasses, and plain values into JSON."""
        return json.dumps(TypeAdapter(Any).dump_python(value, mode="json"), ensure_ascii=False)


def tool(function: Callable[..., Any] | None = None, *, name: str | None = None, guidelines: str = ""):
    """Turn a typed function into an AgentTool.

    Examples:
        @tool
        def add(left: int, right: int) -> int:
            \"\"\"Add two integer values.\"\"\"
            return left + right

        @tool(guidelines="Read the file before changing it.")
        def read_file(path: str) -> str:
            \"\"\"Read a text file.\"\"\"
            return path
    """

    def decorate(function: Callable[..., Any]) -> AgentTool:
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
        description = (inspect.getdoc(function) or "").split("\n\n", 1)[0]
        if not description:
            raise ValueError("A tool needs a docstring description")

        async def invoke(**arguments: Any) -> Any:
            parsed = inputs.model_validate(arguments)
            values = {field: getattr(parsed, field) for field in fields}
            if inspect.iscoroutinefunction(function):
                return await function(**values)
            return await asyncio.to_thread(function, **values)

        return AgentTool(name or function.__name__, description, inputs.model_json_schema(), invoke, guidelines)

    return decorate(function) if function is not None else decorate


def render_tool_guidance(tools: Sequence[AgentTool]) -> str:
    """Append optional author-written usage rules after the main system prompt."""
    rules = [f"- {tool.name}: {tool.guidelines}" for tool in tools if tool.guidelines]
    return "# Tool guidelines\n" + "\n".join(rules) if rules else ""
