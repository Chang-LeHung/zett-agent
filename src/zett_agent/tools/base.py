"""Typed tool definitions, docstring parsing, and prompt guidance."""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ParamSpec, Protocol, TypeVar, get_type_hints, overload

from .._compat import StrEnum, TypeAliasType
from ..model import ToolDefinition
from ..sync_runtime import SyncMethodsMixin

_CURRENT_TOOL_CALL_ID: ContextVar[str | None] = ContextVar("zett_agent_tool_call_id", default=None)

#: Raw result of one local tool invocation, before the runtime turns it into a
#: ``ToolMessage``. ``TextContent`` and ``ImageContent`` parts (or a list of
#: them) become multimodal content, strings pass through unchanged, and every
#: other value is serialized by Pydantic, which covers JSON values, Pydantic
#: models, and dataclasses. The runtime never inspects the value itself, so this
#: alias stays wide instead of claiming an exhaustive union.
ToolResult = TypeAliasType("ToolResult", Any)

_DecoratedParameters = ParamSpec("_DecoratedParameters")
_DecoratedResult = TypeVar("_DecoratedResult")


@lru_cache(maxsize=1)
def _result_serializer() -> Any:
    """Build the permissive JSON adapter once, on the first serialized result.

    Pydantic is imported here rather than at module scope so importing the tool
    framework does not pay for it; only serializing a non-string result does.
    """
    from pydantic import TypeAdapter

    return TypeAdapter(Any)


def _bind_tool_call(call_id: str) -> Token[str | None]:
    """Bind one invocation ID to its isolated asyncio task context."""
    return _CURRENT_TOOL_CALL_ID.set(call_id)


def _reset_tool_call(token: Token[str | None]) -> None:
    """Restore the task's previous invocation identity."""
    _CURRENT_TOOL_CALL_ID.reset(token)


def current_tool_call_id() -> str | None:
    """Return the tool call currently executing in this task, when present."""
    return _CURRENT_TOOL_CALL_ID.get()


class ToolExecutionMode(StrEnum):
    """Local scheduling policy for one Agent tool."""

    #: Execute alone and only after every parallel tool in the response finishes.
    SERIAL = "serial"
    #: The handler may run concurrently with other parallel handlers.
    PARALLEL = "parallel"


@dataclass(slots=True)
class AgentTool(SyncMethodsMixin):
    """A tool is a name, an input schema, and an async callable."""

    #: Unique name exposed to the model and used for dispatch.
    name: str
    #: Natural-language purpose used by the model to select this tool.
    description: str
    #: JSON Schema describing this tool's keyword arguments.
    parameters: dict[str, Any]
    #: Async handler that validates arguments and performs the local operation.
    handler: Callable[..., Awaitable[ToolResult]]
    #: Model-facing rules rendered beside this tool in prompt guidance.
    guidelines: tuple[str, ...]
    #: Optional invocation examples rendered into prompt guidance.
    snippet: str = ""
    #: Whether this local handler runs serially or may run in parallel.
    execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL
    #: Whether Responses API providers may defer loading this tool's schema
    #: through tool search. Other provider protocols ignore this flag.
    deferred: bool = False
    #: Whether this tool answers client-side tool search for one request.
    #:
    #: A search tool is internal: the model never calls it directly. The
    #: Responses adapter declares ``tool_search`` with ``execution: "client"``
    #: and routes every search request to this handler, whose result decides
    #: which tools the model may load. Other provider protocols reject it.
    local_tool_search: bool = False

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("A tool description cannot be empty")
        if not self.guidelines or any(not guideline.strip() for guideline in self.guidelines):
            raise ValueError("A tool needs at least one non-empty guideline")
        if not isinstance(self.execution_mode, ToolExecutionMode):
            raise ValueError("execution_mode must be a ToolExecutionMode")
        if not isinstance(self.deferred, bool):
            raise ValueError("deferred must be a boolean")
        if not isinstance(self.local_tool_search, bool):
            raise ValueError("local_tool_search must be a boolean")
        if self.local_tool_search and self.deferred:
            raise ValueError("A local tool search tool cannot also defer its own schema")

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            self.name,
            self.description,
            self.parameters,
            deferred=self.deferred,
            local_tool_search=self.local_tool_search,
        )

    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult:
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

    def serialize_result(self, value: ToolResult) -> str:
        """Convert Pydantic models, dataclasses, and plain values into JSON.

        Strings pass through unchanged so plain-text tool results reach the
        model without JSON quoting.
        """
        if isinstance(value, str):
            return value
        return json.dumps(_result_serializer().dump_python(value, mode="json"), ensure_ascii=False)


class _ToolDecorator(Protocol):
    """Decorator returned by configured ``@tool(...)`` usage."""

    def __call__(self, function: Callable[_DecoratedParameters, _DecoratedResult], /) -> AgentTool: ...


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


def get_tool_snippet(function: Callable[..., Any]) -> str:
    """Extract the ``Snippet`` section from a function docstring.

    Args:
        function: Undecorated Python function whose docstring follows the tool
            documentation format.

    Returns:
        The normalized snippet, or an empty string when the section is absent.
    """
    return _parse_tool_docstring(function).snippet


def get_tool_guidelines(function: Callable[..., Any]) -> tuple[str, ...]:
    """Extract immutable ``Guidelines`` entries from a function docstring.

    Args:
        function: Undecorated Python function whose docstring follows the tool
            documentation format.
    """
    return _parse_tool_docstring(function).guidelines


def _build_input_schema(
    function: Callable[..., Any], documentation: _ToolDocumentation
) -> tuple[type[Any], tuple[str, ...], dict[str, Any]]:
    """Build the validator and JSON schema for one typed function."""
    from pydantic import ConfigDict, create_model

    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ValueError("Tools accept only named parameters")
        if parameter.name not in hints:
            raise ValueError(f"Missing annotation for {parameter.name}")
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (hints[parameter.name], default)

    # Build the validator in the tool's own module so a forward reference in its
    # annotations resolves there instead of inside this framework module.
    inputs = create_model(
        "ToolInput",
        __config__=ConfigDict(extra="forbid"),
        __module__=getattr(function, "__module__", __name__),
        **fields,
    )
    parameters = inputs.model_json_schema()
    unknown_parameters = documentation.parameter_descriptions.keys() - fields.keys()
    if unknown_parameters:
        unknown = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Docstring Args contains unknown parameters: {unknown}")
    for parameter, description in documentation.parameter_descriptions.items():
        parameters["properties"][parameter]["description"] = description
    return inputs, tuple(fields), parameters


def _resolve_guidelines(documentation: _ToolDocumentation, configured: str | Sequence[str] | None) -> tuple[str, ...]:
    """Resolve explicit decorator guidance before docstring guidance."""
    match configured:
        case None:
            return documentation.guidelines
        case str():
            return (configured,)
        case _:
            return tuple(configured)


def _build_agent_tool(
    function: Callable[..., Any],
    *,
    name: str | None,
    snippet: str | None,
    guidelines: str | Sequence[str] | None,
    execution_mode: ToolExecutionMode,
    deferred: bool,
    local_tool_search: bool,
) -> AgentTool:
    """Create an AgentTool from one function and decorator overrides."""
    documentation = _parse_tool_docstring(function)
    inputs, field_names, parameters = _build_input_schema(function, documentation)

    async def invoke(**arguments: Any) -> Any:
        parsed = inputs.model_validate(arguments)
        values = {field: getattr(parsed, field) for field in field_names}
        if inspect.iscoroutinefunction(function):
            return await function(**values)
        return await asyncio.to_thread(function, **values)

    return AgentTool(
        name=name or function.__name__,
        description=documentation.description,
        parameters=parameters,
        handler=invoke,
        guidelines=_resolve_guidelines(documentation, guidelines),
        snippet=snippet if snippet is not None else documentation.snippet,
        execution_mode=execution_mode,
        deferred=deferred,
        local_tool_search=local_tool_search,
    )


@overload
def tool(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    snippet: str | None = None,
    guidelines: str | Sequence[str] | None = None,
    execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL,
    deferred: bool = False,
    local_tool_search: bool = False,
) -> AgentTool: ...


@overload
def tool(
    function: None = None,
    *,
    name: str | None = None,
    snippet: str | None = None,
    guidelines: str | Sequence[str] | None = None,
    execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL,
    deferred: bool = False,
    local_tool_search: bool = False,
) -> _ToolDecorator: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    snippet: str | None = None,
    guidelines: str | Sequence[str] | None = None,
    execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL,
    deferred: bool = False,
    local_tool_search: bool = False,
) -> AgentTool | _ToolDecorator:
    """Turn a typed function and its structured docstring into an AgentTool.

    Args:
        function: Typed function for bare @tool use; omitted for @tool(...).
        name: Optional public name overriding the Python function name.
        snippet: Optional usage snippet overriding the docstring section.
        guidelines: Optional model guidance overriding docstring Guidelines.
        execution_mode: Whether the local handler must run serially or may run
            concurrently with other parallel handlers. The default is parallel;
            use SERIAL for handlers that require exclusive ordered execution.
        deferred: Whether Responses API providers may load this tool's schema
            through tool search instead of injecting it up front. Other provider
            protocols ignore this flag. Tool search requires a compatible model
            such as gpt-5.4 or later.
        local_tool_search: Whether this tool answers client-side tool search.
            The Responses adapter declares ``tool_search`` with
            ``execution: "client"`` and hands every search query to this
            handler; it must return ``list[FunctionToolParam]`` (OpenAI's
            Responses function definition), which becomes the
            ``tool_search_output`` the model loads from. The model never calls
            this tool as a function, and other provider protocols reject the
            flag.

    Returns:
        An AgentTool for direct decoration, or a decorator when configured first.

    Raises:
        ValueError: If annotations, guidance, or documented parameter names are invalid.

    Note:
        A normal exception raised by the decorated handler is caught by the
        Agent instead of ending the run. The Agent creates an unsuccessful
        ``ToolMessage`` whose ``content`` is ``str(exception)``; that string is
        what the next model request sees. The original exception object remains
        available separately on the ``TOOL_FAILED`` event and after-tool hooks.
        Cancellation and other ``BaseException`` subclasses are not converted.

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

    def decorate(target: Callable[..., Any]) -> AgentTool:
        return _build_agent_tool(
            target,
            name=name,
            snippet=snippet,
            guidelines=guidelines,
            execution_mode=execution_mode,
            deferred=deferred,
            local_tool_search=local_tool_search,
        )

    return decorate(function) if function is not None else decorate


def render_tool_guidance(tools: Sequence[AgentTool]) -> str:
    """Render grouped tool metadata for insertion into a system prompt.

    Snippets are presented together so the model can scan the available call
    shapes first. Guidelines follow in one named group per tool, which avoids
    repeating a tool name for every rule.
    """
    snippets = [_render_tool_snippet(registered) for registered in tools if registered.snippet]
    guideline_groups = [
        f"## {registered.name}\n" + "\n".join(f"- {guideline}" for guideline in registered.guidelines)
        for registered in tools
        if registered.guidelines
    ]
    sections = []
    if snippets:
        sections.append("# Tool snippets\n" + "\n".join(snippets))
    if guideline_groups:
        sections.append("# Tool guidelines\n" + "\n\n".join(guideline_groups))
    return "\n\n".join(sections)


def render_tool_search_text(tool: AgentTool) -> str:
    """Return every model-facing text field of one tool for search indexing.

    The result includes the normalized tool name, description, parameter
    descriptions from its JSON Schema, snippet, and guidelines. Search
    extensions can index this text without knowing how ``AgentTool`` stores or
    renders those fields.
    """
    properties = tool.parameters.get("properties")
    parameter_descriptions = (
        " ".join(str(value.get("description", "")) for value in properties.values() if isinstance(value, Mapping))
        if isinstance(properties, Mapping)
        else ""
    )
    return " ".join(
        (
            tool.name.replace("_", " "),
            tool.description,
            parameter_descriptions,
            tool.snippet,
            " ".join(tool.guidelines),
        )
    )


def _render_tool_snippet(tool: AgentTool) -> str:
    """Keep every line of a multiline snippet inside its tool bullet."""
    lines = tool.snippet.splitlines()
    if len(lines) == 1:
        return f"- {tool.name}: {lines[0]}"
    body = "\n".join(f"  {line}" if line else "" for line in lines)
    return f"- {tool.name}:\n{body}"
