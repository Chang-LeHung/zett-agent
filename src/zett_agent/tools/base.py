"""Typed tool definitions, docstring parsing, and prompt guidance."""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ParamSpec, Protocol, TypeVar, cast, get_type_hints, overload

from .._compat import StrEnum, TypeAliasType
from ..model import ToolDefinition
from ..sync_runtime import SyncMethodsMixin

_CURRENT_TOOL_CALL_ID: ContextVar[str | None] = ContextVar("zett_agent_tool_call_id", default=None)
# A composing execution environment may supply non-pooling workers so a sync
# parent waiting on another sync tool cannot exhaust the default thread pool.
_SYNC_TOOL_RUNNER: ContextVar[Callable[..., Awaitable[Any]] | None] = ContextVar("zett_sync_tool_runner", default=None)

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
    #: Model-facing rules rendered beside this tool in prompt guidance. Tools
    #: that share one rule through another channel, such as a server-level system
    #: message, may leave this empty so the rule is not repeated per tool.
    guidelines: tuple[str, ...]
    #: Optional invocation examples rendered into prompt guidance.
    snippet: str = ""
    #: JSON Schema describing this tool's return value, when its function has a return
    #: annotation that can be described. ``None`` means the tool says nothing about its
    #: result. The value is never validated against a result; it only tells the model what
    #: to expect, and only when a renderer asks for it.
    return_schema: dict[str, Any] | None = None
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
        if any(not guideline.strip() for guideline in self.guidelines):
            raise ValueError("Tool guidelines cannot be blank")
        if not isinstance(self.execution_mode, ToolExecutionMode):
            raise ValueError("execution_mode must be a ToolExecutionMode")
        if not isinstance(self.deferred, bool):
            raise ValueError("deferred must be a boolean")
        if not isinstance(self.local_tool_search, bool):
            raise ValueError("local_tool_search must be a boolean")
        if self.return_schema is not None and not isinstance(self.return_schema, dict):
            raise ValueError("return_schema must be a JSON schema object or None")
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


def _inline_schema_references(schema: dict[str, Any]) -> dict[str, Any]:
    """Expand every local ``$ref`` so a tool schema stands on its own.

    Pydantic publishes nested argument models under ``$defs`` and points at them
    with ``$ref``. Some provider gateways reject that indirection, so tool
    schemas ship the expanded form instead. A recursive model cannot be expanded
    at all and is rejected with an explicit error.
    """
    raw_definitions = schema.get("$defs")
    definitions = raw_definitions if isinstance(raw_definitions, Mapping) else {}
    prefix = "#/$defs/"

    def expand(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [expand(item, stack) for item in node]
        if not isinstance(node, Mapping):
            return node
        reference = node.get("$ref")
        # Pydantic adds an OpenAPI ``discriminator`` whose ``mapping`` values are
        # ``$ref`` strings. Those cannot survive inlining, and the ``oneOf``
        # members keep their own ``const`` constraints, so the keyword is dropped.
        siblings = {key: value for key, value in node.items() if key not in {"$ref", "$defs", "discriminator"}}
        if reference is None:
            return {key: expand(value, stack) for key, value in siblings.items()}
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise ValueError(f"Tool arguments use an unsupported schema reference: {reference!r}")
        name = reference[len(prefix) :]
        if name in stack:
            raise ValueError(
                f"Tool arguments cannot use the recursive model {name!r}: a self-referencing schema "
                "cannot be expanded for providers that reject $ref"
            )
        target = definitions.get(name)
        if not isinstance(target, Mapping):
            raise ValueError(f"Tool arguments reference a missing schema definition: {name!r}")
        # Sibling keywords such as a field description override the definition.
        return {**expand(target, (*stack, name)), **expand(siblings, stack)}

    return expand({key: value for key, value in schema.items() if key != "$defs"}, ())


def _build_input_schema(
    function: Callable[..., Any], documentation: _ToolDocumentation
) -> tuple[type[Any], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    """Build the validator and JSON schema for one typed function.

    The resolved type hints come back with the schema, so the return annotation can be
    described from the same lookup instead of a second ``get_type_hints`` call. That
    call resolves every annotation at once, so a broken one fails here, with the
    arguments' error, whoever asks about the return value.
    """
    from pydantic import ConfigDict, create_model

    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, Any] = {}  # name -> (annotation, default), as ``create_model`` takes them
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
    parameters = _strip_schema_keywords(_inline_schema_references(inputs.model_json_schema()), _TITLE_KEYWORDS)
    unknown_parameters = documentation.parameter_descriptions.keys() - fields.keys()
    if unknown_parameters:
        unknown = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Docstring Args contains unknown parameters: {unknown}")
    for described, description in documentation.parameter_descriptions.items():
        parameters["properties"][described]["description"] = description
    return inputs, tuple(fields), parameters, hints


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
    inputs, field_names, parameters, hints = _build_input_schema(function, documentation)

    async def invoke(**arguments: Any) -> Any:
        parsed = inputs.model_validate(arguments)
        values = {field: getattr(parsed, field) for field in field_names}
        if inspect.iscoroutinefunction(function):
            return await function(**values)
        if (runner := _SYNC_TOOL_RUNNER.get()) is not None:
            return await runner(function, **values)
        return await asyncio.to_thread(function, **values)

    # The handler stands in for the user's function, so it reports the module
    # that function was defined in. Loaders use this to tell a tool a package
    # defines from one it merely imports.
    invoke.__module__ = getattr(function, "__module__", invoke.__module__)
    resolved_guidelines = _resolve_guidelines(documentation, guidelines)
    if not resolved_guidelines:
        raise ValueError("A tool needs at least one non-empty guideline")
    return AgentTool(
        name=name or function.__name__,
        description=documentation.description,
        parameters=parameters,
        handler=invoke,
        guidelines=resolved_guidelines,
        snippet=snippet if snippet is not None else documentation.snippet,
        return_schema=_build_return_schema(hints),
        execution_mode=execution_mode,
        deferred=deferred,
        local_tool_search=local_tool_search,
    )


def _build_return_schema(hints: Mapping[str, Any]) -> dict[str, Any] | None:
    """Describe a tool's return value as JSON Schema, or None when it cannot be described.

    The hint is optional metadata, and nothing validates a result against it, so every
    way of failing to describe it ends in None rather than in an error that would keep
    a working tool from being registered: no return annotation, an annotation Pydantic
    cannot build a schema for, a recursive model the inliner refuses, or a schema with
    nothing in it (``-> object`` and ``-> Any`` describe nothing). Title annotations
    are dropped because they cost prompt tokens on every request and say only the
    class name; a field named ``title`` is a name and stays.
    """
    hint = hints.get("return")
    if hint is None:
        return None
    from pydantic import TypeAdapter

    try:
        schema = TypeAdapter(hint).json_schema()
    except Exception:  # an annotation pydantic cannot describe is simply no hint
        return None
    if not schema:
        return None
    try:
        return _strip_schema_keywords(_inline_schema_references(schema), _TITLE_KEYWORDS)
    except Exception:  # a recursive model cannot be inlined, and the return is optional
        return None


# JSON Schema applicator keywords whose value holds sub-schemas: one, a list of them,
# or a map of them keyed by names. The walk decides how to descend by the value's own
# shape, so a keyword stays in one set no matter whether a draft lets it hold a single
# schema, a list, or both. The keys of a name map are property, definition, or schema
# names, so the walk recurses into their values instead of filtering their keys.
# ``dependencies`` also accepts a list of property names, which ``clean`` passes through,
# and ``dependentRequired`` maps names to names, so it holds no annotations at all.
_NAME_MAPS = frozenset({"properties", "patternProperties", "dependentSchemas", "$defs", "definitions", "dependencies"})
_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "additionalItems",
        "unevaluatedItems",
        "additionalProperties",
        "unevaluatedProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "contentSchema",
        "anyOf",
        "oneOf",
        "allOf",
        "prefixItems",
    }
)
# Pydantic titles every model and property with the name the property already has as
# its key, so the annotation repeats the contract instead of adding to it.
_TITLE_KEYWORDS = frozenset({"title"})


def _strip_schema_keywords(schema: Mapping[str, Any], keywords: Collection[str]) -> dict[str, Any]:
    """Drop annotation keywords from a schema while keeping every name it declares.

    Pydantic titles every model and property with its own name, and a model sees the
    property name as the argument's key already, so ``title`` only costs prompt tokens.
    A property *called* ``title`` is a name, not an annotation: it survives with its
    own entry, and only the ``title`` annotation beside it goes. The caller owns
    ``keywords`` because MCP schemas also carry a ``$schema`` URL to drop.

    Schema nodes are copied; data values such as ``examples``, ``enum``, and
    ``required`` are shared with the input.
    """

    def clean(node: Any) -> Any:
        if not isinstance(node, Mapping):
            return node
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in keywords:
                continue
            if key in _NAME_MAPS and isinstance(value, Mapping):
                result[key] = {name: clean(child) for name, child in value.items()}
            elif key in _SUBSCHEMA_KEYWORDS:
                # One keyword holds a single schema in some drafts and a list of them in
                # others, so the value's shape picks the descent.
                result[key] = [clean(item) for item in value] if isinstance(value, list) else clean(value)
            else:
                result[key] = value
        return result

    return cast(dict[str, Any], clean(schema))


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


def render_tool_guidance(
    tools: Sequence[AgentTool], *, snippets: bool = True, guidelines: bool = True, returns: bool = False
) -> str:
    """Render grouped tool metadata for insertion into a system prompt.

    Snippets are presented together so the model can scan the available call
    shapes first. Guidelines follow in one named group per tool, which avoids
    repeating a tool name for every rule. ``snippets`` and ``guidelines`` switch
    either section off. ``returns`` adds one line after each snippet describing
    the tool's return value as JSON Schema; it is off by default, because most
    tools return plain text the model already reads.
    """
    rendered_snippets = [
        _render_tool_snippet(registered, returns=returns)
        for registered in tools
        if snippets and (registered.snippet or (returns and registered.return_schema is not None))
    ]
    guideline_groups = [
        f"## {registered.name}\n" + "\n".join(f"- {guideline}" for guideline in registered.guidelines)
        for registered in tools
        if guidelines and registered.guidelines
    ]
    sections = []
    if rendered_snippets:
        sections.append("# Tool snippets\n" + "\n".join(rendered_snippets))
    if guideline_groups:
        sections.append("# Tool guidelines\n" + "\n\n".join(guideline_groups))
    return "\n\n".join(sections)


def render_tool_description(
    tool: AgentTool, *, snippet: bool = True, guidelines: bool = True, returns: bool = False, base: str | None = None
) -> str:
    """Return a tool description that carries its snippet and guidelines.

    Some endpoints show the model only a tool's ``description``, and a deferred
    tool reaches the model through tool search with no system prompt beside it.
    Folding the guidance into the description keeps it next to the tool in both
    cases. The result is the plain description when the tool has nothing to add,
    so it is safe to call for every tool.

    Args:
        tool: Tool whose metadata is rendered.
        snippet: Include the ``Snippet`` section when the tool has one.
        guidelines: Include the ``Guidelines`` section when the tool has any.
        returns: Add the tool's return value schema to the snippet section.
        base: Text to put the sections after, instead of ``tool.description``.
            Extensions that each add to a request's descriptions pass the text
            they received, so one does not overwrite what another appended.

    Examples:
        Render everything the model should know in one string::

            text = render_tool_description(read_file)
    """
    sections = [tool.description if base is None else base]
    if snippet and (tool.snippet or (returns and tool.return_schema is not None)):
        bullets = [f"- {line}" for line in tool.snippet.splitlines() if line.strip()]
        if returns and tool.return_schema is not None:
            bullets.append(f"- returns {_schema_text(tool.return_schema)}")
        sections.append("Snippet:\n" + "\n".join(bullets))
    if guidelines and tool.guidelines:
        sections.append("Guidelines:\n" + "\n".join(f"- {guideline}" for guideline in tool.guidelines))
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


def _schema_text(schema: Mapping[str, Any]) -> str:
    """Render a JSON schema on one line, so it stays inside its tool bullet."""
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_tool_snippet(tool: AgentTool, *, returns: bool = False) -> str:
    """Keep every line of a multiline snippet inside its tool bullet.

    Blank lines are dropped: a snippet is a handful of call shapes, and the gesture
    between two of them says nothing to the model.
    """
    lines = [line for line in tool.snippet.splitlines() if line.strip()]
    suffix = ""
    if returns and tool.return_schema is not None:
        suffix = f"\n  returns {_schema_text(tool.return_schema)}"
    if len(lines) == 1:
        return f"- {tool.name}: {lines[0]}{suffix}"
    if not lines:
        return f"- {tool.name}:{suffix}"
    body = "\n".join(f"  {line}" for line in lines)
    return f"- {tool.name}:\n{body}{suffix}"
