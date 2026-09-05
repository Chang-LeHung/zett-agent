import asyncio
import inspect
import json
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from .model import ToolDefinition


@dataclass(slots=True)
class AgentTool:
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
        """Execute with model-provided keyword arguments."""
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

    Examples:
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
    guidelines = [
        f"- {registered.name}: {guideline}" for registered in tools for guideline in registered.guidelines if guideline
    ]
    sections = []
    if snippets:
        sections.append("# Tool snippets\n" + "\n".join(snippets))
    if guidelines:
        sections.append("# Tool guidelines\n" + "\n".join(guidelines))
    return "\n\n".join(sections)


class ReadFileResult(BaseModel):
    """A numbered slice of one UTF-8 text file."""

    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int
    has_more: bool


class WriteFileResult(BaseModel):
    """Result of atomically creating or replacing one UTF-8 text file."""

    path: str
    bytes_written: int
    created: bool


class ReplaceFileResult(BaseModel):
    """Result of replacing exact text in one UTF-8 text file."""

    path: str
    replacements: int
    bytes_written: int


class GlobResult(BaseModel):
    """Paths matched by one working-directory-relative glob pattern."""

    pattern: str
    paths: list[str]
    truncated: bool = False


class GrepMatch(BaseModel):
    """One matching text line and the first match position on that line."""

    path: str
    line_number: int
    column: int
    text: str


class GrepResult(BaseModel):
    """Text matches found across files in the current working directory."""

    pattern: str
    matches: list[GrepMatch]
    files_searched: int
    truncated: bool = False


class ShellResult(BaseModel):
    """Captured result of one shell command."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False


FilePath = Annotated[
    str,
    Field(min_length=1),
]
StartLine = Annotated[int, Field(ge=1)]
LineCount = Annotated[int, Field(ge=1, le=2_000)]
TimeoutSeconds = Annotated[int, Field(ge=1, le=300)]
MaxResults = Annotated[int, Field(ge=1, le=5_000)]


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CHARACTERS = 100_000


def _resolve_working_path(path: str) -> tuple[Path, Path]:
    """Resolve one relative path without allowing it to escape the current directory."""
    working_directory = Path.cwd().resolve()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        raise ValueError("Path must be relative to the current working directory")
    target = (working_directory / candidate).resolve()
    if not target.is_relative_to(working_directory):
        raise ValueError(f"Path escapes the current working directory: {path}")
    return working_directory, target


def _validate_working_pattern(pattern: str) -> Path:
    """Validate a glob pattern before evaluating it from the current directory."""
    candidate = Path(pattern).expanduser()
    if candidate.is_absolute():
        raise ValueError("Pattern must be relative to the current working directory")
    if ".." in candidate.parts:
        raise ValueError("Pattern cannot escape the current working directory")
    return candidate


def _read_working_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte limit")
    return path.read_text(encoding="utf-8")


def _write_working_text(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(f"Content exceeds the {MAX_FILE_BYTES}-byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temporary:
            temporary.write(encoded)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _truncate_output(output: str) -> tuple[str, bool]:
    if len(output) <= MAX_OUTPUT_CHARACTERS:
        return output, False
    return output[:MAX_OUTPUT_CHARACTERS], True


@tool
def glob(pattern: Annotated[str, Field(min_length=1)], max_results: MaxResults = 200) -> GlobResult:
    """Find paths matching a glob pattern in the current working directory.

    Args:
        pattern: Relative glob pattern such as **/*.py or src/**/test_*.py.
        max_results: Maximum number of sorted paths to return.

    Snippet:
        glob(pattern="src/**/*.py", max_results=200)

    Guidelines:
        - Use glob to discover files before reading or searching them.
        - Narrow the pattern when the result is truncated.
    """
    working_directory = Path.cwd().resolve()
    validated_pattern = _validate_working_pattern(pattern)
    paths: list[str] = []
    truncated = False
    for candidate in sorted(working_directory.glob(validated_pattern.as_posix())):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(working_directory):
            continue
        if len(paths) == max_results:
            truncated = True
            break
        paths.append(candidate.relative_to(working_directory).as_posix())
    return GlobResult(pattern=pattern, paths=paths, truncated=truncated)


@tool
def grep(
    pattern: Annotated[str, Field(min_length=1)],
    file_pattern: Annotated[str, Field(min_length=1)] = "**/*",
    case_sensitive: bool = True,
    max_results: MaxResults = 200,
) -> GrepResult:
    """Search UTF-8 text files with a regular expression from the current working directory.

    Args:
        pattern: Python regular expression to search for on each line.
        file_pattern: Relative glob selecting files to search.
        case_sensitive: Whether letter case must match exactly.
        max_results: Maximum number of matching lines to return.

    Snippet:
        grep(pattern="class\\s+Agent", file_pattern="src/**/*.py", max_results=200)

    Guidelines:
        - Use a narrow file_pattern to avoid scanning unrelated files.
        - Escape regular-expression characters when searching for literal text.
        - Use read_file for surrounding context after locating a match.
    """
    working_directory = Path.cwd().resolve()
    validated_file_pattern = _validate_working_pattern(file_pattern)
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        expression = re.compile(pattern, flags)
    except re.error as error:
        raise ValueError(f"Invalid regular expression: {error}") from error

    matches: list[GrepMatch] = []
    files_searched = 0
    truncated = False
    for candidate in sorted(working_directory.glob(validated_file_pattern.as_posix())):
        resolved = candidate.resolve()
        if not candidate.is_file() or not resolved.is_relative_to(working_directory):
            continue
        if candidate.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files_searched += 1
        for line_number, line in enumerate(content.splitlines(), start=1):
            match = expression.search(line)
            if match is None:
                continue
            if len(matches) == max_results:
                truncated = True
                break
            matches.append(
                GrepMatch(
                    path=candidate.relative_to(working_directory).as_posix(),
                    line_number=line_number,
                    column=match.start() + 1,
                    text=line,
                )
            )
        if truncated:
            break
    return GrepResult(pattern=pattern, matches=matches, files_searched=files_searched, truncated=truncated)


@tool
def read_file(path: FilePath, start_line: StartLine = 1, line_count: LineCount = 200) -> ReadFileResult:
    """Read a line range from a UTF-8 text file in the current working directory.

    Args:
        path: Relative path inside the current working directory.
        start_line: One-based first line to return.
        line_count: Maximum number of lines to return.

    Snippet:
        read_file(path="src/app.py", start_line=1, line_count=200)

    Guidelines:
        - Use line ranges for large files.
        - Inspect the current content before editing a file.
    """
    working_directory, target = _resolve_working_path(path)
    if not target.is_file():
        raise ValueError(f"File does not exist: {path}")
    lines = _read_working_text(target).splitlines(keepends=True)
    start_index = min(start_line - 1, len(lines))
    selected = lines[start_index : start_index + line_count]
    end_line = start_index + len(selected)
    return ReadFileResult(
        path=target.relative_to(working_directory).as_posix(),
        content="".join(selected),
        start_line=start_line,
        end_line=end_line,
        total_lines=len(lines),
        has_more=end_line < len(lines),
    )


@tool
def write_file(path: FilePath, content: str, overwrite: bool = True) -> WriteFileResult:
    """Atomically write a UTF-8 text file in the current working directory.

    Args:
        path: Relative path inside the current working directory.
        content: Complete UTF-8 text to write.
        overwrite: Whether an existing file may be replaced.

    Snippet:
        write_file(path="notes/plan.md", content="# Plan\n")

    Guidelines:
        - Use for new files or intentional full-file replacement.
        - Prefer replace_in_file for a small change to an existing file.
    """
    working_directory, target = _resolve_working_path(path)
    if target.exists() and not target.is_file():
        raise ValueError(f"Path is not a file: {path}")
    created = not target.exists()
    if not overwrite and not created:
        raise ValueError(f"File already exists: {path}")
    _write_working_text(target, content)
    return WriteFileResult(
        path=target.relative_to(working_directory).as_posix(),
        bytes_written=len(content.encode("utf-8")),
        created=created,
    )


@tool
def replace_in_file(
    path: FilePath,
    old_text: Annotated[str, Field(min_length=1)],
    new_text: str,
    replace_all: bool = False,
) -> ReplaceFileResult:
    """Replace exact text in a UTF-8 file, requiring one match by default.

    Args:
        path: Relative path inside the current working directory.
        old_text: Exact text to find.
        new_text: Replacement text.
        replace_all: Whether every exact match should be replaced.

    Snippet:
        replace_in_file(path="notes/plan.md", old_text="Draft", new_text="Final")

    Guidelines:
        - Keep replace_all false unless every occurrence should change.
        - Read the file first when the target text may be ambiguous.
    """
    working_directory, target = _resolve_working_path(path)
    if not target.is_file():
        raise ValueError(f"File does not exist: {path}")
    content = _read_working_text(target)
    matches = content.count(old_text)
    if matches == 0:
        raise ValueError("Text to replace was not found")
    if matches > 1 and not replace_all:
        raise ValueError(f"Text to replace is not unique; found {matches} matches")
    replacements = matches if replace_all else 1
    updated = content.replace(old_text, new_text, -1 if replace_all else 1)
    _write_working_text(target, updated)
    return ReplaceFileResult(
        path=target.relative_to(working_directory).as_posix(),
        replacements=replacements,
        bytes_written=len(updated.encode("utf-8")),
    )


@tool
async def run_shell(
    command: Annotated[str, Field(min_length=1)],
    timeout_seconds: TimeoutSeconds = 30,
) -> ShellResult:
    """Run a shell command with captured output from the current working directory.

    Args:
        command: Shell command to execute.
        timeout_seconds: Maximum command runtime in seconds.

    Snippet:
        run_shell(command="git status --short", timeout_seconds=30)

    Guidelines:
        - Use only for bounded commands in a trusted working directory.
        - Inspect exit_code and stderr before assuming the command succeeded.
    """
    if not command.strip():
        raise ValueError("Shell command cannot be blank")
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=Path.cwd(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        async with asyncio.timeout(timeout_seconds):
            stdout_bytes, stderr_bytes = await process.communicate()
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.communicate()
        raise
    stdout, stdout_truncated = _truncate_output(stdout_bytes.decode("utf-8", errors="replace"))
    stderr, stderr_truncated = _truncate_output(stderr_bytes.decode("utf-8", errors="replace"))
    return ShellResult(
        command=command,
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        output_truncated=stdout_truncated or stderr_truncated,
    )
