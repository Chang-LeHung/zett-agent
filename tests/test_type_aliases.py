"""Named type aliases stay uniform, resolvable, and faithful to their constraints."""

from __future__ import annotations

import ast
import importlib
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest
from pydantic import TypeAdapter, ValidationError

from zett_agent._compat import TypeAliasType
from zett_agent.extensions.base import MiddlewareHook, ModelRequestNext, ToolCallNext
from zett_agent.extensions.shell_approval import ShellCommandNext
from zett_agent.messages import (
    AgentMessage,
    AnyMessage,
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from zett_agent.model import ModelEvent, ModelRequest
from zett_agent.tools.base import ToolResult, tool
from zett_agent.tools.coding import LineCount

ROOT = Path(__file__).resolve().parents[1] / "src" / "zett_agent"
ALIASES = {
    "event_queue": ("_QueueItem",),
    "extensions.ask_user": ("Question", "Option", "Options", "AskUserPart"),
    "extensions.base": ("ModelRequestNext", "ToolCallNext"),
    "extensions.goal": ("GoalSummary", "RemainingWork", "NextInstruction"),
    "extensions.mcp": ("McpServer", "McpClientFactory"),
    "extensions.plan_mode": ("PlanReason", "PlanContent"),
    "extensions.shell_approval": ("ShellCommandNext",),
    "extensions.subagent": ("TaskDescription", "TaskPrompt", "SubAgentName"),
    "extensions.todo": ("TodoContent", "TodoList"),
    "json_types": ("JsonValue",),
    "messages": ("ImageSource", "UserContentPart", "UserContent", "AnyMessage"),
    "tools.base": ("ToolResult",),
    "tools.coding": ("FilePath", "StartLine", "LineCount", "TimeoutSeconds", "MaxResults"),
}
PARAMETERS = [(f"zett_agent.{module}", name) for module, names in ALIASES.items() for name in names]


@pytest.mark.parametrize(("module", "name"), PARAMETERS)
def test_aliases_are_named_instances_owned_by_their_defining_module(module, name):
    alias = getattr(importlib.import_module(module), name)
    assert isinstance(alias, TypeAliasType)
    assert alias.__name__ == name
    assert alias.__module__ == module
    assert alias.__type_params__ == ()


def test_source_uses_one_alias_declaration_form_without_annotated_factory_calls():
    """Guard the exact Pylance failure: ``Name: TypeAlias = TypeAliasType(...)``."""
    found = {}
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name != "TypeAlias" for alias in node.names), path
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
                assert value.value.id not in {"Annotated", "Callable", "Union", "Optional", "Literal"}, (
                    path,
                    node.lineno,
                )
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                continue
            if value.func.id != "TypeAliasType":
                continue
            assert isinstance(node, ast.Assign), (path, node.lineno)
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            name = node.targets[0].id
            assert isinstance(value.args[0], ast.Constant) and value.args[0].value == name
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            found.setdefault(module, set()).add(name)
    assert found == {module: set(names) for module, names in ALIASES.items()}


def test_callback_aliases_keep_callable_shapes_and_forward_references():
    assert get_origin(ModelRequestNext.__value__) is Callable
    arguments, result = get_args(ModelRequestNext.__value__)
    assert arguments == ["ModelRequest"]
    assert get_origin(result) is AsyncIterator
    assert get_args(result) == ("ModelEvent",)
    for alias in (ToolCallNext, ShellCommandNext):
        assert get_origin(alias.__value__) is Callable
        arguments, result = get_args(alias.__value__)
        assert arguments == []
        assert get_origin(result) is Awaitable and get_args(result) == (ToolResult,)
    assert ToolResult.__value__ is Any
    hints = get_type_hints(
        MiddlewareHook.on_model_request,
        localns={"AgentRunContext": object, "ModelRequest": ModelRequest, "ModelEvent": ModelEvent},
    )
    assert hints["call_next"] is ModelRequestNext
    assert hints["return"] == AsyncIterator[ModelEvent]


@pytest.mark.parametrize(
    "message",
    [
        SystemMessage(content="rules"),
        UserMessage(content="input"),
        AssistantMessage(content="answer"),
        ToolMessage(tool_call_id="call", name="lookup", content="result"),
        AgentMessage(content="continue"),
    ],
)
def test_message_union_alias_validates_and_serializes_all_roles(message):
    adapter = TypeAdapter(AnyMessage)
    assert adapter.validate_python(message) is message
    assert adapter.dump_python(message)["content"] == message.content
    assert set(get_args(AnyMessage.__value__)) == {
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolMessage,
        AgentMessage,
    }
    with pytest.raises(ValidationError):
        adapter.validate_python(ToolCall("call", "lookup"))


CONSTRAINTS = [
    ("tools.coding", "FilePath", "file.txt", "", "minLength", 1),
    ("tools.coding", "StartLine", 1, 0, "minimum", 1),
    ("tools.coding", "LineCount", 2_000, 2_001, "maximum", 2_000),
    ("tools.coding", "TimeoutSeconds", 300, 301, "maximum", 300),
    ("tools.coding", "MaxResults", 5_000, 5_001, "maximum", 5_000),
    ("extensions.ask_user", "Question", "question", "", "maxLength", 4_000),
    ("extensions.ask_user", "Option", "choice", "", "maxLength", 500),
    ("extensions.ask_user", "Options", ["choice"], [""] * 21, "maxItems", 20),
    ("extensions.goal", "GoalSummary", "summary", "", "maxLength", 8_000),
    ("extensions.goal", "RemainingWork", [], ["x"] * 101, "maxItems", 100),
    ("extensions.goal", "NextInstruction", None, "x" * 16_001, "anyOf", None),
    ("extensions.plan_mode", "PlanReason", "reason", "  ", "maxLength", 2_000),
    ("extensions.plan_mode", "PlanContent", "plan", "  ", "maxLength", 100_000),
    ("extensions.subagent", "TaskDescription", "description", "", "maxLength", 120),
    ("extensions.subagent", "TaskPrompt", "prompt", "", "maxLength", 100_000),
    ("extensions.subagent", "SubAgentName", "name", "", "maxLength", 64),
    ("extensions.todo", "TodoContent", "content", "  ", "maxLength", 4_000),
    (
        "extensions.todo",
        "TodoList",
        [{"content": "task", "status": "in_progress"}],
        [],
        "maxItems",
        100,
    ),
]


@pytest.mark.parametrize(("module", "name", "valid", "invalid", "constraint", "limit"), CONSTRAINTS)
def test_named_constraints_preserve_validation_and_schema(module, name, valid, invalid, constraint, limit):
    alias = getattr(importlib.import_module(f"zett_agent.{module}"), name)
    adapter = TypeAdapter(alias)
    assert adapter.dump_python(adapter.validate_python(valid), mode="json") == valid
    with pytest.raises(ValidationError):
        adapter.validate_python(invalid)
    schema = adapter.json_schema()
    definition = schema["$defs"][name] if "$ref" in schema else schema
    if name == "NextInstruction":
        assert definition["anyOf"] == [{"maxLength": 16_000, "type": "string"}, {"type": "null"}]
    else:
        assert definition[constraint] == limit


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("extensions.todo", "TodoContent"),
        ("extensions.plan_mode", "PlanReason"),
        ("extensions.plan_mode", "PlanContent"),
    ],
)
def test_named_string_constraints_still_strip_whitespace(module, name):
    alias = getattr(importlib.import_module(f"zett_agent.{module}"), name)
    assert TypeAdapter(alias).validate_python("  text \n") == "text"


def test_nested_aliases_keep_item_constraints_and_discriminated_union():
    from zett_agent.extensions.ask_user import AskUserImagePart, AskUserPart, AskUserTextPart, Options
    from zett_agent.extensions.todo import TodoList

    for invalid in ([""], ["x" * 501], ["x"] * 21):
        with pytest.raises(ValidationError):
            TypeAdapter(Options).validate_python(invalid)
    adapter = TypeAdapter(AskUserPart)
    assert isinstance(adapter.validate_python({"type": "text", "text": "answer"}), AskUserTextPart)
    assert isinstance(
        adapter.validate_python({"type": "image", "name": "a", "content_url": "data:image/png;base64,AA=="}),
        AskUserImagePart,
    )
    for invalid in ({"type": "unknown"}, {"text": "missing tag"}, {"type": "image", "name": "", "content_url": "x"}):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)
    schema = adapter.json_schema()
    assert schema["discriminator"]["propertyName"] == "type"
    assert len(schema["oneOf"]) == 2
    for invalid in ([{"content": " ", "status": "pending"}], [{"content": "x", "status": "processing"}]):
        with pytest.raises(ValidationError):
            TypeAdapter(TodoList).validate_python(invalid)


async def test_custom_tools_accept_named_aliases_in_function_annotations():
    @tool(guidelines="Use to validate a named tool parameter.")
    async def bounded(value: LineCount) -> int:
        """Return a value only after named constraints are checked."""
        return value

    assert await bounded({"value": 2_000}) == 2_000
    with pytest.raises(ValidationError):
        await bounded({"value": 2_001})
    assert bounded.parameters["properties"]["value"]["$ref"] == "#/$defs/LineCount"
    assert bounded.parameters["$defs"]["LineCount"]["maximum"] == 2_000
