"""Generate the public reference from the supported public API."""

import importlib
import inspect
from pathlib import Path

from pydantic import BaseModel

from zett_agent.extensions.base import AgentExtension
from zett_agent.sync_runtime import SyncMethodsMixin
from zett_agent.tools.base import AgentTool

#: The supported public API, grouped by the module that defines each name.
#: Packages re-export nothing, so this table is the contract the reference
#: documents: add a name here when a module-level object becomes public.
PUBLIC_API: dict[str, tuple[str, ...]] = {
    "zett_agent.agent": ("Agent", "AgentRunConfig", "AgentRunContext", "AgentState"),
    "zett_agent.client": ("AgentClient", "create_agent"),
    "zett_agent.dispatcher": ("AgentEventDispatcher",),
    "zett_agent.event_queue": ("AgentEventQueue",),
    "zett_agent.events": (
        "AgentEvent",
        "AgentEventType",
        "AgentPhase",
        "AgentPhaseTransitionMixin",
        "ModelOutputTracker",
    ),
    "zett_agent.exceptions": ("AgentError", "AgentIterationLimitError", "AgentProtocolError"),
    "zett_agent.extensions.agents_md": ("AGENTS_FILE_NAME", "AgentsMdExtension"),
    "zett_agent.extensions.ask_user": (
        "ASK_USER_EVENT_NAME",
        "ASK_USER_RESPONSE_EVENT_NAME",
        "ASK_USER_TOOL_NAME",
        "AskUserEvent",
        "AskUserExtension",
        "AskUserRequest",
        "AskUserResult",
    ),
    "zett_agent.extensions.base": (
        "AgentEventHooksMixin",
        "AgentExtension",
        "AgentModelHooksMixin",
        "AgentRunHooksMixin",
        "AgentSetupHooksMixin",
        "AgentToolHooksMixin",
        "AgentTurnHooksMixin",
        "MiddlewareHook",
        "ModelRequestNext",
        "ToolCallNext",
    ),
    "zett_agent.extensions.coding": ("CodingExtension",),
    "zett_agent.extensions.compaction": ("CompactionExtension",),
    "zett_agent.extensions.events": (
        "CompactionEvent",
        "ContentCompletedEvent",
        "ContentStartedEvent",
        "ExtensionEvent",
        "InternalMessageEvent",
        "MessageAppendedEvent",
        "MessageTiming",
        "PhaseTransitionEvent",
        "ReasoningCompletedEvent",
        "ReasoningStartedEvent",
        "RunCancelledEvent",
        "SteeringMessageEvent",
    ),
    "zett_agent.extensions.external": ("ExternalEvent", "ExternalEventExtension"),
    "zett_agent.extensions.file_system": ("FileSystemExtension",),
    "zett_agent.extensions.goal": (
        "GOAL_EVALUATION_TOOL_NAME",
        "GOAL_MODE_EVENT_NAME",
        "GoalEvaluation",
        "GoalExtension",
        "default_goal_subagent",
    ),
    "zett_agent.extensions.internal_message": ("INTERNAL_MESSAGE_EVENT_NAME", "InternalMessageExtension"),
    "zett_agent.extensions.jsonl": ("JSONLExtension",),
    "zett_agent.extensions.mcp": (
        "DEFAULT_MCP_CONFIG_PATH",
        "DEFAULT_MCP_SERVER_KEYS",
        "McpClient",
        "McpClientFactory",
        "McpConfiguration",
        "McpExtension",
        "McpHttpServer",
        "McpServer",
        "McpStdioServer",
    ),
    "zett_agent.extensions.memory": ("InMemoryMessageAccumulator",),
    "zett_agent.extensions.model_request_trace": (
        "MODEL_REQUEST_TRACE_ATTRIBUTE",
        "MODEL_REQUEST_TRACE_SCHEMA_VERSION",
        "ModelRequestTrace",
        "ModelRequestTraceExtension",
        "ServerToolDefinitionTrace",
        "ToolDefinitionTrace",
        "model_request_trace",
    ),
    "zett_agent.extensions.persistence": (
        "BaseSessionPersistenceExtension",
        "ContextSnapshot",
        "RawMessageRecord",
        "SessionStorage",
        "SessionSummary",
        "SessionView",
    ),
    "zett_agent.extensions.plan_mode": (
        "ENTER_PLAN_MODE_EVENT_NAME",
        "ENTER_PLAN_MODE_RESPONSE_EVENT_NAME",
        "ENTER_PLAN_MODE_TOOL_NAME",
        "EXIT_PLAN_MODE_EVENT_NAME",
        "EXIT_PLAN_MODE_RESPONSE_EVENT_NAME",
        "EXIT_PLAN_MODE_TOOL_NAME",
        "EnterPlanModeEvent",
        "EnterPlanModeRequest",
        "EnterPlanModeResponse",
        "EnterPlanModeResult",
        "ExitPlanModeEvent",
        "ExitPlanModeRequest",
        "ExitPlanModeResponse",
        "ExitPlanModeResult",
        "PLAN_MODE_ENTERED_EVENT_NAME",
        "PLAN_MODE_EXITED_EVENT_NAME",
        "PLAN_MODE_SYSTEM_PROMPT",
        "PlanModeEnteredEvent",
        "PlanModeExitedEvent",
        "PlanModeExtension",
    ),
    "zett_agent.extensions.server_tools.anthropic": ("AnthropicServerToolExtension",),
    "zett_agent.extensions.server_tools.base": ("ServerToolExtension",),
    "zett_agent.extensions.server_tools.deepseek": ("DeepSeekServerToolExtension",),
    "zett_agent.extensions.server_tools.google": ("GoogleServerToolExtension",),
    "zett_agent.extensions.server_tools.openai": ("OpenAIServerToolExtension",),
    "zett_agent.extensions.server_tools.openrouter": ("OpenRouterServerToolExtension",),
    "zett_agent.extensions.session_persistence": ("SessionPersistenceExtension",),
    "zett_agent.extensions.shell_approval": (
        "SHELL_APPROVAL_EVENT_NAME",
        "SHELL_APPROVAL_RESPONSE_EVENT_NAME",
        "SHELL_APPROVAL_TOOL_NAME",
        "ShellApprovalExtension",
        "ShellApprovalMode",
        "ShellApprovalRequestedEvent",
        "ShellApprovalResponse",
        "ShellApprovalStorage",
        "ShellCommandAborted",
    ),
    "zett_agent.extensions.skill": (
        "DEFAULT_SKILL_ROOTS",
        "READ_SKILL_TOOL_NAME",
        "SKILL_FILE_NAME",
        "SkillDefinition",
        "SkillExtension",
        "SkillFileParser",
    ),
    "zett_agent.extensions.sqlite": ("SQLiteSessionExtension",),
    "zett_agent.extensions.steering": ("STEERING_MESSAGE_EVENT_NAME", "SteeringExtension"),
    "zett_agent.extensions.subagent": (
        "SubAgentDefinition",
        "SubAgentExtension",
        "SubAgentResult",
        "TASK_TOOL_NAME",
        "default_subagents",
    ),
    "zett_agent.extensions.todo": (
        "TODO_WRITE_TOOL_NAME",
        "TodoItem",
        "TodoStatus",
        "TodoWriteExtension",
        "TodoWriteResult",
    ),
    "zett_agent.extensions.tool_guidelines": ("ToolGuidelinesExtension",),
    "zett_agent.extensions.tool_search": (
        "BM25Search",
        "TOOL_SEARCH_TOOL_NAME",
        "ToolSearchExtension",
        "ToolSearchStorage",
    ),
    "zett_agent.extensions.usage_activity": (
        "ModelUsageActivityDay",
        "ModelUsageActivityRecord",
        "ModelUsageActivityStorage",
        "UsageActivityExtension",
    ),
    "zett_agent.ids": ("new_uuid7",),
    "zett_agent.json_types": ("JsonValue",),
    "zett_agent.messages": (
        "AgentMessage",
        "AnyMessage",
        "AssistantMessage",
        "ImageBytesSource",
        "ImageContent",
        "ImageDetail",
        "ImageSource",
        "ImageUrlSource",
        "Message",
        "MessageRole",
        "SystemMessage",
        "TextContent",
        "ToolCall",
        "ToolMessage",
        "UserContent",
        "UserContentPart",
        "UserMessage",
    ),
    "zett_agent.model": (
        "AgentModel",
        "ModelEvent",
        "ModelEventType",
        "ModelRequest",
        "ModelResponse",
        "ModelUsage",
        "ReasoningEffort",
        "RetryOptions",
        "ServerToolCall",
        "ServerToolDefinition",
        "ServerToolInputDelta",
        "ServerToolResult",
        "ToolCallDelta",
        "ToolDefinition",
    ),
    "zett_agent.providers.anthropic": ("AnthropicProvider", "_to_anthropic_content_blocks"),
    "zett_agent.providers.base": (
        "ProviderAuthError",
        "ProviderError",
        "ProviderResponseError",
        "_message_to_openai_payload",
        "_normalize_image_source",
        "_usage_from_mapping",
    ),
    "zett_agent.providers.deepseek": ("DeepSeekProvider",),
    "zett_agent.providers.google": ("GoogleProvider",),
    "zett_agent.providers.ollama": ("OllamaProvider",),
    "zett_agent.providers.openai": ("OpenAIProvider",),
    "zett_agent.storage": ("SQLiteSessionStorage",),
    "zett_agent.sync": ("SyncAgent", "SyncModelAdapter", "create_agent_sync"),
    "zett_agent.sync_runtime": ("SyncContext", "SyncObject", "SyncRuntime", "SyncStream"),
    "zett_agent.tools.base": (
        "AgentTool",
        "ToolExecutionMode",
        "ToolResult",
        "get_tool_guidelines",
        "get_tool_snippet",
        "render_tool_guidance",
        "render_tool_search_text",
        "tool",
    ),
    "zett_agent.tools.coding": (
        "delete_file",
        "glob",
        "grep",
        "read_file",
        "replace_in_file",
        "run_shell",
        "write_file",
    ),
    "zett_agent.tools.images": ("view_image",),
}

GROUPS = {
    "client": "Application client",
    "agent": "Runtime and configuration",
    "messages": "Messages and images",
    "events": "Events and dispatch",
    "model": "Model protocol and usage",
    "providers": "Provider adapters",
    "tools": "Tools and filesystem operations",
    "storage": "Session storage",
    "extensions": "Extensions and lifecycle hooks",
    "constants": "Constants and type aliases",
}

GUIDES = {
    "client": ("../learn/first-agent", "Start here to create, run, and stream an application client."),
    "agent": ("../concepts/lifecycle", "Understand request ownership, phases, and runtime limits."),
    "messages": ("../concepts/context", "Choose message roles and distinguish model input from stored history."),
    "events": ("../concepts/events", "Distinguish UI streams, extension notifications, and external replies."),
    "model": ("../extending/model-adapter", "Implement the model contract and normalize usage and stream output."),
    "providers": ("../learn/providers", "Connect a provider and configure retries, credentials, and transport."),
    "tools": ("../learn/tools", "Build typed tools from annotations and docstrings."),
    "storage": ("../extending/storage-adapter", "Restore model context and preserve immutable original messages."),
    "extensions": ("../extending/first-extension", "Build extensions, choose hooks, and manage request-scoped state."),
    "constants": ("../concepts/events", "Shared event names and public type aliases."),
}


def public_group(name, obj):
    """Assign every exported symbol to one stable navigation category."""
    module = getattr(obj, "__module__", "")
    if not inspect.isclass(obj) and not inspect.isfunction(obj) and not isinstance(obj, AgentTool):
        return "constants"
    if module.endswith(("persistence", "session_persistence", "sqlite", "storage")):
        return "storage"
    if module.endswith(("events", "dispatcher")):
        return "events"
    for group in ("providers", "tools", "extensions", "messages", "model", "client", "agent"):
        if f".{group}" in module:
            return group
    return "agent"


def reference_exports():
    """Map each documented name to the import path that defines it."""
    return {name: f"{module}.{name}" for module, names in PUBLIC_API.items() for name in names}


def reference_object(path):
    """Return the object a generated page documents."""
    return getattr(importlib.import_module(path.rpartition(".")[0]), path.rpartition(".")[2])


def reference_page(name, path):
    """Render one canonical object, excluding framework-generated implementation APIs."""
    obj = reference_object(path)
    lines = [name, "=" * len(name), ""]
    guide, introduction = GUIDES[public_group(name, obj)]
    lines += [f"{introduction} See :doc:`the guide <{guide}>`.", ""]
    if isinstance(obj, AgentTool):
        # Decorators produce callable instances, not Python functions. Their
        # description and JSON Schema remain the authoritative public contract.
        import json

        lines += [obj.description, "", "Parameters", "----------", "", ".. code-block:: json", ""]
        lines += [f"   {line}" for line in json.dumps(obj.parameters, indent=2).splitlines()]
        lines += ["", "Model-facing snippet", "--------------------", "", ".. code-block:: text", ""]
        lines += [f"   {line}" for line in obj.snippet.splitlines()]
        lines += [""]
        lines += ["Guidelines", "----------", "", *(f"* {item}" for item in obj.guidelines), ""]
        lines += [f".. autodata:: {path}", "   :no-value:", ""]
        lines += [
            "Direct Python calls use ``await tool(arguments_mapping)``. The snippet",
            "above describes model-facing invocation syntax, not the Python call signature.",
            "",
        ]
    elif inspect.isclass(obj):
        lines += [f".. autoclass:: {path}"]
        members = set(getattr(obj, "__annotations__", {}))
        members.update(
            key for key in vars(obj) if not key.startswith("_") and key not in {"model_config", "model_fields"}
        )
        if name.endswith("Provider"):
            members.update(("stream", "aclose"))
        is_extension = issubclass(obj, AgentExtension)
        has_sync_view = issubclass(obj, SyncMethodsMixin)
        if has_sync_view:
            members.add("sync")
        if is_extension:
            for base in obj.__mro__:
                if base.__module__.startswith("zett_agent"):
                    members.update(key for key in vars(base) if not key.startswith("_"))
        members = sorted(key for key in members if not key.startswith("_"))
        if members:
            lines += ["   :members: " + ", ".join(members)]
            if name.endswith("Provider") or is_extension or has_sync_view:
                lines += ["   :inherited-members:"]
        if name == "ExternalEventExtension":
            lines += ["   :private-members: _wait_for_external_event, _take_external_event"]
        if name == "AgentTool":
            lines += ["   :special-members: __call__"]
        lines += [""]
        if issubclass(obj, BaseModel):
            lines += ["Fields", "------", ""]
            for field_name, info in obj.model_fields.items():
                status = "required" if info.is_required() else "optional"
                lines += [
                    f"``{field_name}`` ({status})",
                    f"    {info.description or 'See the typed model signature for this field.'}",
                    "",
                ]
    elif inspect.isfunction(obj):
        lines += [f".. autofunction:: {path}", ""]
    else:
        lines += [f".. autodata:: {path}", ""]
    return "\n".join(lines)


def write_changed(path, content):
    """Avoid rebuilding unchanged generated sources."""
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def generate_reference(app):
    """Write one page per public export before Sphinx discovers source files."""
    output = Path(app.srcdir) / "_generated"
    output.mkdir(exist_ok=True)
    exports = reference_exports()
    expected = {f"{name}.rst".casefold() for name in exports}
    expected.update(f"group-{key}.rst" for key in GROUPS)
    # Only generated RST files in this dedicated directory are disposable.
    for path in output.glob("*.rst"):
        if path.name.casefold() not in expected:
            path.unlink()
    groups = {key: [] for key in GROUPS}
    for name, path in exports.items():
        obj = reference_object(path)
        groups[public_group(name, obj)].append(name)
        write_changed(output / f"{name}.rst", reference_page(name, path))
    for key, names in groups.items():
        title = GROUPS[key]
        lines = [
            title,
            "=" * len(title),
            "",
            GUIDES[key][1],
            "",
            f"Read :doc:`the walkthrough <{GUIDES[key][0]}>` before consulting individual signatures.",
            "",
            ".. list-table:: Public interfaces",
            "   :header-rows: 1",
            "   :widths: 35 65",
            "",
            "   * - Interface",
            "     - Purpose",
        ]
        for name in names:
            obj = reference_object(exports[name])
            description = obj.description if isinstance(obj, AgentTool) else inspect.getdoc(obj)
            summary = description.splitlines()[0] if description else "Public constant or type alias."
            if not inspect.isclass(obj) and not inspect.isfunction(obj) and not isinstance(obj, AgentTool):
                summary = "Public constant or type alias; see its definition."
            lines += [f"   * - :doc:`{name}`", f"     - {summary}"]
        lines += [
            "",
            ".. toctree::",
            "   :hidden:",
            "   :maxdepth: 1",
            "",
            *[f"   {name}" for name in names],
            "",
        ]
        write_changed(output / f"group-{key}.rst", "\n".join(lines))


def setup(app):
    """Register generation without touching runtime source or model resources."""
    app.connect("builder-inited", generate_reference)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
