"""Generate the public reference from the package's explicit export contract."""

import inspect
from pathlib import Path

from pydantic import BaseModel

import zett_agent

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
    if not inspect.isclass(obj) and not inspect.isfunction(obj) and not isinstance(obj, zett_agent.AgentTool):
        return "constants"
    if module.endswith(("persistence", "session_persistence", "sqlite", "storage")):
        return "storage"
    if module.endswith(("events", "dispatcher")):
        return "events"
    for group in ("providers", "tools", "extensions", "messages", "model", "client", "agent"):
        if f".{group}" in module:
            return group
    return "agent"


def reference_page(name):
    """Render one canonical object, excluding framework-generated implementation APIs."""
    obj = getattr(zett_agent, name)
    lines = [name, "=" * len(name), ""]
    guide, introduction = GUIDES[public_group(name, obj)]
    lines += [f"{introduction} See :doc:`the guide <{guide}>`.", ""]
    if isinstance(obj, zett_agent.AgentTool):
        # Decorators produce callable instances, not Python functions. Their
        # description and JSON Schema remain the authoritative public contract.
        import json

        lines += [obj.description, "", "Parameters", "----------", "", ".. code-block:: json", ""]
        lines += [f"   {line}" for line in json.dumps(obj.parameters, indent=2).splitlines()]
        lines += ["", "Model-facing snippet", "--------------------", "", ".. code-block:: text", ""]
        lines += [f"   {line}" for line in obj.snippet.splitlines()]
        lines += [""]
        lines += ["Guidelines", "----------", "", *(f"* {item}" for item in obj.guidelines), ""]
        lines += [f".. autodata:: zett_agent.{name}", "   :no-value:", ""]
        lines += [
            "Direct Python calls use ``await tool(arguments_mapping)``. The snippet",
            "above describes model-facing invocation syntax, not the Python call signature.",
            "",
        ]
    elif inspect.isclass(obj):
        lines += [f".. autoclass:: zett_agent.{name}"]
        members = set(getattr(obj, "__annotations__", {}))
        members.update(
            key for key in vars(obj) if not key.startswith("_") and key not in {"model_config", "model_fields"}
        )
        if name.endswith("Provider"):
            members.update(("stream", "aclose"))
        is_extension = issubclass(obj, zett_agent.AgentExtension)
        if is_extension:
            for base in obj.__mro__:
                if base.__module__.startswith("zett_agent"):
                    members.update(key for key in vars(base) if not key.startswith("_"))
        members = sorted(key for key in members if not key.startswith("_"))
        if members:
            lines += ["   :members: " + ", ".join(members)]
            if name.endswith("Provider") or is_extension:
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
        lines += [f".. autofunction:: zett_agent.{name}", ""]
    else:
        lines += [f".. autodata:: zett_agent.{name}", ""]
    return "\n".join(lines)


def write_changed(path, content):
    """Avoid rebuilding unchanged generated sources."""
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def generate_reference(app):
    """Write one page per public export before Sphinx discovers source files."""
    output = Path(app.srcdir) / "_generated"
    output.mkdir(exist_ok=True)
    expected = {f"{name}.rst".casefold() for name in zett_agent.__all__}
    expected.update(f"group-{key}.rst" for key in GROUPS)
    # Only generated RST files in this dedicated directory are disposable.
    for path in output.glob("*.rst"):
        if path.name.casefold() not in expected:
            path.unlink()
    groups = {key: [] for key in GROUPS}
    for name in zett_agent.__all__:
        obj = getattr(zett_agent, name)
        groups[public_group(name, obj)].append(name)
        write_changed(output / f"{name}.rst", reference_page(name))
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
            obj = getattr(zett_agent, name)
            description = obj.description if isinstance(obj, zett_agent.AgentTool) else inspect.getdoc(obj)
            summary = description.splitlines()[0] if description else "Public constant or type alias."
            if not inspect.isclass(obj) and not inspect.isfunction(obj) and not isinstance(obj, zett_agent.AgentTool):
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
