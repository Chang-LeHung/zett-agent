"""Packages stay thin, and importing the runtime never imports a provider SDK."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import zett_agent

ROOT = Path(__file__).resolve().parents[1] / "src" / "zett_agent"

#: Packages that must not re-export the modules they contain.
THIN_PACKAGES = (
    "zett_agent",
    "zett_agent.tools",
    "zett_agent.providers",
    "zett_agent.extensions",
    "zett_agent.extensions.server_tools",
)

#: Provider SDKs and storage engines a bare import must not pull in.
DEFERRED_THIRD_PARTY = ("openai", "anthropic", "mcp", "sqlalchemy")

#: Names applications reach through their defining module, never through a package.
MODULE_LEVEL_IMPORTS = {
    "Agent": "zett_agent.agent",
    "AgentRunConfig": "zett_agent.agent",
    "tool": "zett_agent.tools.base",
    "AgentTool": "zett_agent.tools.base",
    "read_file": "zett_agent.tools.coding",
    "UserMessage": "zett_agent.messages",
    "ModelRequest": "zett_agent.model",
    "AgentExtension": "zett_agent.extensions.base",
    "SessionStorage": "zett_agent.extensions.persistence",
    "ExternalEventExtension": "zett_agent.extensions.external",
    "CodingExtension": "zett_agent.extensions.coding",
    "McpExtension": "zett_agent.extensions.mcp",
    "SQLiteSessionExtension": "zett_agent.extensions.sqlite",
    "OpenAIServerToolExtension": "zett_agent.extensions.server_tools.openai",
    "OpenAIProvider": "zett_agent.providers.openai",
    "AnthropicProvider": "zett_agent.providers.anthropic",
    "SQLiteSessionStorage": "zett_agent.storage",
}


def run_python(code: str) -> list[str]:
    """Run one snippet in a fresh interpreter and return its stdout fields."""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return result.stdout.split()


@pytest.mark.parametrize("package", THIN_PACKAGES)
def test_packages_re_export_nothing(package):
    path = ROOT.joinpath(*package.split(".")[1:], "__init__.py")
    tree = ast.parse(path.read_text())
    imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imported == [], f"{package} imports {len(imported)} module(s) at import time"
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assigned <= {"__version__"}, f"{package} assigns {sorted(assigned)}"


def test_packages_hold_no_public_attributes_in_a_fresh_interpreter():
    # Importing a submodule binds it on its package, so this must be measured
    # before anything else imports one.
    fields = run_python("import zett_agent; print(*[n for n in vars(zett_agent) if not n.startswith('_')])")
    assert fields == [], f"zett_agent exposes {fields}"
    subpackages = ["zett_agent.tools", "zett_agent.providers", "zett_agent.extensions"]
    fields = run_python(
        "import sys; "
        "import zett_agent.tools, zett_agent.providers, zett_agent.extensions; "
        f"print(*[n for p in {subpackages!r} for n in vars(sys.modules[p]) if not n.startswith('_')])"
    )
    assert fields == [], f"subpackages expose {fields}"
    fields = run_python(
        "import sys, zett_agent.extensions.server_tools; "
        "print(*[n for n in vars(sys.modules['zett_agent.extensions.server_tools']) if not n.startswith('_')])"
    )
    assert fields == [], f"the hosted-tool package exposes {fields}"


def test_public_names_live_in_the_module_that_defines_them():
    for name, module_name in MODULE_LEVEL_IMPORTS.items():
        assert getattr(importlib.import_module(module_name), name) is not None
        assert not hasattr(zett_agent, name), f"{name} should not be reachable as zett_agent.{name}"


def test_package_import_of_a_moved_name_fails():
    with pytest.raises(ImportError):
        exec("from zett_agent import Agent", {})


def test_bare_package_import_stays_free_of_provider_sdks():
    # A fresh interpreter proves the package init itself imports nothing heavy.
    fields = run_python(
        "import sys, zett_agent; "
        "print(zett_agent.__version__); "
        f"print(*[name for name in {DEFERRED_THIRD_PARTY!r} if name in sys.modules])"
    )
    assert fields[0] == zett_agent.__version__
    assert fields[1:] == [], f"import zett_agent pulled in {fields[1:]}"


def test_runtime_modules_and_adapters_defer_their_sdks():
    fields = run_python(
        "import sys; "
        "import zett_agent.agent, zett_agent.events, zett_agent.messages, zett_agent.model, zett_agent.tools.base; "
        "import zett_agent.providers.base, zett_agent.providers.responses, zett_agent.providers.anthropic, "
        "zett_agent.providers.openai, zett_agent.extensions.mcp, zett_agent.extensions.coding, "
        "zett_agent.extensions.subagent, zett_agent.extensions.goal; "
        f"print(*[name for name in {DEFERRED_THIRD_PARTY!r} if name in sys.modules])"
    )
    assert fields == [], f"importing runtime or adapter modules pulled in {fields}"


def test_builtin_subagents_import_their_sqlite_persistence_lazily():
    """Importing the module stays cheap; only building a profile pays for SQLAlchemy."""
    fields = run_python(
        "import sys; "
        "import zett_agent.extensions.subagent as subagent; "
        "imported = 'sqlalchemy' in sys.modules; "
        "subagent.default_subagents(object()); "
        "print(imported, 'sqlalchemy' in sys.modules)"
    )
    assert fields == ["False", "True"]


def test_constructing_a_provider_imports_its_sdk():
    fields = run_python(
        "import sys; "
        "from zett_agent.providers.openai import OpenAIProvider; "
        "provider = OpenAIProvider(model='m', api_key='k', base_url='http://localhost/v1'); "
        "print('openai' in sys.modules)"
    )
    assert fields == ["True"]
