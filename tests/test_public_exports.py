"""Packages stay thin, and importing the runtime never imports a provider SDK."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import zett_agent

ROOT = Path(__file__).resolve().parents[1] / "src" / "zett_agent"

#: Packages that must not re-export anything: the root holds the version and nothing else.
THIN_PACKAGES = ("zett_agent",)

#: Facades that re-export lazily (PEP 562). Importing one imports nothing; a name loads its defining
#: module on first use. ``jit_harness`` is the eager facade, tied to its reference page by
#: ``test_jit_harness_facade_is_the_documented_contract``.
LAZY_FACADES = (
    "zett_agent.tools",
    "zett_agent.providers",
    "zett_agent.extensions",
    "zett_agent.extensions.server_tools",
    "zett_agent.jit_harness",
    "zett_agent.sandbox",
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


#: The reference table is the published contract; the manual and the docs check
#: both read it, so the facade cannot drift from what is documented.
REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "_ext" / "reference.py"


def test_jit_harness_facade_is_the_documented_contract():
    """jit_harness re-exports on purpose, so its facade mirrors its reference page."""
    import zett_agent.jit_harness as jit_harness

    spec = importlib.util.spec_from_file_location("reference_contract", REFERENCE)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    documented = {
        name
        for module_name, names in reference.PUBLIC_API.items()
        if module_name.startswith("zett_agent.jit_harness.")
        for name in names
    }
    exported = list(jit_harness.__all__)
    assert len(set(exported)) == len(exported), exported
    assert set(exported) == documented, sorted(set(exported) ^ documented)
    for name in documented:
        exported_object = getattr(jit_harness, name)
        module_name = getattr(exported_object, "__module__", None)
        if module_name is not None:
            assert module_name.startswith("zett_agent.jit_harness."), (name, module_name)


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


def _reference():
    spec = importlib.util.spec_from_file_location("reference_contract", REFERENCE)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


def _facade_tree(package):
    return ast.parse(ROOT.joinpath(*package.split(".")[1:], "__init__.py").read_text())


@pytest.mark.parametrize("package", LAZY_FACADES)
def test_lazy_facades_import_only_under_type_checking(package):
    """The only eager imports are the lazy helper; every re-export sits under ``if TYPE_CHECKING``."""
    tree = _facade_tree(package)
    eager = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert {node.module for node in eager} <= {"typing", "_lazy"}, [node.module for node in eager]
    guarded = [node for node in tree.body if isinstance(node, ast.If)]
    assert len(guarded) == 1, "exactly one `if TYPE_CHECKING:` block mirrors the exports"
    typed = {
        alias.asname
        for node in guarded[0].body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == alias.name
    }
    module = importlib.import_module(package)
    assert typed == set(module.__all__), sorted(typed ^ set(module.__all__))
    assert len(module.__all__) == len(set(module.__all__))


@pytest.mark.parametrize("package", LAZY_FACADES)
def test_lazy_facades_resolve_each_name_from_its_documented_module(package):
    """A facade name is the object its reference page documents, defined in the module listed there."""
    documented = {name: module_name for module_name, names in _reference().PUBLIC_API.items() for name in names}
    module = importlib.import_module(package)
    for name in module.__all__:
        value = getattr(module, name)
        defining = documented[name]
        assert defining.rpartition(".")[0] == package, (name, defining)
        assert value is getattr(importlib.import_module(defining), name), name
        assert vars(module)[name] is value, "the first access caches the name on the package"
    assert set(module.__all__) <= set(dir(module))
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchName'"):
        module.NoSuchName  # noqa: B018


@pytest.mark.parametrize("package", LAZY_FACADES)
def test_lazy_facades_load_nothing_until_a_name_is_used(package):
    parts = package.split(".")
    parents = {".".join(parts[: i + 1]) for i in range(len(parts))}
    fields = run_python(
        "import sys; "
        f"import {package} as facade; "
        f"print(*sorted(name for name in sys.modules if name.startswith('zett_agent') and name not in {sorted(parents)!r})); "
        "print('--'); "
        "print(*[n for n in facade.__all__ if n in vars(facade)])"
    )
    separator = fields.index("--")
    assert fields[:separator] == ["zett_agent._lazy"], "the package loads its helper and nothing else"
    assert fields[separator + 1 :] == [], "no name is bound before it is used"


def test_lazy_facades_keep_the_sdks_and_pydantic_out_until_asked():
    names = ", ".join(f"{p!r}" for p in LAZY_FACADES)
    fields = run_python(
        "import importlib, sys; "
        f"[importlib.import_module(p) for p in ({names},)]; "
        f"print(*[name for name in {DEFERRED_THIRD_PARTY + ('pydantic', 'httpx')!r} if name in sys.modules])"
    )
    assert fields == [], f"importing the facades pulled in {fields}"
    fields = run_python(
        "import sys; "
        "from zett_agent.providers import OpenAIProvider, AnthropicProvider, GoogleProvider; "
        f"print(*[name for name in {DEFERRED_THIRD_PARTY!r} if name in sys.modules])"
    )
    assert fields == [], f"naming the adapters pulled in {fields}"


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
        "zett_agent.extensions.subagent, zett_agent.extensions.goal, zett_agent.extensions.tool_search; "
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


#: Modules a leaf runtime module must never pull in just by being imported.
EAGER_LEAF_IMPORTS = ("asyncio", "httpx", "tiktoken", "sqlalchemy", "truststore", "uuid6")


def test_leaf_runtime_modules_avoid_heavy_eager_imports():
    fields = run_python(
        "import sys; "
        "import zett_agent.messages, zett_agent.model, zett_agent.json_types, zett_agent.ids; "
        f"print(*[name for name in {EAGER_LEAF_IMPORTS!r} if name in sys.modules])"
    )
    assert fields == [], f"leaf runtime modules pulled in {fields}"


def test_extension_modules_defer_clients_and_tokenizers():
    fields = run_python(
        "import sys; "
        "import zett_agent.extensions.coding, zett_agent.extensions.goal, "
        "zett_agent.extensions.compaction, zett_agent.extensions.jsonl; "
        "print(*[name for name in ('httpx', 'tiktoken', 'truststore', 'sqlalchemy') if name in sys.modules])"
    )
    assert fields == [], f"extension modules pulled in {fields}"


def test_uuid_helper_resolves_lazily_and_prefers_the_standard_library():
    fields = run_python(
        "import sys, zett_agent.ids; "
        "print('uuid6' in sys.modules); "
        "zett_agent.ids.new_uuid7(); "
        "print('uuid6' in sys.modules)"
    )
    # uuid.uuid7 is stdlib from 3.14; older interpreters fall back to uuid6 only
    # when an identifier is actually requested.
    assert fields == ["False", "False" if sys.version_info >= (3, 14) else "True"]


def test_provider_modules_defer_http_clients_and_json_repair():
    fields = run_python(
        "import sys; "
        "import zett_agent.providers.base, zett_agent.providers.openai, zett_agent.providers.anthropic, "
        "zett_agent.providers.google, zett_agent.providers.ollama, zett_agent.providers.deepseek; "
        "print(*[name for name in ('httpx', 'truststore', 'json_repair', 'pydantic') if name in sys.modules])"
    )
    assert fields == [], f"provider modules pulled in {fields}"


def test_sqlite_storage_imports_sqlalchemy_only_when_asked_for():
    """The message codec is cheap; the ORM layer loads on first attribute access."""
    fields = run_python(
        "import sys; "
        "import zett_agent.storage; "
        "from zett_agent.storage import encode_messages, MessageKind; "
        "print('sqlalchemy' in sys.modules)"
    )
    assert fields == ["False"]
    fields = run_python(
        "import sys; "
        "from zett_agent.storage import SQLiteSessionStorage; "
        "print('sqlalchemy' in sys.modules, SQLiteSessionStorage.__name__)"
    )
    assert fields == ["True", "SQLiteSessionStorage"]
    fields = run_python("import zett_agent.storage as storage; print(hasattr(storage, 'AgentSessionModel'))")
    assert fields == ["False"]


def test_pydantic_loads_only_when_validation_is_needed():
    """Core modules defer pydantic; the first decode or tool call pays for it."""
    fields = run_python(
        "import sys; "
        "import zett_agent.agent, zett_agent.tools.base, zett_agent.storage, "
        "zett_agent.providers.responses; "
        "print('pydantic' in sys.modules)"
    )
    assert fields == ["False"]
    fields = run_python("import sys; import zett_agent.tools.coding; print('pydantic' in sys.modules)")
    assert fields == ["True"], "declaring tools must build their schema"
    fields = run_python(
        "import sys; "
        "from zett_agent.messages import UserMessage; "
        "from zett_agent.storage import decode_messages, encode_messages; "
        "decode_messages(encode_messages([UserMessage(content='x')])); "
        "print('pydantic' in sys.modules)"
    )
    assert fields == ["True"]
