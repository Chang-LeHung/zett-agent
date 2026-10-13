"""Load trusted Python packages using the runtime's existing tool contract."""

from __future__ import annotations

import sys
from copy import deepcopy
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from threading import RLock
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from ..ids import new_uuid7
from ..tools.base import AgentTool, ToolExecutionMode, tool
from .snapshot import _ENTRYPOINT_FILE, PythonToolSource, _validate_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .runtime import JITHarness

#: Loading installs a process-global import hook, so one thread loads at a time.
_LOAD_LOCK = RLock()


class ToolPackageLoader(Loader):
    """Internal: serve one tool package's modules from validated in-memory source."""

    def __init__(self, root: str, entrypoint: str, files: Mapping[str, str], harness: JITHarness) -> None:
        self.root = root
        self.entrypoint = entrypoint
        self.files = files
        self.harness = harness

    def resolve(self, fullname: str) -> tuple[str, str] | None:
        """Return the virtual path and source of one module, or None when unknown."""
        if fullname == self.root:
            return _ENTRYPOINT_FILE, self.entrypoint
        relative = fullname[len(self.root) + 1 :].replace(".", "/")
        for candidate in (f"{relative}.py", f"{relative}/{_ENTRYPOINT_FILE}"):
            source = self.files.get(candidate)
            if source is not None:
                return candidate, source
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        """Execute one module body, exactly as a normal import would."""
        resolved = self.resolve(module.__name__)
        if resolved is None:  # pragma: no cover - find_spec resolved this name already
            raise ModuleNotFoundError(f"No JIT source for {module.__name__}")
        path, source = resolved
        # The decorator, the scheduling mode, and the harness handle are the
        # package's whole interface; every module of it receives the same ones.
        module.__dict__.update(tool=tool, ToolExecutionMode=ToolExecutionMode, harness=self.harness)
        exec(compile(source, f"<jit:{self.root}/{path}>", "exec"), module.__dict__)


class ToolPackageFinder(MetaPathFinder):
    """Internal: resolve only one tool package's modules, and only while it loads."""

    def __init__(self, loader: ToolPackageLoader) -> None:
        self.loader = loader

    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> ModuleSpec | None:
        if fullname != self.loader.root and not fullname.startswith(self.loader.root + "."):
            return None
        resolved = self.loader.resolve(fullname)
        if resolved is None:
            return None
        relative, _ = resolved
        origin = f"<jit:{self.loader.root}/{relative}>"
        is_package = relative == _ENTRYPOINT_FILE or relative.endswith(f"/{_ENTRYPOINT_FILE}")
        return spec_from_loader(fullname, self.loader, is_package=is_package, origin=origin)


def _install(root: str, loader: ToolPackageLoader) -> ToolPackageFinder:
    """Install the load-only import hook for one package."""
    finder = ToolPackageFinder(loader)
    sys.meta_path.insert(0, finder)
    return finder


def _uninstall(root: str, finder: ToolPackageFinder) -> None:
    """Drop the hook and every module it imported, leaving no session residue."""
    for name in [name for name in sys.modules if name == root or name.startswith(root + ".")]:
        del sys.modules[name]
    try:
        sys.meta_path.remove(finder)
    except ValueError:  # pragma: no cover - only reachable after external mutation
        pass


def _load_python_tool(
    source: str,
    entrypoint: str,
    harness: JITHarness,
    files: Mapping[str, str] | None = None,
) -> tuple[AgentTool, PythonToolSource]:
    """Execute a fresh trusted package and validate its named AgentTool.

    The entry point module is the package root, so its siblings are reached with
    relative imports such as ``from .helpers import value``. Module bodies and
    imports run here, at registration time; function bodies are not executed for
    validation. The package is not retained in ``sys.modules`` afterwards, so a
    tool body cannot lazily import a sibling later: import what it needs at
    module level. Functions keep their globals, including imported modules.
    """
    if not entrypoint.isidentifier():
        raise ValueError("entrypoint must be a Python identifier")
    package = f"zett_jit_{new_uuid7().replace('-', '_')}"
    package_files = dict(files or {})
    loader = ToolPackageLoader(package, entrypoint=source, files=package_files, harness=harness)
    with _LOAD_LOCK:
        finder = _install(package, loader)
        try:
            module = import_module(package)
            registered = module.__dict__.get(entrypoint)
            if not isinstance(registered, AgentTool):
                raise ValueError(f"Entrypoint {entrypoint!r} must be decorated with @tool")
            registered = cast(AgentTool, registered)
            _validate_name(registered.name)
            if registered.local_tool_search or registered.deferred:
                raise ValueError("JIT tools must be ordinary, non-deferred function tools")
            record = PythonToolSource(
                name=registered.name,
                source=source,
                entrypoint=entrypoint,
                description=registered.description,
                parameters=deepcopy(registered.parameters),
                guidelines=registered.guidelines,
                snippet=registered.snippet,
                execution_mode=registered.execution_mode,
                files=package_files,
            )
            return registered, record
        finally:
            _uninstall(package, finder)
