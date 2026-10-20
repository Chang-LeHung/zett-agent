"""Session-owned Python tool environments with a shared invocation boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextvars import ContextVar, copy_context
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Literal

from ..agent import AgentRunContext
from ..ids import new_uuid7
from ..messages import ImageContent, TextContent, ToolCall, ToolMessage
from ..tools.base import (
    _SYNC_TOOL_RUNNER,
    AgentTool,
    ToolResult,
    _bind_tool_call,
    _reset_tool_call,
    current_tool_call_id,
)
from ..tools.invocation import invoke_tool
from .enums import ToolCallStatus, ToolOrigin
from .errors import ToolErrorReport
from .profiling import MAX_RETAINED_CALLS, ActiveToolCall, JITHarnessProfile, ToolProfiler, ToolProfileStats
from .repo import (
    INIT_FILE,
    MANAGEMENT_NAMES,
    SKELETON,
    LoadedRepo,
    TextResource,
    load_repo,
    scan_repo,
    validate_name,
    validate_repo_name,
)
from .results import RepoResult, ResourceInfo, ToolCatalogEntry, ToolMutationResult, ToolStatsReport, ToolStatsRow


@dataclass(frozen=True)
class HarnessBinding:
    """One request's harness binding and loop, published as a contextvar.

    Internal: composition entry points read it to decide whether a call belongs
    to the running request.
    """

    harness: JITHarness
    context: AgentRunContext | None
    loop: asyncio.AbstractEventLoop


_BINDING: ContextVar[HarnessBinding | None] = ContextVar("zett_jit_binding", default=None)
_CALL_STACK: ContextVar[tuple[str, ...]] = ContextVar("zett_jit_stack", default=())
#: Rows of the overview, tools one question may name, and the length of an error each row may quote.
#: A row with its errors is under a thousand characters, so both fit one response comfortably.
MAX_PROFILE_ROWS = 20
MAX_PROFILE_NAMES = 10
PROFILE_ERROR_CHARS = 160

_PROFILE_CALL: ContextVar[tuple[JITHarness, ActiveToolCall] | None] = ContextVar("zett_jit_profile_call", default=None)


async def _run_sync_tool(function: Callable[..., Any], **arguments: Any) -> ToolResult:
    """Use one worker per active sync invocation, never a pool parents can block.

    Cancelling the await cannot stop Python's worker. It is daemonized, receives
    an independent Context, and can no longer dispatch after request detachment.
    """
    loop = asyncio.get_running_loop()
    completed = loop.create_future()
    copied = copy_context()

    def settle(value: Any, error: BaseException | None) -> None:
        if not completed.done():
            if error is None:
                completed.set_result(value)
            else:
                completed.set_exception(error)

    def run() -> None:
        try:
            value, error = copied.run(function, **arguments), None
        except BaseException as failure:
            value, error = None, failure
        try:
            loop.call_soon_threadsafe(settle, value, error)
        except RuntimeError:
            # An uninterruptible worker may finish after its loop has closed.
            pass

    Thread(target=run, name="zett-jit-tool", daemon=True).start()
    return await completed


class JITHarness:
    """An independently evolving toolbox of Python tools and Markdown notes.

    The toolbox is one repository: a Python package in a directory, written with
    ordinary file tools. :meth:`load_repo` imports it and registers every ``@tool``
    it defines, and lists the Markdown notes under ``resources/``. The notes are
    files the model reads itself, not tools.

    Args:
        directory: Where this harness keeps its repository. The repository is a
            subdirectory of it, created by :meth:`create_repo`. Nothing is
            written before then.
        tools: Application-owned base tools; they are never loaded from disk.
        manage_base_tools: Wrap base tools for unloading and profiling. False
            leaves their request registrations intact and skips their profiling.
        max_call_depth: Bound nested composition, including recursive calls.
        error_capacity: Maximum distinct recent errors across this harness.
        error_message_chars: Maximum characters retained in each error message.
        call_capacity: Maximum finished call records retained, oldest dropped
            first; lifetime aggregates survive eviction.

    Note:
        This is not a Python sandbox. Source has the process's filesystem,
        network, and import privileges. Unload controls harness dispatch, not
        what arbitrary Python can do. Session-local profiles retain a bounded
        window of managed dispatches without payloads; inspect them with profile().

    Examples:
        Create a repository, write a tool into it, and load it::

            harness = JITHarness("jit", tools=[read_file])
            path = harness.create_repo("helpers")["path"]
            Path(path, "preview.py").write_text(source)
            harness.load_repo()
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        tools: Sequence[AgentTool] = (),
        manage_base_tools: bool = True,
        max_call_depth: int = 32,
        error_capacity: int = 100,
        error_message_chars: int = 4000,
        call_capacity: int = MAX_RETAINED_CALLS,
    ) -> None:
        if max_call_depth < 1:
            raise ValueError("max_call_depth must be positive")
        if not isinstance(manage_base_tools, bool):
            raise ValueError("manage_base_tools must be a boolean")
        self.manage_base_tools = manage_base_tools
        self.max_call_depth = max_call_depth
        self._directory = Path(directory).expanduser().absolute()
        # Guards every registry mutation below. It is a plain re-entrant lock
        # because helpers such as load_repo call each other.
        self._lock = RLock()
        # True only while a module body executes. It stops that trusted code from
        # mutating the registry or dispatching tools through the harness.
        self._compiling = False
        # The request currently bound by attach(). Every wrapper checks this
        # identity, so a stale wrapper cannot run against a later request.
        self._context: AgentRunContext | None = None
        # Tasks performing composed calls, mapped to the request that owns them.
        # detach() cancels them so nested work cannot outlive its request.
        self._nested_tasks: dict[asyncio.Task[Any], AgentRunContext] = {}
        # Origin of every dynamic tool (always PYTHON) for the repository's
        # currently loaded version. Unloading keeps these entries.
        self._kinds: dict[str, ToolOrigin] = {}
        # The Python tools of the loaded version. A reload replaces them as one
        # set; unload only hides the names.
        self._dynamic: dict[str, AgentTool] = {}
        # The Markdown notes the last successful scan found. They are files the
        # model reads itself, so nothing about them is registered or hidden.
        self._resources: dict[str, TextResource] = {}
        # The repository this harness serves, if any: its name, and the imported
        # package while it is loaded. A failed first load leaves the name only.
        self._repo: str | None = None
        self._loaded: LoadedRepo | None = None
        # True while unload_repo() hides the loaded version.
        self._hidden = False
        # Whether the session directory was already searched for a repository
        # left by an earlier process; done once, on the first attach().
        self._adopted = False
        # Base tools bound for the current request: constructor tools plus any a
        # tool-registering extension added. Rebuilt on attach, dropped on detach.
        self._bases: dict[str, AgentTool] = {}
        # Tools passed to the constructor. They are the ones that stay known
        # across requests when a request adds no binding.
        self._provided: dict[str, AgentTool] = {}
        # Control tools registered for the current request; cleared on detach
        # because they close over that request's harness state.
        self._management: dict[str, AgentTool] = {}
        # Wrapper handlers created in this request. Registry synchronization
        # recognizes old wrappers (for example a restored policy baseline)
        # without mistaking them for a new base implementation or wrapping twice.
        self._wrapper_handlers: dict[int, Callable[..., Any]] = {}
        # Session-local counters, call records, and the recent-error cache. Kept
        # for the lifetime of this harness.
        self._profiler = ToolProfiler(
            error_capacity=error_capacity, error_message_chars=error_message_chars, call_capacity=call_capacity
        )
        # Every base name this harness has seen, accumulated across requests. An
        # unbound name is reported as "missing" instead of unknown.
        self._base_names: set[str] = set()
        # Names hidden from dispatch. Unload only edits this set; loading removes
        # the name again.
        self._unloaded: set[str] = set()
        for candidate in tools:
            validate_name(candidate.name)
            if candidate.name in self._provided:
                raise ValueError(f"Duplicate tool: {candidate.name}")
            if candidate.local_tool_search:
                raise ValueError("JIT harness does not manage client-side tool search")
            self._provided[candidate.name] = replace(candidate, parameters=deepcopy(candidate.parameters))
        self._base_names.update(self._provided)
        self._bases = dict(self._provided)

    def _check_mutation(self) -> None:
        if self._compiling:
            raise RuntimeError("Cannot change the harness while loading Python source")
        binding = _BINDING.get()
        if binding is not None and binding.harness is self and binding.context is not self._context:
            raise RuntimeError("This tool's request has already closed")

    def _known(self, name: str) -> None:
        if name in MANAGEMENT_NAMES:
            raise ValueError("JIT management tools cannot be changed or composed")
        if name not in self._kinds and name not in self._base_names:
            raise ValueError(f"Unknown tool: {name}")

    def origin(self, name: str) -> ToolOrigin:
        """Return which registry currently serves one tool name.

        Labels observability events and profile records: ``BASE`` for application
        tools, ``PYTHON`` for tools the repository defines, ``MANAGEMENT`` for the
        harness's own tools, and ``UNKNOWN`` for a
        name this harness has never seen. The answer does not depend on the
        request binding.
        """
        with self._lock:
            return self._resolve_origin(name)

    def _resolve_origin(self, name: str) -> ToolOrigin:
        if name in MANAGEMENT_NAMES:
            return ToolOrigin.MANAGEMENT
        if name in self._kinds:
            return self._kinds[name]
        if name in self._base_names:
            return ToolOrigin.BASE
        return ToolOrigin.UNKNOWN

    def _raw_tool(self, name: str, context: AgentRunContext | None) -> AgentTool:
        if self._compiling:
            raise RuntimeError("Cannot call tools while loading Python source")
        self._known(name)
        if context is not self._context:
            raise RuntimeError("This tool's request has already closed")
        if self._is_unloaded(name):
            raise ValueError(f"Tool {name!r} is unloaded")
        if context is not None and name not in context.tools:
            raise ValueError(f"Tool {name!r} is not registered in the current request")
        registered = self._dynamic.get(name) or self._bases.get(name)
        if registered is None:
            raise ValueError(f"Missing base tool: {name}")
        return registered

    def _is_unloaded(self, name: str) -> bool:
        return name in self._unloaded and (self.manage_base_tools or name in self._kinds)

    def _wrapper(self, registered: AgentTool, context: AgentRunContext | None) -> AgentTool:
        name = registered.name

        async def guarded(**arguments: Any) -> ToolResult:
            # Resolve on every invocation: an old wrapper cannot call a tool
            # after unload or retain a superseded implementation.
            return await self._profiled_call(name, context, arguments)

        if context is not None:
            self._wrapper_handlers[id(guarded)] = guarded
        # Preserve provider deferral: a base tool marked deferred must keep
        # reaching the provider as such, and the manager never defers its own.
        return replace(registered, handler=guarded, parameters=deepcopy(registered.parameters))

    async def _execute_handler(
        self, name: str, target: AgentTool, context: AgentRunContext | None, arguments: dict[str, Any]
    ) -> ToolResult:
        binding_token = _BINDING.set(HarnessBinding(self, context, asyncio.get_running_loop()))
        stack_token = _CALL_STACK.set((*_CALL_STACK.get(), name))
        worker_token = _SYNC_TOOL_RUNNER.set(_run_sync_tool)
        try:
            return await target(arguments)
        finally:
            _SYNC_TOOL_RUNNER.reset(worker_token)
            _CALL_STACK.reset(stack_token)
            _BINDING.reset(binding_token)

    async def _profiled_call(self, name: str, context: AgentRunContext | None, arguments: dict[str, Any]) -> ToolResult:
        # Invalid source-loading calls and stale references cannot append to a
        # new request's profile. Every actual dispatch attempt is tracked once.
        with self._lock:
            if self._compiling:
                raise RuntimeError("Cannot call tools while loading Python source")
            if context is not self._context:
                raise RuntimeError("This tool's request has already closed")
            origin = self._resolve_origin(name)
            if origin is ToolOrigin.BASE and not self.manage_base_tools:
                target = self._raw_tool(name, context)
                if len(_CALL_STACK.get()) >= self.max_call_depth:
                    raise RuntimeError(f"JIT tool call depth exceeded {self.max_call_depth}")
            else:
                target = None
        if target is not None:
            return await self._execute_handler(name, target, context, arguments)
        previous = _PROFILE_CALL.get()
        parent = previous[1] if previous is not None and previous[0] is self else None
        call_id = current_tool_call_id()
        # Direct Python calls do not bind an ID; do not reuse their parent's ID.
        if call_id is None or (parent is not None and call_id == parent.call_id):
            call_id = new_uuid7()
        active = self._profiler.start(
            name,
            arguments,
            call_id=call_id,
            parent=parent,
            origin=origin,
        )
        token = _PROFILE_CALL.set((self, active))
        output = None
        error: BaseException | None = None
        status = ToolCallStatus.REJECTED
        try:
            with self._lock:
                if name in MANAGEMENT_NAMES:
                    if context is None or name not in context.tools:
                        raise ValueError(f"Tool {name!r} is not registered in the current request")
                    target = self._management[name]
                else:
                    target = self._raw_tool(name, context)
            stack = _CALL_STACK.get()
            if len(stack) >= self.max_call_depth:
                raise RuntimeError(f"JIT tool call depth exceeded {self.max_call_depth}")
            status = ToolCallStatus.FAILED
            output = await self._execute_handler(name, target, context, arguments)
            status = ToolCallStatus.COMPLETED
            return output
        except asyncio.CancelledError as failure:
            status, error = ToolCallStatus.CANCELLED, failure
            raise
        except BaseException as failure:
            error = failure
            raise
        finally:
            try:
                self._profiler.finish(active, status, output, error)
            finally:
                _PROFILE_CALL.reset(token)

    def _publish(self, name: str) -> None:
        context = self._context
        if context is not None:
            if self._is_unloaded(name):
                context.tools.pop(name, None)
            else:
                registered = self._dynamic.get(name) or self._bases.get(name)
                if registered is not None:
                    context.tools[name] = (
                        registered
                        if name in self._bases and not self.manage_base_tools
                        else self._wrapper(registered, context)
                    )

    @property
    def active(self) -> bool:
        """Return whether this harness is currently bound to one request.

        The extension checks this before releasing a session, so it never drops
        a harness whose request is still running.
        """
        with self._lock:
            return self._context is not None

    def attach(self, context: AgentRunContext, management_tools: Sequence[AgentTool] = ()) -> None:
        """Bind this harness to one request and mount its callable tools.

        Rebuilds the base registry from the request's current tools, installs
        harness-owned wrappers, hides unloaded tools, and registers the
        management tools the model may call. Called by
        :class:`~zett_agent.jit_harness.extension.JITHarnessExtension` once per
        request before the model runs; attaching twice without :meth:`detach`
        raises instead of silently replacing the binding.

        Args:
            context: Request-scoped tool registry and session identity to bind.
            management_tools: Harness control tools to expose to the model.
        """
        with self._lock:
            if self._context is not None:
                raise RuntimeError("This harness already has an active request")
            bases = dict(self._provided)
            for registered in context.tools.values():
                validate_name(registered.name)
                if registered.name in self._dynamic:
                    raise ValueError(f"Base tool conflicts with dynamic tool: {registered.name}")
                if registered.local_tool_search:
                    raise ValueError("JIT harness does not manage client-side tool search")
                bases[registered.name] = registered
            self._context = context
            self._bases = bases
            self._base_names.update(bases)
            self._adopt_existing()
            for name in self._base_names | self._kinds.keys():
                self._publish(name)
            for registered in management_tools:
                self._management[registered.name] = registered
                context.register_tool(self._wrapper(registered, context))

    def sync_tools(self, context: AgentRunContext) -> None:
        """Adopt new/replaced request base tools without restoring policy removals.

        Called by the extension before model requests and tool dispatch so tools
        registered after attach(), including policy replacements, receive the
        same unload/profiling treatment. Missing registrations stay missing.
        Wrappers already owned by this request are recognized rather than
        re-adopted as raw tools. Disabled base management preserves raw handlers.
        """
        with self._lock:
            if context is not self._context:
                raise RuntimeError("This tool's request has already closed")
            replacements: dict[str, AgentTool] = {}
            for name, registered in tuple(context.tools.items()):
                if name != registered.name:
                    raise ValueError("Tool registry key must match the registered tool name")
                if id(registered.handler) in self._wrapper_handlers:
                    continue
                validate_name(name)
                if name in self._kinds:
                    raise ValueError(f"Base tool conflicts with dynamic tool: {name}")
                if registered.local_tool_search:
                    raise ValueError("JIT harness does not manage client-side tool search")
                replacements[name] = registered
            self._bases.update(replacements)
            self._base_names.update(replacements)
            for name in tuple(context.tools):
                if self._is_unloaded(name):
                    context.tools.pop(name, None)
                elif name in replacements and self.manage_base_tools:
                    self._publish(name)

    def detach(self, context: AgentRunContext) -> None:
        """Release one request binding and stop tracking its nested calls.

        Deliberately tolerant of repeated or late cleanup: a context that is not
        the currently bound one is ignored, so releasing an older request cannot
        dismantle the request that is running now.
        """
        with self._lock:
            if self._context is context:
                self._context = None
                # Extension tools often close over a request. Do not retain
                # those callables or allow them to run after their request.
                self._bases = dict(self._provided)
                self._management.clear()
                self._wrapper_handlers.clear()
                for task, owner in tuple(self._nested_tasks.items()):
                    if owner is context:
                        task.get_loop().call_soon_threadsafe(task.cancel)

    def _adopt_existing(self) -> None:
        """Pick up the repository an earlier process left in this directory, once.

        A resumed session should find its tools again without the model asking.
        Only a single repository is adopted, and a load failure is not fatal here:
        the model can call ``repo_load`` to see the error and fix the files.
        """
        if self._adopted:
            return
        self._adopted = True
        if self._repo is not None or not self._directory.is_dir():
            return
        found = [
            path.name
            for path in self._directory.iterdir()
            if path.is_dir() and (path / INIT_FILE).is_file() and not path.name.startswith(".")
        ]
        if len(found) == 1:
            self._repo = found[0]
            try:
                self._load()
            except Exception:
                # The files stay on disk and ``repo_load`` reports the error.
                pass

    @property
    def repo(self) -> str | None:
        """The repository's name, or None before one exists."""
        return self._repo

    @property
    def repo_path(self) -> Path | None:
        """The repository's directory, or None before one exists."""
        return None if self._repo is None else self._directory / self._repo

    def _result(self, status: Literal["created", "loaded", "unloaded"] | None = None) -> RepoResult:
        assert self._repo is not None
        if status is None:
            status = "unloaded" if self._hidden or self._loaded is None else "loaded"
        return {
            "repo": self._repo,
            "path": str(self._directory / self._repo),
            "status": status,
            "tools": sorted(self._dynamic),
            "resources": self._resource_infos(),
        }

    def _resource_infos(self) -> list[ResourceInfo]:
        assert self._repo is not None
        root = self._directory / self._repo
        return [
            {
                "name": resource.name,
                "kind": resource.kind,
                "description": resource.description,
                "path": str(root / resource.path),
            }
            for resource in sorted(self._resources.values(), key=lambda item: item.name)
        ]

    def create_repo(self, repo: str) -> RepoResult:
        """Create this harness's repository, or return the one that already exists.

        The repository is a Python package in a subdirectory named ``repo``. There
        is one per harness: asking again for the same name returns it unchanged,
        and asking for another name is an error. Write modules into the returned
        path with ordinary file tools, then call :meth:`load_repo`.

        Args:
            repo: Directory name, 1–64 letters, digits, underscores, or hyphens.
        """
        with self._lock:
            self._check_mutation()
            validate_repo_name(repo)
            self._adopt_existing()
            if self._repo is not None and self._repo != repo:
                raise ValueError(f"This session already has the repository {self._repo!r}; keep using it")
            path = self._directory / repo
            existed = (path / INIT_FILE).is_file()
            if not existed:
                path.mkdir(parents=True, exist_ok=True)
                (path / INIT_FILE).write_text(SKELETON, encoding="utf-8")
            self._repo = repo
            return self._result(None if existed else "created")

    def _load(self) -> None:
        """Make the repository's tools current and list its notes.

        The files are checked first, so a bad one fails the call before anything
        changes. When no Python source differs from what is imported, nothing is
        imported again: the modules are resident, and importing them would only run
        their bodies a second time. Otherwise the new package is imported before
        anything changes, so a failure keeps the previous version registered. Module
        bodies run with ``_compiling`` set, which stops them from mutating the
        harness or dispatching tools.
        """
        assert self._repo is not None
        directory = self._directory / self._repo
        scan = scan_repo(directory)
        self._resources = scan.resources
        if self._loaded is not None and self._loaded.fingerprint == scan.fingerprint:
            self._reveal()
            return
        self._compiling = True
        try:
            loaded = load_repo(directory, self, scan)
        finally:
            self._compiling = False
        clash = sorted(loaded.tools.keys() & self._base_names)
        if clash:
            loaded.close()
            raise ValueError(f"Tool {clash[0]!r} is already an application tool; choose another name")
        previous, previous_names = self._loaded, set(self._dynamic)
        self._loaded = loaded
        self._dynamic = dict(loaded.tools)
        self._kinds = dict.fromkeys(loaded.tools, ToolOrigin.PYTHON)
        # A fresh version starts loaded, whatever state the old one was in.
        self._hidden = False
        self._unloaded -= previous_names | self._dynamic.keys()
        context = self._context
        if context is not None:
            for name in previous_names - self._dynamic.keys():
                context.tools.pop(name, None)
            for name in self._dynamic:
                self._publish(name)
        if previous is not None:
            previous.close()

    def _reveal(self) -> None:
        """Make every tool of the loaded version callable again, importing nothing."""
        revived = set(self._dynamic) if self._hidden else self._unloaded & self._dynamic.keys()
        self._hidden = False
        self._unloaded -= revived
        for name in revived:
            self._publish(name)

    def load_repo(self) -> RepoResult:
        """Load the repository, or confirm it is loaded, and list what it provides.

        Safe to call at any time. The answer always names the tools and describes the
        Markdown notes under ``resources/`` (name, kind, description, path); the notes
        are files the model reads itself, so none of them becomes a tool.

        * **Not loaded yet**: every module below the package root is imported and each
          top-level ``@tool`` is registered.
        * **Loaded and no Python source changed**: nothing is imported again. Tools
          that were hidden by :meth:`unload_repo` or :meth:`unload_tool` are made
          callable, so calling it twice leaves the same state as calling it once.
        * **Loaded and a source changed**: the whole package is imported afresh and
          replaces the previous version in one step. Changes are found by content, so
          even an edit of the same length is seen.

        A failure leaves the previous version in place and raises a ``ValueError``
        naming the file and line. Module bodies run with the application's
        privileges; their side effects cannot be rolled back.
        """
        with self._lock:
            self._check_mutation()
            if self._repo is None:
                raise ValueError("Create the repository first with repo_create")
            self._load()
            return self._result()

    def unload_repo(self) -> RepoResult:
        """Hide every tool of the repository; the files stay on disk.

        Calling a hidden tool is rejected. :meth:`load_repo` restores them without
        importing anything again, unless a Python file changed meanwhile. The notes
        are files, so they stay listed.
        """
        with self._lock:
            self._check_mutation()
            if self._repo is None:
                raise ValueError("Create the repository first with repo_create")
            self._unloaded.update(self._dynamic)
            self._hidden = True
            for name in self._dynamic:
                self._publish(name)
            return self._result("unloaded")

    def close(self) -> None:
        """Remove the repository's modules and import hook from the interpreter.

        The files stay on disk. The harness can load them again afterwards.
        """
        with self._lock:
            if self._loaded is not None:
                self._loaded.close()
                self._loaded = None

    def unload_tool(self, name: str) -> ToolMutationResult:
        """Unload a callable registration while retaining its available implementation.

        Dynamic tools stay imported; base implementations are rebound per request.
        Requires manage_base_tools=True for base tools. :meth:`load_tool` restores
        a bound tool, and :meth:`unload_repo` hides every repository tool at once.
        """
        with self._lock:
            self._check_mutation()
            self._known(name)
            if name in self._base_names and not self.manage_base_tools:
                raise ValueError("Base tool management is disabled; cannot unload a base tool")
            self._unloaded.add(name)
            self._publish(name)
            return {"name": name, "status": "unloaded"}

    def load_tool(self, name: str) -> ToolMutationResult:
        """Restore a retained tool; acknowledge only its name and loaded state.

        Fails explicitly if the tool's base implementation is not bound.
        """
        with self._lock:
            self._check_mutation()
            self._known(name)
            if name in self._base_names and not self.manage_base_tools:
                raise ValueError("Base tool management is disabled; cannot load a base tool")
            if name not in self._dynamic and name not in self._bases:
                raise ValueError(f"Missing base tool: {name}")
            self._unloaded.discard(name)
            self._publish(name)
            return {"name": name, "status": "loaded"}

    def list_tools(self) -> list[ToolCatalogEntry]:
        """List one compact catalog entry per known tool.

        Entries come from :meth:`describe`, so a long catalog stays readable and
        never repeats source or call history.
        """
        with self._lock:
            return [self.describe(name) for name in sorted(self._base_names | self._kinds.keys())]

    def describe(self, name: str) -> ToolCatalogEntry:
        """Return one tool's compact catalog entry.

        Holds the name, origin, status, and description, without measurements.
        Read profile() for statistics.
        """
        with self._lock:
            self._known(name)
            registered = self._dynamic.get(name) or self._bases.get(name)
            status: Literal["loaded", "unloaded", "missing", "unavailable"] = (
                "unloaded" if self._is_unloaded(name) else "loaded" if registered is not None else "missing"
            )
            if self._context is not None and status == "loaded" and name not in self._context.tools:
                status = "unavailable"
            return {
                "name": name,
                "origin": self._resolve_origin(name),
                "status": status,
                "description": registered.description if registered is not None else "",
            }

    def profile(self, name: str | None = None, *, offset: int = 0, limit: int = 100) -> JITHarnessProfile:
        """Read aggregate counters and a page of finished, payload-free calls.

        All managed dispatch attempts through harness wrappers are counted, including
        nested calls, validation failures, and cancellations. Middleware and
        lifecycle hooks run outside this measured boundary. Histories remain
        in this harness across requests/reloads/unload, bounded by call_capacity,
        and are not written to disk. Base calls are excluded when
        manage_base_tools=False. Export with profile().model_dump_json().
        """
        return self._profiler.report(name, offset, limit)

    def profile_tools(self, names: Sequence[str] | None = None) -> ToolStatsReport:
        """Summarize how tools are doing, in a size a model can read at once.

        Any tool can be asked about by name, an application tool as well as one the
        model wrote: ``names`` lists them, and the answer has one row for each in the
        order given, with up to three recent distinct errors, which is where to look
        before fixing a tool. A tool that was never called gets a row of zeros, so
        "unused" is an answer too. At most :data:`MAX_PROFILE_NAMES` names are taken.

        Without ``names`` the answer is an overview: every tool of the repository,
        called or not, and every application tool that has been called, most called
        first, one recent error each, and at most :data:`MAX_PROFILE_ROWS` rows with
        ``omitted`` counting the rest. The management tools are never measured.

        Application tools are measured only with ``manage_base_tools=True``; with it
        off they have no numbers, so asking for one is an error instead of a row of
        zeros that would read as "never used". This is a view of :meth:`profile` and
        :meth:`tool_errors`, which keep the full measurements.

        Args:
            names: Tools to look at, or None for the overview.

        Raises:
            ValueError: If a name is unknown, is a management tool, is an application
                tool while base management is off, or if there are too many names.
        """
        explicit = names is not None
        with self._lock:
            known = self._base_names | self._kinds.keys()
            if names is None:
                wanted = sorted(known)
            else:
                wanted = list(dict.fromkeys(names))
                if not wanted or len(wanted) > MAX_PROFILE_NAMES:
                    raise ValueError(f"Name between 1 and {MAX_PROFILE_NAMES} tools")
                unknown = [item for item in wanted if item not in known]
                if unknown:
                    raise ValueError(f"Unknown tool: {', '.join(unknown)} (the repo_* tools are not measured)")
                if not self.manage_base_tools:
                    unmeasured = [item for item in wanted if item not in self._kinds]
                    if unmeasured:
                        raise ValueError(
                            f"Application tools are not measured here: {', '.join(unmeasured)} "
                            "(base tool management is off)"
                        )
            catalog = {item: self.describe(item) for item in wanted}
        rows: list[ToolStatsRow] = []
        for tool_name in wanted:
            entry = catalog[tool_name]
            stats = self._profiler.summary(tool_name)
            if not explicit and entry["origin"] is ToolOrigin.BASE and stats.call_count == 0:
                continue
            finished = stats.completed_count + stats.failed_count
            errors = self._profiler.errors(tool_name, 0, 3 if explicit else 1).errors
            rows.append(
                {
                    "name": tool_name,
                    "origin": entry["origin"],
                    "status": entry["status"],
                    "calls": stats.call_count,
                    "success_rate": round(stats.completed_count / finished, 3) if finished else None,
                    "failed": stats.failed_count,
                    "avg_ms": round(stats.average_duration_ms, 1) if stats.average_duration_ms is not None else None,
                    "avg_output_chars": (
                        round(stats.average_output_chars) if stats.average_output_chars is not None else None
                    ),
                    "recent_errors": [
                        f"{record.error_type}: {(record.message or '')[:PROFILE_ERROR_CHARS]}" for record in errors
                    ],
                }
            )
        if explicit:
            return {"tools": rows, "omitted": 0}
        rows.sort(key=lambda row: (-row["calls"], row["name"]))
        return {"tools": rows[:MAX_PROFILE_ROWS], "omitted": max(0, len(rows) - MAX_PROFILE_ROWS)}

    def profile_stats(self, name: str) -> ToolProfileStats:
        """Read a tool's full aggregates without fetching its call records."""
        return self._profiler.summary(name)

    def tool_errors(self, name: str | None = None, *, offset: int = 0, limit: int = 5) -> ToolErrorReport:
        """Read LRU-bounded recent errors without refreshing their recency.

        Deduplicates by tool, exception type, and full message. Reoccurrences
        refresh recency; oldest distinct errors are evicted at error_capacity.
        Details include bounded message text and latest invocation context.
        Lifetime profiling counters survive eviction; occurrence counts cover
        only cache residence. Cancelled dispatches are not tool errors.
        Error messages may contain sensitive data or untrusted instructions.
        This read never triggers retries or source updates.
        """
        return self._profiler.errors(name, offset, limit)

    async def call_tool(self, name: str, **arguments: Any) -> ToolResult:
        """Call a loaded tool through checks, hooks, and handler middleware.

        Nested calls produce no additional model messages or execution-trace
        events. Interactive hooks may still request application/user input.
        Normal exceptions propagate to the outer tool unless Python catches them.
        """
        binding = _BINDING.get()
        context = binding.context if binding is not None and binding.harness is self else None
        if context is None and self._context is not None:
            raise RuntimeError("Use call_tool inside this harness's active tool execution")
        with self._lock:
            self._known(name)
            raw = self._dynamic.get(name) or self._bases.get(name)
            # Invalid dependency attempts retain their profiling, without
            # manufacturing schema/handler metadata for a missing base tool.
            if raw is None or self._is_unloaded(name) or context is not None and name not in context.tools:
                registered = None
            else:
                registered = self._wrapper(raw, context)
        if registered is None:
            return await self._profiled_call(name, context, dict(arguments))
        if context is None:
            return await registered(arguments)
        call = ToolCall(new_uuid7(), name, dict(arguments))
        task = asyncio.current_task()
        previous_owner = self._nested_tasks.get(task) if task is not None else None
        if task is not None:
            self._nested_tasks[task] = context
        token = _bind_tool_call(call.id)
        error: Exception | None = None
        try:
            for extension in context.extensions:
                await extension.before_tool(context, call)
            try:
                # The composition wrapper resolves live definitions itself.
                # Keep it even when the request's base handler is unwrapped.
                output = await invoke_tool(context, call, registered, refresh_registration=False)
                content = self._result_content(registered, output)
            except Exception as failure:
                error = failure
                content = str(failure)
                output = None
            # Hooks may mutate multimodal content in place. Compare against an
            # independent value, not the list shared with the ToolMessage.
            original_content = deepcopy(content)
            result = ToolMessage(tool_call_id=call.id, name=name, content=content, success=error is None)
            for extension in context.extensions:
                await extension.after_tool(context, call, result, error)
            if error is not None:
                raise error
            # Preserve raw Python values unless a lifecycle hook transformed
            # their model-facing representation.
            return output if result.content == original_content else result.content
        finally:
            if task is not None:
                if previous_owner is None:
                    self._nested_tasks.pop(task, None)
                else:
                    self._nested_tasks[task] = previous_owner
            _reset_tool_call(token)

    @staticmethod
    def _result_content(registered: AgentTool, output: ToolResult) -> str | list[TextContent | ImageContent]:
        if isinstance(output, (TextContent, ImageContent)):
            return [output]
        if (
            isinstance(output, list)
            and output
            and all(isinstance(part, (TextContent, ImageContent)) for part in output)
        ):
            return output
        return registered.serialize_result(output)

    def call_tool_sync(self, name: str, **arguments: Any) -> ToolResult:
        """Compose from a synchronous tool worker, on its original request loop.

        Never call this blocking method on an event-loop thread. Cancellation
        cannot forcibly stop an already running Python worker thread.
        """
        binding = _BINDING.get()
        if binding is None or binding.harness is not self:
            raise RuntimeError("call_tool_sync must run inside a synchronous JIT tool")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("call_tool_sync cannot block an event-loop thread; use await call_tool")
        if binding.context is not self._context or binding.loop.is_closed():
            raise RuntimeError("This tool's request has already closed")
        future = asyncio.run_coroutine_threadsafe(self.call_tool(name, **arguments), binding.loop)
        return future.result()
