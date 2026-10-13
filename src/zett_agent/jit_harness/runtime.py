"""Session-owned Python tool environments with a shared invocation boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar, copy_context
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock, Thread
from typing import Any

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
from .enums import ToolCallStatus, ToolKind, ToolOrigin
from .errors import ToolErrorReport
from .profiling import MAX_RETAINED_CALLS, ActiveToolCall, JITHarnessProfile, ToolProfiler, ToolProfileStats
from .python import _load_python_tool
from .results import ToolCatalogEntry, ToolInspection, ToolMutationResult
from .snapshot import (
    _ENTRYPOINT_FILE,
    _MANAGEMENT_NAMES,
    JITHarnessSnapshot,
    PythonToolSource,
    TextToolSource,
    _validate_file_path,
    _validate_name,
)
from .text import load_text_tool


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
    """An independently evolving toolbox of Python and Markdown capabilities.

    Args:
        snapshot: Definitions to activate. Activation executes Python module
            bodies, including imports, but does not call tool entry points.
        tools: Application-owned base tools; they are never serialized as code.
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
        what arbitrary Python can do. Module globals are not persisted.
        Session-local profiles retain a bounded window of managed dispatches
        without payloads, independently of executable snapshots; inspect them
        with profile().

    Examples:
        Save definitions for another session::

            harness = JITHarness(tools=[read_file])
            harness.create_tool(source, entrypoint="preview_file")
            harness.save("harness.json")
    """

    def __init__(
        self,
        snapshot: JITHarnessSnapshot | None = None,
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
        # Guards every registry mutation and snapshot read below. It is a plain
        # re-entrant lock because helpers such as create_tool call each other.
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
        # Serializable Python or Markdown source for dynamic tools. Unloading keeps
        # these entries; this mapping is what snapshots and inspect_tool expose.
        self._sources: dict[str, PythonToolSource | TextToolSource] = {}
        # Executable Python tools and non-executable Markdown readers. Updates
        # replace the entry; unload only hides the name, without reloading source.
        self._dynamic: dict[str, AgentTool] = {}
        # Base tools bound for the current request: constructor tools plus any a
        # tool-registering extension added. Rebuilt on attach, dropped on detach.
        self._bases: dict[str, AgentTool] = {}
        # Tools passed to the constructor. They are never serialized and are the
        # ones that stay known across requests when a request adds no binding.
        self._provided: dict[str, AgentTool] = {}
        # Control tools registered for the current request; cleared on detach
        # because they close over that request's harness state.
        self._management: dict[str, AgentTool] = {}
        # Wrapper handlers created in this request. Registry synchronization
        # recognizes old wrappers (for example a restored policy baseline)
        # without mistaking them for a new base implementation or wrapping twice.
        self._wrapper_handlers: dict[int, Callable[..., Any]] = {}
        # Session-local counters, call records, and the recent-error cache. Kept
        # for the lifetime of this harness and never written into snapshots.
        self._profiler = ToolProfiler(
            error_capacity=error_capacity, error_message_chars=error_message_chars, call_capacity=call_capacity
        )
        initial = snapshot or JITHarnessSnapshot()
        # Every base name this harness has seen, accumulated across requests. An
        # unbound name is reported as "missing" instead of unknown.
        self._base_names = set(initial.base_tools)
        # Names hidden from dispatch while their source is retained. Unload only
        # edits this set; loading removes the name again.
        self._unloaded = set(initial.unloaded_tools)
        dynamic_names = {record.name for record in (*initial.tools, *initial.text_tools)}
        for candidate in tools:
            _validate_name(candidate.name)
            if candidate.name in dynamic_names or candidate.name in self._provided:
                raise ValueError(f"Duplicate tool: {candidate.name}")
            if candidate.local_tool_search:
                raise ValueError("JIT harness does not manage client-side tool search")
            self._provided[candidate.name] = replace(candidate, parameters=deepcopy(candidate.parameters))
        self._base_names.update(self._provided)
        self._bases = dict(self._provided)
        for record in initial.tools:
            registered, rebuilt = self._compile(record.source, record.entrypoint, record.files)
            if registered.name != record.name:
                raise ValueError(f"Source entrypoint does not match stored tool name {record.name!r}")
            self._sources[record.name] = rebuilt
            self._dynamic[record.name] = registered
        for text_record in initial.text_tools:
            self._sources[text_record.name] = text_record.model_copy(deep=True)
            self._dynamic[text_record.name] = load_text_tool(text_record)

    def _compile(
        self, source: str, entrypoint: str, files: Mapping[str, str] | None = None
    ) -> tuple[AgentTool, PythonToolSource]:
        self._compiling = True
        try:
            return _load_python_tool(source, entrypoint, self, files)
        finally:
            self._compiling = False

    def _check_mutation(self) -> None:
        if self._compiling:
            raise RuntimeError("Cannot change the harness while loading Python source")
        binding = _BINDING.get()
        if binding is not None and binding.harness is self and binding.context is not self._context:
            raise RuntimeError("This tool's request has already closed")

    def _known(self, name: str) -> None:
        if name in _MANAGEMENT_NAMES:
            raise ValueError("JIT management tools cannot be changed or composed")
        if name not in self._sources and name not in self._base_names:
            raise ValueError(f"Unknown tool: {name}")

    def origin(self, name: str) -> ToolOrigin:
        """Return which registry currently serves one tool name.

        Labels observability events and profile records: ``BASE`` for application
        tools, ``PYTHON`` for executable dynamic tools, ``TEXT`` for Markdown
        workflows, ``MANAGEMENT`` for
        the harness control functions, and ``UNKNOWN`` for a name this harness has
        never seen. The answer does not depend on the request binding.
        """
        with self._lock:
            return self._resolve_origin(name)

    def _resolve_origin(self, name: str) -> ToolOrigin:
        if name in _MANAGEMENT_NAMES:
            return ToolOrigin.MANAGEMENT
        if name in self._sources:
            return ToolOrigin.TEXT if isinstance(self._sources[name], TextToolSource) else ToolOrigin.PYTHON
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
        return name in self._unloaded and (self.manage_base_tools or name in self._sources)

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
                if name in _MANAGEMENT_NAMES:
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
                _validate_name(registered.name)
                if registered.name in self._dynamic:
                    raise ValueError(f"Base tool conflicts with dynamic tool: {registered.name}")
                if registered.local_tool_search:
                    raise ValueError("JIT harness does not manage client-side tool search")
                bases[registered.name] = registered
            self._context = context
            self._bases = bases
            self._base_names.update(bases)
            for name in self._base_names | self._sources.keys():
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
                _validate_name(name)
                if name in self._sources:
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

    def _prepare_tool(
        self,
        source: str,
        *,
        kind: ToolKind,
        entrypoint: str | None,
        name: str | None,
        description: str | None,
        files: Mapping[str, str] | None = None,
    ) -> tuple[AgentTool, PythonToolSource | TextToolSource]:
        if kind is ToolKind.TEXT:
            if entrypoint is not None:
                raise ValueError("Text tools do not have a Python entrypoint")
            if files:
                raise ValueError("Text tools have a single document, not a package")
            if name is None or description is None:
                raise ValueError("Text tools require name and description")
            record = TextToolSource(name=name, source=source, description=description)
            return load_text_tool(record), record
        if entrypoint is None:
            raise ValueError("Python tools require entrypoint")
        if name is not None or description is not None:
            raise ValueError("Python tool name and description come from the @tool entrypoint")
        return self._compile(source, entrypoint, files)

    def create_tool(
        self,
        source: str,
        *,
        entrypoint: str | None = None,
        kind: ToolKind = ToolKind.PYTHON,
        name: str | None = None,
        description: str | None = None,
        files: Mapping[str, str] | None = None,
    ) -> ToolMutationResult:
        """Create a Python tool or a reusable Markdown workflow after validation.

        Imports and module-level code execute with application privileges.
        Their external side effects cannot be rolled back on validation failure.
        With kind=TEXT, source is inert Markdown and name/description are required
        instead of entrypoint. Calling a loaded text tool reads a bounded page;
        the model follows its workflow using separately available tools.
        A Python tool may ship a package: ``files`` maps relative module paths to
        source, reachable from the entry point with relative imports. Source and
        files are validated as one unit before anything is registered.
        Returns only the name and resulting state, not metadata or profiling.
        """
        with self._lock:
            self._check_mutation()
            registered, record = self._prepare_tool(
                source,
                kind=ToolKind(kind),
                entrypoint=entrypoint,
                name=name,
                description=description,
                files=files,
            )
            if record.name in self._sources or record.name in self._base_names:
                raise ValueError(f"Tool {record.name!r} already exists; use update_tool or load_tool")
            self._sources[record.name] = record
            self._dynamic[record.name] = registered
            self._publish(record.name)
            return {"name": record.name, "status": "loaded"}

    def update_tool(
        self,
        name: str,
        source: str | None = None,
        *,
        entrypoint: str | None = None,
        description: str | None = None,
        files: Mapping[str, str] | None = None,
        remove: Sequence[str] = (),
    ) -> ToolMutationResult:
        """Atomically replace or extend one dynamic tool; preserve its unload state.

        The existing kind is preserved; text updates need replacement source and
        may replace the description. Python updates may omit source and
        entrypoint to keep them, replace or add package files, and delete files
        by path. Everything the call provides is validated and compiled as one
        new version, so a failed load retains the previous definition and a
        multi-file refactor is applied at once. Executions already started keep
        their original implementation; later calls use the new one.
        Returns only the name and resulting state, not metadata or profiling.
        """
        with self._lock:
            self._check_mutation()
            self._known(name)
            if name not in self._sources:
                raise ValueError("Only dynamic tools can be updated")
            previous = self._sources[name]
            if isinstance(previous, TextToolSource):
                if files is not None or remove:
                    raise ValueError("Text tools have a single document, not a package")
                if source is None:
                    raise ValueError("Text tools require replacement source")
                registered, record = self._prepare_tool(
                    source,
                    kind=ToolKind.TEXT,
                    entrypoint=entrypoint,
                    name=name,
                    description=description if description is not None else previous.description,
                )
            else:
                if source is None and files is None and not remove:
                    raise ValueError("Python updates need source, files, or remove")
                updated = dict(previous.files)
                for path in remove:
                    _validate_file_path(path)
                    if path not in updated:
                        raise ValueError(f"Unknown package file: {path}")
                    del updated[path]
                updated.update(files or {})
                registered, record = self._prepare_tool(
                    previous.source if source is None else source,
                    kind=ToolKind.PYTHON,
                    entrypoint=previous.entrypoint if entrypoint is None else entrypoint,
                    name=None,
                    description=description,
                    files=updated,
                )
            if record.name != name:
                raise ValueError("update_tool cannot rename a tool")
            self._sources[name] = record
            self._dynamic[name] = registered
            self._publish(name)
            return {"name": name, "status": "unloaded" if name in self._unloaded else "loaded"}

    def unload_tool(self, name: str) -> ToolMutationResult:
        """Unload a callable registration while retaining its available implementation.

        Dynamic source is retained; base implementations are rebound per request.
        Requires manage_base_tools=True for base tools. The resulting state is
        serialized as ``unloaded``; load_tool() restores a bound tool.
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
            return [self.describe(name) for name in sorted(self._base_names | self._sources.keys())]

    def describe(self, name: str) -> ToolCatalogEntry:
        """Return one tool's compact catalog entry.

        Holds the name, origin, status, and description, without measurements.
        Read inspect_tool() for schema and source, or profile() for statistics.
        """
        with self._lock:
            self._known(name)
            registered = self._dynamic.get(name) or self._bases.get(name)
            status = "unloaded" if self._is_unloaded(name) else "loaded" if registered is not None else "missing"
            if self._context is not None and status == "loaded" and name not in self._context.tools:
                status = "unavailable"
            return {
                "name": name,
                "origin": self._resolve_origin(name),
                "status": status,
                "description": registered.description if registered is not None else "",
            }

    def definition(self, name: str) -> ToolInspection:
        """Read metadata and the argument schema, without source or measurements."""
        with self._lock:
            info: ToolInspection = {**self.describe(name)}
            registered = self._dynamic.get(name) or self._bases.get(name)
            if registered is not None:
                info["parameters"] = deepcopy(registered.parameters)
                info["guidelines"] = list(registered.guidelines)
            source = self._sources.get(name)
            if isinstance(source, PythonToolSource):
                info["entrypoint"] = source.entrypoint
                if source.files:
                    # A Python tool is a package whose entry point is __init__.
                    sizes = {path: len(text) for path, text in sorted(source.files.items())}
                    info["files"] = {_ENTRYPOINT_FILE: len(source.source), **sizes}
            return info

    def source(self, name: str, path: str | None = None) -> str:
        """Read exact retained source of one tool file; base tools have none.

        ``path`` selects an extra module of a Python tool package, and the
        default or ``__init__.py`` reads the entry point module. Text tools,
        which have a single document, reject a path.
        """
        with self._lock:
            self._known(name)
            record = self._sources.get(name)
            if record is None:
                raise ValueError(f"Base tool {name!r} has no retained Python source")
            if isinstance(record, TextToolSource):
                if path is not None:
                    raise ValueError("Text tools hold a single document, not a package")
                return record.source
            if path is None or path == _ENTRYPOINT_FILE:
                return record.source
            _validate_file_path(path)
            try:
                return record.files[path]
            except KeyError:
                raise ValueError(f"Unknown package file: {path}") from None

    def inspect_tool(self, name: str) -> ToolInspection:
        """Return metadata, schema, guidelines, and retained source.

        This is the detailed application view: the compact entry from
        :meth:`describe` plus the argument schema, guidelines, and source.
        Measurements stay separate: read :meth:`profile_stats` for aggregates
        and :meth:`profile` for individual call records.
        """
        with self._lock:
            self._known(name)
            info: ToolInspection = {**self.definition(name)}
            if name in self._sources:
                info["source"] = self.source(name)
            return info

    def profile(self, name: str | None = None, *, offset: int = 0, limit: int = 100) -> JITHarnessProfile:
        """Read aggregate counters and a page of finished, payload-free calls.

        All managed dispatch attempts through harness wrappers are counted, including
        nested calls, validation failures, and cancellations. Middleware and
        lifecycle hooks run outside this measured boundary. Histories remain
        in this harness across requests/updates/unload, bounded by call_capacity,
        but are not restored from executable snapshots. Base calls are excluded when
        manage_base_tools=False. Export with profile().model_dump_json().
        """
        return self._profiler.report(name, offset, limit)

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
        This read never triggers retries, source updates, or snapshot writes.
        """
        return self._profiler.errors(name, offset, limit)

    def snapshot(self) -> JITHarnessSnapshot:
        """Copy Python/text definitions and unload state, not runtime globals."""
        with self._lock:
            tools = tuple(
                record.model_copy(deep=True)
                for _, record in sorted(self._sources.items())
                if isinstance(record, PythonToolSource)
            )
            return JITHarnessSnapshot(
                # Only a package needs the newer reader; single-file snapshots stay
                # byte-compatible with earlier releases.
                format_version=2 if any(record.files for record in tools) else 1,
                tools=tools,
                text_tools=tuple(
                    record.model_copy(deep=True)
                    for _, record in sorted(self._sources.items())
                    if isinstance(record, TextToolSource)
                ),
                base_tools=tuple(sorted(self._base_names)),
                unloaded_tools=tuple(sorted(self._unloaded)),
            )

    def save(self, path: str | Path) -> None:
        """Export the current snapshot to an explicit, atomically replaced file."""
        self.snapshot().save(path)

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
