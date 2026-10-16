"""Expose session-local JIT tool evolution through ordinary function calling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from importlib.resources import files
from threading import RLock
from typing import Literal

from .._compat import aclosing_stream
from ..agent import AgentRunContext
from ..events import AgentEvent, AgentEventType, AgentPhase
from ..extensions.base import AgentExtension, ModelRequestNext, ToolCallNext
from ..extensions.events import ExtensionEvent, PhaseTransitionEvent, RunCancelledEvent
from ..messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage
from ..model import ModelEvent, ModelRequest
from ..tools.base import AgentTool, ToolExecutionMode, ToolResult, render_tool_guidance, tool
from .enums import InspectionToolExposure, ToolKind, ToolOrigin
from .profiling import MAX_RETAINED_CALLS, ToolProfiler, ToolProfileStats
from .queries import ToolQueries
from .results import (
    ToolCallsPage,
    ToolDefinitionView,
    ToolErrorsPage,
    ToolListPage,
    ToolMutationResult,
    ToolTextPage,
)
from .runtime import JITHarness
from .snapshot import JITHarnessSnapshot

JIT_TOOL_STARTED_EVENT_NAME = "jit_tool_started"
JIT_TOOL_COMPLETED_EVENT_NAME = "jit_tool_completed"

_GUIDANCE_KEY = "jit_harness:guidance"
#: Origins announced as CUSTOM events. Base tools are skipped because the runtime
#: already publishes TOOL_STARTED/TOOL_COMPLETED when the model requests them.
_OBSERVED_ORIGINS = (ToolOrigin.MANAGEMENT, ToolOrigin.PYTHON, ToolOrigin.TEXT)
_INSTRUCTIONS = files("zett_agent.jit_harness").joinpath("instructions.md").read_text(encoding="utf-8")
_INSPECTION_INSTRUCTIONS = files("zett_agent.jit_harness").joinpath("inspection.md").read_text(encoding="utf-8")
_DEFERRED_INSPECTION_NOTE = (
    "The provider may defer loading these inspection definitions: search for one by name when you need it."
)
_HIDDEN_INSPECTION_NOTE = (
    "Inspection tools are not exposed in this session: do not plan to read source, schemas, statistics, or error "
    "history; ask the application instead."
)

#: Management tools whose answers a session often does not need. They are the
#: tier InspectionToolExposure controls; the rest of the control loop stays
#: exposed so the model can always build, change, and save capabilities.
_INSPECTION_TOOL_NAMES = frozenset(
    {"read_tool_source", "read_tool_schema", "inspect_tool_stats", "inspect_tool_calls", "inspect_tool_errors"}
)


class JITToolStartedEvent(AgentEvent):
    """CUSTOM event announcing one management or dynamic tool call.

    Emitted before the handler runs, for model-requested calls and for calls a
    dynamic tool composes through the harness. ``payload["tool_call_id"]`` matches
    both the conversation's ToolCall and the matching record from
    :meth:`~zett_agent.jit_harness.runtime.JITHarness.profile`.
    """

    __slots__ = ()

    def __init__(self, session_id: str, call: ToolCall, origin: ToolOrigin) -> None:
        super().__init__(
            AgentEventType.CUSTOM,
            session_id=session_id,
            tool_calls=[call],
            name=JIT_TOOL_STARTED_EVENT_NAME,
            payload={
                "session_id": session_id,
                "tool_call_id": call.id,
                "tool_name": call.name,
                "origin": origin.value,
            },
        )


class JITToolCompletedEvent(AgentEvent):
    """CUSTOM event reporting how one management or dynamic tool call ended.

    ``success`` mirrors the ``ToolMessage`` and ``error_type`` names the exception
    class when the call failed. Arguments and results are not repeated here; read
    the matching ``profile()`` record, joined on ``tool_call_id``, for sizes,
    timing, and caller.
    """

    __slots__ = ()

    def __init__(
        self,
        session_id: str,
        call: ToolCall,
        origin: ToolOrigin,
        *,
        success: bool,
        error: Exception | None,
    ) -> None:
        super().__init__(
            AgentEventType.CUSTOM,
            session_id=session_id,
            tool_calls=[call],
            name=JIT_TOOL_COMPLETED_EVENT_NAME,
            payload={
                "session_id": session_id,
                "tool_call_id": call.id,
                "tool_name": call.name,
                "origin": origin.value,
                "success": success,
                "error_type": type(error).__name__ if error is not None else None,
            },
        )


class JITHarnessExtension(AgentExtension):
    """Let a model evolve independent Python tools and text workflows per session.

    Args:
        snapshot: Initial template, copied independently into every new session.
        manage_base_tools: Enable base-tool unload/load and profiling. False
            protects application base tools and leaves their handlers unchanged.
        max_call_depth: Maximum nested calls, including recursive composition.
        error_capacity: Maximum distinct recent errors retained per session.
        error_message_chars: Maximum characters retained per error message.
        call_capacity: Maximum finished call records retained per session;
            oldest records are dropped first and lifetime aggregates stay complete.
        inspection_tools: How the model sees the inspection tools that read
            source, schemas, statistics, call records, and recent errors.

    Note:
        Opt in only for trusted generated Python. The control tools are ordinary
        functions on any protocol with function calling. Inspection tools can be
        deferred, so a provider that supports deferred loading fetches them on
        demand, while protocols that ignore that flag still receive every
        definition. Application tools added later in setup or before-model hooks
        are adopted before dispatch; existing policy removals are not restored.

    Examples:
        Enable evolution and explicitly save the resulting toolbox::

            config = AgentRunConfig(session_id="my-session")
            jit = JITHarnessExtension()
            client = await create_agent(model, config=config, tools=[read_file], extensions=[history, jit])
            await client.run("Build a reusable file preview tool.")
            jit.get_harness("my-session").save("harness.json")
    """

    priority = 900_000

    def __init__(
        self,
        snapshot: JITHarnessSnapshot | None = None,
        *,
        manage_base_tools: bool = True,
        max_call_depth: int = 32,
        error_capacity: int = 100,
        error_message_chars: int = 4000,
        call_capacity: int = MAX_RETAINED_CALLS,
        inspection_tools: InspectionToolExposure = InspectionToolExposure.DEFERRED,
    ) -> None:
        if max_call_depth < 1:
            raise ValueError("max_call_depth must be positive")
        if not isinstance(manage_base_tools, bool):
            raise ValueError("manage_base_tools must be a boolean")
        self._template = (snapshot or JITHarnessSnapshot()).model_copy(deep=True)
        self._manage_base_tools = manage_base_tools
        self._max_call_depth = max_call_depth
        # Accept the enum or its value so applications can pass either.
        self._inspection_tools = InspectionToolExposure(inspection_tools)
        # Validate every retention bound before the first session request.
        ToolProfiler(
            error_capacity=error_capacity, error_message_chars=error_message_chars, call_capacity=call_capacity
        )
        ToolProfiler(
            error_capacity=error_capacity, error_message_chars=error_message_chars, call_capacity=call_capacity
        )
        self._error_capacity = error_capacity
        self._error_message_chars = error_message_chars
        self._call_capacity = call_capacity
        self._sessions: dict[str, JITHarness] = {}
        self._lock = RLock()

    def get_harness(self, session_id: str) -> JITHarness:
        """Return an existing session's toolbox for explicit inspection or saving."""
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(f"No JIT harness for session {session_id!r}") from None

    def forget_session(self, session_id: str) -> None:
        """Release an idle in-memory toolbox; this does not delete saved files."""
        with self._lock:
            harness = self.get_harness(session_id)
            if harness.active:
                raise RuntimeError("Cannot forget a harness with an active request")
            del self._sessions[session_id]

    async def on_tool(self, context: AgentRunContext) -> None:
        """Rebind current base tools, mount wrappers, and remove unloaded tools."""
        session_id = context.session_id
        with self._lock:
            harness = self._sessions.get(session_id)
            if harness is None:
                harness = JITHarness(
                    self._template,
                    manage_base_tools=self._manage_base_tools,
                    max_call_depth=self._max_call_depth,
                    error_capacity=self._error_capacity,
                    error_message_chars=self._error_message_chars,
                    call_capacity=self._call_capacity,
                )
                self._sessions[session_id] = harness
            harness.attach(context, self._management_tools(harness))

    async def on_state(self, context: AgentRunContext) -> None:
        """Adopt tools registered by later setup hooks before context loading."""
        self.get_harness(context.session_id).sync_tools(context)

    async def on_tool_call(self, context: AgentRunContext, call: ToolCall, call_next: ToolCallNext) -> ToolResult:
        """Adopt late before-tool registrations before handler middleware runs."""
        self.get_harness(context.session_id).sync_tools(context)
        return await call_next()

    async def before_tool(self, context: AgentRunContext, call: ToolCall) -> None:
        """Announce a management or dynamic tool call before its handler runs.

        The event carries identity and classification only. Calls to application
        base tools emit nothing, and a name this harness has never seen is ignored
        instead of being reported as a JIT call.
        """
        origin = self._observed_origin(context, call.name)
        if origin is not None:
            await context.emit(JITToolStartedEvent(context.session_id, call, origin))

    async def after_tool(
        self,
        context: AgentRunContext,
        call: ToolCall,
        result: ToolMessage,
        error: Exception | None,
    ) -> None:
        """Report the outcome of a management or dynamic tool call."""
        origin = self._observed_origin(context, call.name)
        if origin is not None:
            await context.emit(
                JITToolCompletedEvent(
                    context.session_id,
                    call,
                    origin,
                    success=result.success,
                    error=error,
                )
            )

    def _observed_origin(self, context: AgentRunContext, name: str) -> ToolOrigin | None:
        with self._lock:
            harness = self._sessions.get(context.session_id)
        if harness is None:
            return None
        origin = harness.origin(name)
        return origin if origin in _OBSERVED_ORIGINS else None

    def _management_tools(self, harness: JITHarness) -> tuple[AgentTool, ...]:
        """Build this session's control tools, tiered by the configured exposure."""
        queries = ToolQueries(harness)
        # Only a protocol that renders defer_loading loads a definition on
        # demand; the flag is inert elsewhere, so no capability is lost there.
        deferred = self._inspection_tools is InspectionToolExposure.DEFERRED

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def create_tool(
            source: str,
            entrypoint: str | None = None,
            # A literal keeps the shared enum's documentation out of every request;
            # the argument description already explains both modes.
            kind: Literal["python", "text"] = "python",
            name: str | None = None,
            description: str | None = None,
            files: dict[str, str] | None = None,
        ) -> ToolMutationResult:
            """Create a Python tool or reusable Markdown workflow (kind="text").

            Returns only the name and loaded state.

            Args:
                source: Trusted Python entry point, or Markdown for kind="text".
                entrypoint: Decorated Python variable; omit for text.
                kind: python executes code; text returns instructions to follow.
                name: Required text name; omit for Python.
                description: Required text hint (1–512 characters); omit for Python.
                files: Extra modules by relative path, imported from the entry point.

            Guidelines:
                - Reuse existing capabilities; create only for missing functionality or worthwhile workflow savings.
                - Put repetitive tool work in one tool that calls others, not in many separate calls.
            """
            return harness.create_tool(
                source,
                entrypoint=entrypoint,
                kind=ToolKind(kind),
                name=name,
                description=description,
                files=files,
            )

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def update_tool(
            name: str,
            source: str | None = None,
            entrypoint: str | None = None,
            description: str | None = None,
            files: dict[str, str] | None = None,
            remove: list[str] | None = None,
        ) -> ToolMutationResult:
            """Replace part of a dynamic tool; base tools cannot be replaced.

            Args:
                name: Existing dynamic name; its kind is preserved.
                source: Replacement entry point; omit to keep it.
                entrypoint: Entry point variable; omit to keep it.
                description: Optional new text hint; omit for Python.
                files: Modules to add or replace, by relative path.
                remove: Package file paths to delete.

            Guidelines:
                - Inspect source before updating; send only the files that changed.
            """
            return harness.update_tool(
                name,
                source,
                entrypoint=entrypoint,
                description=description,
                files=files,
                remove=tuple(remove or ()),
            )

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def unload_tool(name: str) -> ToolMutationResult:
            """Unload a retained tool; base tools require base management.

            Guidelines:
                - Unload tools no longer needed; dependent calls will fail.
            """
            return harness.unload_tool(name)

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def load_tool(name: str) -> ToolMutationResult:
            """Restore a retained tool registration without regenerating source.

            Base tools require base management and a current application binding.

            Guidelines:
                - Load an unloaded dependency before calling it.
            """
            return harness.load_tool(name)

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def list_tools(offset: int = 0, limit: int = 20) -> ToolListPage:
            """Read a bounded catalog page of names, states, and description previews.

            Args:
                offset: Zero-based index in the name-sorted catalog.
                limit: Maximum entries, 1 to 100; pages also respect the output budget.

            Guidelines:
                - Use to find tool names and loaded/unloaded/missing states.
            """
            return queries.list_tools(offset, limit)

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def inspect_tool(name: str) -> ToolDefinitionView:
            """Read a tool's definition, including package files, without measurements.

            Args:
                name: Tool name, including an unloaded tool.

            Guidelines:
                - Inspect parameters before composition; use read_tool_source before updating source.
            """
            return queries.inspect_tool(name)

        @tool(execution_mode=ToolExecutionMode.SERIAL, deferred=deferred)
        async def read_tool_source(
            name: str, path: str | None = None, offset: int = 0, max_chars: int = 2000
        ) -> ToolTextPage:
            """Read an exact fragment of one source file, including unloaded tools.

            Args:
                name: Dynamic Python or text tool name, including unloaded tools.
                path: Package file; default the entry point.
                offset: Character offset in that file's source.
                max_chars: Maximum source characters, 1 to 4000.

            Guidelines:
                - Follow next_offset for more source; compare content_id across pages after updates.
            """
            return queries.read_source(name, path, offset, max_chars)

        @tool(execution_mode=ToolExecutionMode.SERIAL, deferred=deferred)
        async def read_tool_schema(name: str, offset: int = 0, max_chars: int = 2000) -> ToolTextPage:
            """Read an exact JSON Schema fragment when inspect_tool omitted an oversized schema.

            Args:
                name: Tool whose argument schema is available.
                offset: Character offset in the schema JSON.
                max_chars: Maximum schema characters, 1 to 4000.

            Guidelines:
                - Use only when parameters_omitted is true; concatenate pages before parsing JSON.
            """
            return queries.read_schema(name, offset, max_chars)

        @tool(execution_mode=ToolExecutionMode.SERIAL, deferred=deferred)
        async def inspect_tool_stats(name: str) -> ToolProfileStats:
            """Read one tool's lifetime usage and timing statistics.

            Args:
                name: Tool name to measure.

            Guidelines:
                - Use to compare frequency, failure counts, latency, and input/output sizes.
            """
            return queries.stats(name)

        @tool(execution_mode=ToolExecutionMode.SERIAL, deferred=deferred)
        async def inspect_tool_calls(name: str, offset: int = 0, limit: int = 5) -> ToolCallsPage:
            """Read compact single-call measurements without aggregates or source.

            Args:
                name: Tool name to inspect.
                offset: Zero-based offset in finished calls, ordered by dispatch.
                limit: Maximum records, 1 to 100; pages also respect the output budget.

            Guidelines:
                - Read when caller relationships or per-call timing and sizes help; follow next_offset.
            """
            return queries.calls(name, offset, limit)

        @tool(execution_mode=ToolExecutionMode.SERIAL, deferred=deferred)
        async def inspect_tool_errors(
            name: str | None = None, offset: int = 0, limit: int = 3, max_chars: int = 512
        ) -> ToolErrorsPage:
            """Read recent errors with bounded messages and occurrence counts.

            Args:
                name: Optional tool name; omit for all retained errors.
                offset: Zero-based offset within retained matching errors.
                limit: Maximum entries, 1 to 100; pages also respect the output budget.
                max_chars: Maximum message characters per entry, 1 to 4000.

            Guidelines:
                - Inspect recent errors before deciding whether to retry, repair source, or load dependencies.
            """
            return queries.errors(name, offset, limit, max_chars)

        return (
            create_tool,
            update_tool,
            unload_tool,
            load_tool,
            list_tools,
            inspect_tool,
            *(
                ()
                if self._inspection_tools is InspectionToolExposure.HIDDEN
                else (
                    read_tool_source,
                    read_tool_schema,
                    inspect_tool_stats,
                    inspect_tool_calls,
                    inspect_tool_errors,
                )
            ),
        )

    async def on_model_request(
        self, context: AgentRunContext, request: ModelRequest, call_next: ModelRequestNext
    ) -> AsyncIterator[ModelEvent]:
        """Render current guidance afresh, without persisting stale definitions."""
        harness = self.get_harness(context.session_id)
        harness.sync_tools(context)
        # Preserve a caller's request-level tool subset; synchronization must
        # never re-expose capabilities filtered by another policy middleware.
        # A name this harness has never seen belongs to another extension, whose
        # request-scoped definition is not ours to remove.
        definitions = tuple(
            definition
            for definition in request.tools
            if definition.name in context.tools or harness.origin(definition.name) is ToolOrigin.UNKNOWN
        )
        if definitions != request.tools:
            request = replace(request, tools=definitions)
        # A later policy extension may have filtered tools. Respect the actual
        # request rather than revealing every locally registered capability.
        offered = {definition.name for definition in request.tools}
        if "create_tool" not in offered:
            filtered = request
        else:
            guidance = render_tool_guidance([tool for name, tool in context.tools.items() if name in offered])
            messages = tuple(
                message
                for message in request.messages
                if not isinstance(message, SystemMessage)
                or not (message.attributes.get(_GUIDANCE_KEY) or message.attributes.get("tool_guidelines"))
            )
            instructions = SystemMessage(
                content=self._guidance(guidance),
                persist=False,
                attributes={_GUIDANCE_KEY: True},
            )
            filtered = replace(request, messages=(instructions, *messages))
        async with aclosing_stream(call_next(filtered)) as events:
            async for event in events:
                yield event

    def _guidance(self, guidance: str) -> str:
        """Assemble the prompt block for this session's modes, then tool guidance.

        Instructions describe only what the model can actually see: hidden
        inspection tools are never named, and a deferring provider is told that
        their definitions may need a search. Tool guidance stays last so the
        stable instruction prefix remains cacheable between requests.
        """
        parts = [
            _INSTRUCTIONS,
            (
                "Base tool management is enabled: base tools may be unloaded and profiled; only dynamic source can be updated."
                if self._manage_base_tools
                else "Base tool management is disabled: do not unload/load base tools; their calls are not profiled. Dynamic tools may still compose them."
            ),
        ]
        if self._inspection_tools is InspectionToolExposure.HIDDEN:
            parts.append(_HIDDEN_INSPECTION_NOTE)
        else:
            parts.append(_INSPECTION_INSTRUCTIONS)
            if self._inspection_tools is InspectionToolExposure.DEFERRED:
                parts.append(_DEFERRED_INSPECTION_NOTE)
        if guidance:
            parts.append(guidance)
        return "\n".join(parts)

    async def on_success(self, context: AgentRunContext, result: AssistantMessage) -> None:
        """Release request bindings while retaining the session's definitions."""
        self._release(context)

    async def on_error(self, context: AgentRunContext, error: Exception) -> None:
        """Release bindings after setup, model, or tool-hook failures."""
        self._release(context)

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Detach cancelled requests so late worker calls cannot use a new run."""
        if isinstance(event, RunCancelledEvent) or (
            isinstance(event, PhaseTransitionEvent) and event.current_phase is AgentPhase.COMPLETED
        ):
            self._release(context)

    def _release(self, context: AgentRunContext) -> None:
        with self._lock:
            if (harness := self._sessions.get(context.session_id)) is not None:
                harness.detach(context)

    def _request_finished(self, context: AgentRunContext) -> None:
        """Also detach when an earlier extension interrupts terminal hooks."""
        self._release(context)
        super()._request_finished(context)
