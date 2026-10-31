"""Expose a session-local tool repository through ordinary function calling."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from threading import RLock
from time import monotonic

from .._compat import aclosing_stream
from ..agent import AgentRunContext
from ..events import AgentEvent, AgentEventType, AgentPhase
from ..extensions.base import AgentExtension, ModelRequestNext, ToolCallNext
from ..extensions.events import ExtensionEvent, PhaseTransitionEvent, RunCancelledEvent
from ..messages import AnyMessage, AssistantMessage, SystemMessage, ToolCall, ToolMessage
from ..model import ModelEvent, ModelRequest
from ..tools.base import AgentTool, ToolExecutionMode, ToolResult, render_tool_guidance, tool
from .enums import ToolOrigin
from .host import JITToolHost
from .profiling import MAX_RETAINED_CALLS, ToolProfiler
from .repo import session_directory_name
from .results import RepoResult, ToolStatsReport
from .runtime import JITHarness

JIT_TOOL_STARTED_EVENT_NAME = "jit_tool_started"
JIT_TOOL_COMPLETED_EVENT_NAME = "jit_tool_completed"

_GUIDANCE_KEY = "jit_harness:guidance"
#: Origins announced as CUSTOM events. Base tools are skipped because the runtime
#: already publishes TOOL_STARTED/TOOL_COMPLETED when the model requests them.
_OBSERVED_ORIGINS = (ToolOrigin.MANAGEMENT, ToolOrigin.PYTHON)
_INSTRUCTIONS = files("zett_agent.jit_harness").joinpath("instructions.md").read_text(encoding="utf-8")


def _system_section_end(messages: Sequence[AnyMessage]) -> int:
    """Return the index just past the leading run of system instructions.

    Guidance belongs inside that run, after the request's leading system
    instructions and before the first dialogue message. A system message that
    appears later in the history stays where it is: only the leading run is the
    request's system section. The caller separately drops stale guidance wherever
    it sits.
    """
    index = 0
    for message in messages:
        if not isinstance(message, SystemMessage):
            break
        index += 1
    return index


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
    """Let a model keep its own tools and notes in a repository on disk.

    Each session gets a directory ``<root>/<session id>/`` holding one Python
    package, the repository. The model writes into it with ordinary file tools;
    the harness imports it, injects a ``harness`` global into its modules, and
    registers every ``@tool`` it defines and lists its Markdown notes.

    Args:
        root: Directory that holds one subdirectory per session. It is created
            when a session first makes a repository, not before.
        manage_base_tools: Enable base-tool unload/load and profiling. False
            protects application base tools and leaves their handlers unchanged.
        max_call_depth: Maximum nested calls, including recursive composition.
        error_capacity: Maximum distinct recent errors retained per session.
        error_message_chars: Maximum characters retained per error message.
        call_capacity: Maximum finished call records retained per session;
            oldest records are dropped first and lifetime aggregates stay complete.
        host_factory: Builds the place each session's repository code runs, called
            once per session. None imports it into this process. Pass
            ``lambda: SubprocessToolHost(sandbox=...)`` to keep crashes and hangs
            out of the agent and to bound what the code may touch.
        max_sessions: How many sessions may keep a harness in memory at once. A new
            session closes the harness that has gone unused the longest, as long as
            no request is running in it, which is what releases a session's child
            process. Sessions are also released explicitly with
            :meth:`forget_session`. When every other session is mid-request the cap
            is exceeded rather than dropping live work.

    A session's resources live as long as it does:

    .. code-block:: text

        first request for a session
              │
              ▼
        JITHarness ──owns──► JITToolHost ──owns──► child process (subprocess host)
              ▲                                        │
              │  reused on later turns                 │  spawned by repo_load
              │                                        │  killed by a reload, timeout, crash
              └──────── forgotten on forget_session ◄──┘  or eviction by max_sessions

    Nothing closes a session when a run ends: a later turn of the same conversation
    expects its tools and its child process to still be there.

    Note:
        Opt in only for trusted generated Python. The model can only run what it
        can write, so give it file tools that reach ``root``. The three control
        tools are ordinary functions on any protocol with function calling.
        Application tools added later in setup or before-model hooks are adopted
        before dispatch; existing policy removals are not restored.

    Examples:
        Keep each session's tools under the application's data directory::

            config = AgentRunConfig(session_id="my-session")
            jit = JITHarnessExtension(Path.home() / ".myapp" / "jit_harness")
            client = await create_agent(model, config=config, tools=[read_file], extensions=[history, jit])
            await client.run("Build a reusable file preview tool.")
    """

    priority = 900_000

    def __init__(
        self,
        root: str | Path,
        *,
        manage_base_tools: bool = True,
        max_call_depth: int = 32,
        error_capacity: int = 100,
        error_message_chars: int = 4000,
        call_capacity: int = MAX_RETAINED_CALLS,
        host_factory: Callable[[], JITToolHost] | None = None,
        max_sessions: int = 8,
    ) -> None:
        if max_call_depth < 1:
            raise ValueError("max_call_depth must be positive")
        if not isinstance(manage_base_tools, bool):
            raise ValueError("manage_base_tools must be a boolean")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self._root = Path(root).expanduser().absolute()
        self._manage_base_tools = manage_base_tools
        self._max_call_depth = max_call_depth
        # One host per session: a subprocess host owns a process that one session's
        # repository runs in.
        self._host_factory = host_factory
        # Validate every retention bound before the first session request.
        ToolProfiler(
            error_capacity=error_capacity, error_message_chars=error_message_chars, call_capacity=call_capacity
        )
        self._error_capacity = error_capacity
        self._error_message_chars = error_message_chars
        self._call_capacity = call_capacity
        self._sessions: dict[str, JITHarness] = {}
        # When each session was last served, so a full cache can drop the stalest one.
        self._last_used: dict[str, float] = {}
        self._max_sessions = max_sessions
        self._lock = RLock()

    def get_harness(self, session_id: str) -> JITHarness:
        """Return an existing session's toolbox for explicit inspection."""
        with self._lock:
            try:
                harness = self._sessions[session_id]
            except KeyError:
                raise KeyError(f"No JIT harness for session {session_id!r}") from None
            self._last_used[session_id] = monotonic()
            return harness

    def session_directory(self, session_id: str) -> Path:
        """Return the directory that holds one session's repository.

        The path is derived from the session id alone, so an application can
        locate, back up, or delete it without a running session.
        """
        return self._root / session_directory_name(session_id)

    def forget_session(self, session_id: str) -> None:
        """Release an idle in-memory toolbox; this does not delete its files."""
        with self._lock:
            harness = self.get_harness(session_id)
            if harness.active:
                raise RuntimeError("Cannot forget a harness with an active request")
            harness.close()
            del self._sessions[session_id]
            self._last_used.pop(session_id, None)

    def _release_stalest(self, incoming: str) -> None:
        """Make room for one more session by closing the least recently used idle one.

        A harness with a request in flight is never closed, so the cap is soft: when
        every other session is busy the new one is added anyway, because refusing it
        would fail the request that is already running.
        """
        while len(self._sessions) >= self._max_sessions:
            candidates = [
                (self._last_used.get(session, 0.0), session)
                for session, harness in self._sessions.items()
                if session != incoming and not harness.active
            ]
            if not candidates:
                return
            _, stale = min(candidates)
            harness = self._sessions.pop(stale)
            self._last_used.pop(stale, None)
            harness.close()

    async def on_tool(self, context: AgentRunContext) -> None:
        """Rebind current base tools, mount wrappers, and remove unloaded tools."""
        session_id = context.session_id
        with self._lock:
            harness = self._sessions.get(session_id)
            if harness is None:
                self._release_stalest(session_id)
                harness = JITHarness(
                    self.session_directory(session_id),
                    manage_base_tools=self._manage_base_tools,
                    max_call_depth=self._max_call_depth,
                    error_capacity=self._error_capacity,
                    error_message_chars=self._error_message_chars,
                    call_capacity=self._call_capacity,
                    host=self._host_factory() if self._host_factory is not None else None,
                )
                self._sessions[session_id] = harness
            self._last_used[session_id] = monotonic()
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
        """Build this session's four repository tools."""

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def repo_create(repo: str) -> RepoResult:
            """Create this session's tool repository, a Python package on disk, and return its path.

            An existing repository is returned as it is; if it has no MANIFEST, a starter
            one is written. A new one carries a starter MANIFEST too; edit it as the
            repository grows, and add BAN_TOOLS to forbid tool names. Write modules into
            the path with your file tools, then call repo_load.

            Args:
                repo: Directory name, 1–64 letters, digits, underscores, or hyphens.

            Snippet:
                repo_create(repo="kit")

            Guidelines:
                - Reuse existing capabilities; create tools only for missing functionality or worthwhile workflow savings.
                - Keep every reusable tool and note in this one repository.
            """
            return harness.create_repo(repo)

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def repo_load() -> RepoResult:
            """Load the repository and list its tools and notes; calling it again is harmless.

            Not loaded yet: imports every module and registers each tool. Already loaded: imports
            nothing again unless a Python file changed, and just returns the tools and notes. The
            notes are the Markdown files under resources/, listed with kind, description and path;
            open one with your file tools when its description fits the task. A repository whose
            MANIFEST or BAN_TOOLS rules it out is refused with the reasons.

            Snippet:
                repo_load()

            Guidelines:
                - Call after every edit; a failed load keeps the previous version and names the file and line, or the declaration it rejects.
                - Call it first in a resumed session to see which tools and notes already exist.
            """
            return harness.load_repo()

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def repo_unload() -> RepoResult:
            """Hide every tool of the repository; the files stay on disk.

            Snippet:
                repo_unload()

            Guidelines:
                - Unload when the tools are no longer needed; repo_load restores them.
            """
            return harness.unload_repo()

        @tool(execution_mode=ToolExecutionMode.SERIAL)
        async def profile_tools(names: list[str] | None = None) -> ToolStatsReport:
            """Show how tools have done: calls, success rate, time, output size, and recent errors.

            Works for any tool, yours or the application's. A tool never called shows zeros.

            Args:
                names: Up to 10 tool names to inspect, each with its recent errors. Omit it to
                    see the repository's tools and every tool called so far.

            Snippet:
                profile_tools()
                profile_tools(names=["twice"])

            Guidelines:
                - Read it before building or changing a tool, and when deciding what to unload or delete.
                - A low success rate or a repeated error means fix the tool; calls of 0 means it may not be needed.
            """
            return harness.profile_tools(names)

        return (repo_create, repo_load, repo_unload, profile_tools)

    async def on_model_request(
        self, context: AgentRunContext, request: ModelRequest, call_next: ModelRequestNext
    ) -> AsyncIterator[ModelEvent]:
        """Place current guidance after the request's leading system instructions and before dialogue."""
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
        if "repo_create" not in offered:
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
                content=self._guidance(harness, guidance),
                persist=False,
                attributes={_GUIDANCE_KEY: True},
            )
            # Close the leading system run before the dialogue. The application's
            # own instructions stay first, and the block stays out of the
            # persisted history.
            index = _system_section_end(messages)
            filtered = replace(request, messages=(*messages[:index], instructions, *messages[index:]))
        async with aclosing_stream(call_next(filtered)) as events:
            async for event in events:
                yield event

    def _guidance(self, harness: JITHarness, guidance: str) -> str:
        """Assemble the prompt block: instructions, this session's repository, tool guidance.

        The repository line tells a resumed session where its tools already live,
        so the model need not rediscover the path. The block follows the request's
        leading system instructions, and tool guidance stays last inside the
        block, so providers that join system texts in order keep the stable
        instructions at the front of the joined prompt.
        """
        parts = [_INSTRUCTIONS]
        if (path := harness.repo_path) is not None:
            parts.append(f"Your repository is `{harness.repo}` at `{path}`. `repo_load` lists its tools and notes.")
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
