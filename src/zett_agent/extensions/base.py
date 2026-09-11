"""Lifecycle hook Mixins and the aggregate AgentExtension contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..events import AgentEvent, AgentEventType

if TYPE_CHECKING:
    from ..agent import AgentConfig, AgentContext
    from ..messages import AssistantMessage, ToolCall, ToolMessage
    from ..model import ModelRequest, ModelResponse
    from .events import ExtensionEvent
    from .external import ExternalEvent


class AgentSetupHooksMixin:
    """Hooks that prepare request-scoped tools and conversation context."""

    async def on_tool(self, context: AgentContext) -> None:
        """Register tools before any extension restores or injects messages.

        All extensions finish this hook before the first on_state() call, so
        prompt extensions can reliably inspect the complete request tool set.

        Examples:
            Usage::

                async def on_tool(self, context):
                    context.register_tool(read_file)
        """

    async def on_state(self, context: AgentContext) -> None:
        """Restore history and initialize state before any input conversion.

        Examples:
            Usage::

                async def on_state(self, context):
                    context.state.messages.insert(0, SystemMessage(content="Use concise answers."))
        """

    async def on_message(self, context: AgentContext) -> None:
        """Transform context.input_message after state restoration, before append.

        Replace the pending UserMessage to implement commands such as /goal.
        Do not append or persist it here: the runtime appends the final message
        exactly once after all input hooks finish. before_run sees that message
        in state.messages and persistence subscribers have already received it.
        """


class AgentRunHooksMixin:
    """Hooks around one complete request and its terminal outcome."""

    async def before_run(self, context: AgentContext) -> None:
        """Run after the transformed user input is appended and published."""

    async def after_run(self, context: AgentContext, result: AssistantMessage) -> None:
        """Run after a successful final answer is appended."""

    async def on_success(self, context: AgentContext, result: AssistantMessage) -> None:
        """Run once after all after_run hooks succeed, before RUN_COMPLETED.

        This callback observes a completed request, not an intermediate model
        step. Failed or cancelled requests do not trigger it. Callback errors
        propagate through on_error and prevent RUN_COMPLETED from being emitted.

        Examples:
            Usage::

                async def on_success(self, context, result):
                    await save_answer(context.config.session_id, result.content)
        """

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Run before an agent error is propagated to the caller."""


class AgentModelHooksMixin:
    """Hooks immediately before and after each primary model invocation."""

    async def before_model(self, context: AgentContext, request: ModelRequest) -> None:
        """Inspect the current request before preprocessing a primary model call.

        Request fields are frozen. Change messages through context.state.messages
        and tools through context.tools; the runtime rebuilds these fields before
        the next hook and before calling the provider. This is a shallow request
        view, not an immutable copy of each message or tool definition.

        Examples:
            Inspect the tools offered in this iteration::

                async def before_model(self, context, request):
                    offered = {definition.name for definition in request.tools}
                    if "write_file" in offered:
                        context.state.messages.append(
                            SystemMessage(content="Read existing files before editing.")
                        )
        """

    async def after_model(self, context: AgentContext, response: ModelResponse) -> None:
        """Run after the complete assistant message is appended."""


class AgentToolHooksMixin:
    """Hooks immediately before and after each requested tool invocation."""

    async def before_tool(self, context: AgentContext, call: ToolCall) -> None:
        """Run before one requested tool is invoked."""

    async def after_tool(
        self,
        context: AgentContext,
        call: ToolCall,
        result: ToolMessage,
        error: Exception | None,
    ) -> None:
        """Inspect or modify a tool result before it is appended.

        ``error`` is the original execution exception for a failed result and
        None after successful execution. Changes made to ``result`` are included
        in context, persistence, and the next model request. Raising prevents the
        result from being appended.
        """


class AgentEventHooksMixin:
    """Hooks for streaming, internal notifications, and external input."""

    def accept(self, config: AgentConfig | None, event: ExternalEvent) -> bool:
        """Handle one external event and report whether it was accepted.

        The Agent broadcasts each event to every registered extension. Override
        this synchronous hook when an extension waits for input from another
        thread, HTTP request, TUI, or Web UI. The default ignores the event.
        Config is passed separately and can be None before initialization.
        Each extension decides whether the supplied identity and payload match
        its protocol. Return True to include this extension's name in the
        broadcast result; returning False does not stop later receivers.
        """
        return False

    async def before_model_events(self, context: AgentContext, request: ModelRequest) -> AsyncIterator[AgentEvent]:
        """Stream extension-owned events before a primary model request.

        This hook is intended for visible preprocessing operations such as
        context compaction. CUSTOM events do not change the request phase.
        Request contains the latest messages, tool definitions, and reasoning
        effort at hook entry, including changes made by earlier hooks. After
        changing context, use context itself for the updated values; the supplied
        request is not a live view. The next hook receives a rebuilt request.

        Examples:
            Usage::

                yield AgentEvent(
                    AgentEventType.CUSTOM,
                    session_id=context.config.session_id,
                    name="retrieval_progress",
                    payload={"completed": 3, "total": 10},
                )
        """
        if False:
            yield AgentEvent(AgentEventType.MODEL_STARTED, context.config.session_id)

    async def after_model_events(self, context: AgentContext, response: ModelResponse) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after after_model and MODEL_COMPLETED.

        The response is already appended. Hooks run in priority order in READY,
        before tool dispatch or final-answer handling. Errors fail the request;
        closing the stream closes the hook iterator and runs its finally block.
        """
        if False:
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id)

    async def after_tool_events(
        self,
        context: AgentContext,
        call: ToolCall,
        result: ToolMessage,
        error: Exception | None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after after_tool and TOOL_COMPLETED/TOOL_FAILED.

        ``error`` is the same original execution exception passed to after_tool,
        or None after success. Skipped or cancelled tools do not call this hook.
        Events are emitted before steering selection or the next tool; execution
        remains in READY.
        """
        if False:
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id)

    async def before_tool_events(self, context: AgentContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after before_tool and before each tool starts.

        Hooks run in priority order while the request remains READY. Equal
        priorities retain registration order. Hook failures abort the request
        before tool execution. Closing the consumer closes this iterator so its
        finally blocks can release resources.

        Examples:
            Usage::

                yield AgentEvent(
                    AgentEventType.CUSTOM,
                    session_id=context.config.session_id,
                    name="tool_preparation",
                    payload={"tool_call_id": call.id},
                )
        """
        if False:
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id, name="example")

    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        """Process a published notification; inspect its concrete type with match."""


class AgentExtension(
    AgentSetupHooksMixin,
    AgentRunHooksMixin,
    AgentModelHooksMixin,
    AgentToolHooksMixin,
    AgentEventHooksMixin,
):
    """Combine all optional hooks for the model-tool request lifecycle.

    .. note::
        Override only the hooks needed by one concern. Hook groups are barriers:
        every ``on_tool`` completes before any ``on_state`` starts, regardless
        of extension priority. Lifecycle hooks use ``async def`` even with
        SyncAgent; they receive the original AgentContext on the runtime loop.
        External-event ``accept`` remains a synchronous method.

    .. seealso::
        :doc:`/extending/first-extension` builds a complete extension,
        :doc:`/extending/hooks` lists every hook boundary, and
        :doc:`/extending/state` covers concurrency and cleanup.

    ``priority`` controls the order in which the Agent invokes extensions.
    Lower numbers run first; extensions with the same priority retain their
    registration order. Override the class attribute for one extension type or
    assign it on an instance when one registration needs a different order::

        class RestoreHistory(AgentExtension):
            priority = 10

        metrics = MetricsExtension()
        metrics.priority = 200

    ``[E]`` marks an internal event published to every extension.

    .. zett-diagram:: extension-lifecycle

        +-----------------------------------------+
        | NEW REQUEST                             |
        +-----------------------------------------+
                             |
                             v
        +-----------------------------------------+
        | SETUP                                   |
        | on_tool() -> on_state()                 |
        | on_message() -> append UserMessage      |
        | before_run()                            |
        +-----------------------------------------+
                             |
                             v
        +-----------------------------------------+
        | PRE-MODEL                               |<------------------------------------------------------+
        | before_model()                          |                                                       |
        | before_model_events() / compaction      |                                                       |
        +-----------------------------------------+                                                       |
                             |                                                                            |
                             v                                                                            |
        +-----------------------------------------+                                                       |
        | MODEL STEP                              |                                                       |
        | append AssistantMessage                 |                                                       |
        | after_model() / after_model_events()    |                                                       |
        +-----------------------------------------+                                                       |
                             |                                                                            |
                             v                                                                            |
        +-----------------------------------------+           +-------------------------------------+     |
        | HAS TOOL CALLS?                         |-- yes --->| TOOL STEP                           |     |
        +-----------------------------------------+           | before_tool()                       |     |
                             |                                | before_tool_events()                |     |
                             |                                | execute -> after_tool()             |-----+
                             |                                | append ToolMessage / TOOL_*         |     |
                             | no                             | after_tool_events()                 |     |
                             |                                +-------------------------------------+     |
                             |                                                                            |
                             |                                                                            |
                             v                                                                            |
        +-----------------------------------------+           +-------------------------------------+     |
        | STEERING OR INTERNAL INPUT?             |-- yes --->| SELECT INPUT                        |     |
        +-----------------------------------------+           | append typed input                  |-----+
                             |                                | reset iteration budget              |
                             |                                +-------------------------------------+
                             | empty
                             |
                             |
                             v
        +-----------------------------------------+
        | SUCCESS                                 |
        | after_run() -> on_success()             |
        | COMPLETED [E] -> RUN_COMPLETED          |
        +-----------------------------------------+



        +-----------------------------+                       +-------------------------------------+
        | ANY ACTIVE STAGE            |-- exception --------->| FAILED [E]                          |
        +-----------------------------+                       | on_error() -> re-raise              |
                       |                                      +-------------------------------------+
                       |
                       |
                       |                                      +-------------------------------------+
                       +- cancellation ---------------------->| CANCELLED [E]                       |
                                                              | RunCancelledEvent -> re-raise       |
                                                              +-------------------------------------+



        +-----------------------------+                       +-------------------------------------+
        | ExternalEvent               |-- emit_external_event | accept() on each extension          |
        +-----------------------------+                       +-------------------------------------+

    All on_tool() hooks finish before on_state() restores history and system
    instructions. All on_state() hooks finish before on_message() transforms
    context.input_message. The runtime then appends and publishes the transformed
    UserMessage exactly once, followed by before_run(). Each hook
    runs by ascending priority; equal priorities retain registration order. The
    right return line runs after every tool in the response has been processed,
    or after one AgentMessage is appended with a fresh iteration budget.
    Internal messages have a per-request count limit. Their processing starts
    after the current model/tool loop completes. SteeringExtension is checked
    after MODEL_COMPLETED and every TOOL_COMPLETED/TOOL_FAILED, in READY. Urgent
    user input takes priority over internal input and the remaining tool calls.
    Those calls receive skipped ToolMessages [E] and TOOL_SKIPPED events before
    the steering UserMessage [E] and STEERING_STARTED event. The loop then returns
    to MODEL STEP with a fresh iteration budget. An active internal/steering input
    emits ``*_INTERRUPTED`` when superseded, or ``*_COMPLETED`` after a final answer.

    before_model_events() can stream CUSTOM or compaction events; compaction
    enters COMPACTING and returns to READY. before_tool_events() streams CUSTOM
    events in READY. These AgentEvents reach the caller; [E] marks an awaited,
    sequential broadcast through the separate async on_event() hook, not a
    synchronous Python callback.

    Tool execution errors become failed ToolMessages and still run after_tool(),
    which receives the original exception separately and can modify the message
    before it is appended and published.
    Unhandled model or hook errors take the exception exit. Cancellation skips
    on_error(), after_run(), and on_success(). Errors raised by subscribers after
    a terminal phase was committed do not change that terminal phase.

    Setup, run, model, tool, and event hooks are supplied by their corresponding
    Mixins. Subclasses override only the hooks they need; defaults are no-ops.
    External events are independent from a request's internal [E] notifications:
    callers emit them through the Agent instead of addressing an extension.
    """

    priority: int = 100

    @property
    def name(self) -> str:
        """Unique registration identity, defaulting to the concrete class name.

        Override with a class attribute or assign an instance name to register
        independently configured instances of the same extension type.
        """
        return getattr(self, "_extension_name", type(self).__name__)

    @name.setter
    def name(self, value: str) -> None:
        self._extension_name = value
