"""Lifecycle hook Mixins and the aggregate AgentExtension contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..events import AgentEvent, AgentEventType

if TYPE_CHECKING:
    from ..agent import AgentConfig, AgentContext
    from ..messages import AssistantMessage, ToolCall, ToolMessage
    from ..model import ModelResponse
    from .events import ExtensionEvent
    from .external import ExternalEvent


class AgentSetupHooksMixin:
    """Hooks that prepare request-scoped tools and conversation context."""

    async def on_tool(self, context: AgentContext) -> None:
        """Register tools before any extension restores or injects messages.

        All extensions finish this hook before the first on_message() call, so
        prompt extensions can reliably inspect the complete request tool set.

        Example:
            async def on_tool(self, context):
                context.register_tool(read_file)
        """

    async def on_message(self, context: AgentContext) -> None:
        """Fill a fresh state before every request's user input is appended.

        Example:
            async def on_message(self, context):
                context.state.messages.insert(0, SystemMessage(content="Use concise answers."))
        """


class AgentRunHooksMixin:
    """Hooks around one complete request and its terminal outcome."""

    async def before_run(self, context: AgentContext) -> None:
        """Run before the new user message is appended."""

    async def after_run(self, context: AgentContext, result: AssistantMessage) -> None:
        """Run after a successful final answer is appended."""

    async def on_success(self, context: AgentContext, result: AssistantMessage) -> None:
        """Run once after all after_run hooks succeed, before RUN_COMPLETED.

        This callback observes a completed request, not an intermediate model
        step. Failed or cancelled requests do not trigger it. Callback errors
        propagate through on_error and prevent RUN_COMPLETED from being emitted.

        Example:
            async def on_success(self, context, result):
                await save_answer(context.config.session_id, result.content)
        """

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Run before an agent error is propagated to the caller."""


class AgentModelHooksMixin:
    """Hooks immediately before and after each primary model invocation."""

    async def before_model(self, context: AgentContext) -> None:
        """Run immediately before each model request is assembled."""

    async def after_model(self, context: AgentContext, response: ModelResponse) -> None:
        """Run after the complete assistant message is appended."""


class AgentToolHooksMixin:
    """Hooks immediately before and after each requested tool invocation."""

    async def before_tool(self, context: AgentContext, call: ToolCall) -> None:
        """Run before one requested tool is invoked."""

    async def after_tool(self, context: AgentContext, call: ToolCall, result: ToolMessage) -> None:
        """Run after one tool result is appended, including failed results."""


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

    async def before_model_events(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Stream extension-owned events before a primary model request.

        This hook is intended for visible preprocessing operations such as
        context compaction. CUSTOM events do not change the request phase.

        Example:
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
        self, context: AgentContext, call: ToolCall, result: ToolMessage
    ) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after after_tool and TOOL_COMPLETED/TOOL_FAILED.

        Inspect result.success to distinguish success from a reported tool error.
        Skipped or cancelled tools do not call this hook. Events are emitted
        before steering selection or the next tool; execution remains in READY.
        """
        if False:
            yield AgentEvent(AgentEventType.CUSTOM, context.config.session_id)

    async def before_tool_events(self, context: AgentContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after before_tool and before each tool starts.

        Hooks run in priority order while the request remains READY. Equal
        priorities retain registration order. Hook failures abort the request
        before tool execution. Closing the consumer closes this iterator so its
        finally blocks can release resources.

        Example:
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

    ``priority`` controls the order in which the Agent invokes extensions.
    Lower numbers run first; extensions with the same priority retain their
    registration order. Override the class attribute for one extension type or
    assign it on an instance when one registration needs a different order::

        class RestoreHistory(AgentExtension):
            priority = 10

        metrics = MetricsExtension()
        metrics.priority = 200

    Complete lifecycle; read each box from top to bottom. The scope branch
    applies to every active operation, including hooks and event subscribers::

              +----------------------------------------------+         +----------------------------------+
              | NEW REQUEST: fresh state and tools           |--scope->| ANY ACTIVE STAGE                 |
              +----------------------------------------------+         +----------------------------------+
                                      |                                                 |
                                      |                                                 |
                                      v                                                 v
              +----------------------------------------------+         +----------------------------------+
              | SETUP                                        |         | Exception:                       |
              | LOADING_CONTEXT [E]                          |         |   FAILED [E]                     |
              | on_tool()                                    |         |   on_error()                     |
              | on_message()                                 |         |   re-raise error                 |
              | READY [E]                                    |         |                                  |
              | before_run()                                 |         | Cancellation:                    |
              | append UserMessage [E]                       |         |   CANCELLED [E]                  |
              +----------------------------------------------+         |   RunCancelledEvent [E]          |
                                      |                                |   re-raise cancellation          |
                                      |                                +----------------------------------+
                                      v
              +----------------------------------------------+
         +--->| MODEL STEP                                   |
         |    | before_model()                               |
         |    | before_model_events()                        |
         |    |   optional compaction [E]                    |
         |    | GENERATING [E]                               |
         |    | stream output / timing [E]                   |
         |    | READY [E]                                    |
         |    | append AssistantMessage [E]                  |
         |    | after_model()                                |
         |    | emit MODEL_COMPLETED                         |
         |    | after_model_events(); check steering          |
         |    +----------------------------------------------+
         |                            |
         |                            |
         |                            v
         |    +----------------------------------------------+         +----------------------------------------------+
         |    | HAS TOOL CALLS?                              |-- no -->| FINAL ANSWER                                 |
         |    +----------------------------------------------+         | complete active internal/user input          |
         |                          | yes                              | emit its *_COMPLETED event                   |
         |                          v                                  +----------------------+-----------------------+
         |    +----------------------------------------------+                                |
         |    | FOR EACH TOOL CALL                           |                                v
         |    | before_tool()                                |         +----------------------------------------------+
         |    | before_tool_events()                         |         | TAKE STEERING FIRST, THEN INTERNAL           |
         |    | RUNNING_TOOL [E]                             |         +----------------------+-----------------------+
         |    | execute tool                                 |                                |
         |    | READY [E]                                    |                +---------------+---------------+
         |    | append ToolMessage [E]                       |                | available                     | empty
         |    | after_tool()                                 |                v                               v
         |    | emit TOOL_*                                  |         +----------------------+  +--------------------+
         |    | after_tool_events(); check steering           |         |                      |  |                    |
         |    +----------------------------------------------+         | emit *_STARTED       |  | SUCCESS            |
         |                          |                                  | append typed input   |  | close inboxes      |
         |     all tools done       v                                  | reset budget         |  | after_run()        |
         +--------------------------+                                  +----------+-----------+  | on_success()       |
         |                                                                        |              | COMPLETED [E]      |
         |                                                                        |              | RUN_COMPLETED     |
         |                                                                        |              +--------------------+
         |                                                                        |
         +------------------------ next model step -------------------------------+

              +-------------------------------------------------------------------------------------------+
              | [E] context.publish(event) -> on_event() for every extension, in priority order.          |
              | Phase changes and message appends publish events; output timing publishes boundaries.     |
              +-------------------------------------------------------------------------------------------+

              +-------------------------------------------------------------------------------------------+
              | ExternalEvent -> Agent.emit_external_event() -> accept() on every registered extension.   |
              +-------------------------------------------------------------------------------------------+

    All on_tool() hooks finish before any on_message() hook starts. Each hook
    runs by ascending priority; equal priorities retain registration order. The
    left return line runs after every tool in the response has been processed,
    or after one AgentMessage is appended with a fresh iteration budget.
    Internal messages have a per-request count limit. Their processing starts
    after the current model/tool loop completes. SteeringExtension is checked
    after MODEL_COMPLETED and every TOOL_COMPLETED/TOOL_FAILED, in READY. Urgent
    user input takes priority over internal input and the remaining tool calls.
    Those calls receive skipped ToolMessages [E] and TOOL_SKIPPED events before
    the steering UserMessage [E] and STEERING_STARTED event. The loop then returns
    to MODEL STEP with a fresh iteration budget. An active internal/steering input
    emits *_INTERRUPTED when superseded, or *_COMPLETED after a final answer.

    before_model_events() can stream CUSTOM or compaction events; compaction
    enters COMPACTING and returns to READY. before_tool_events() streams CUSTOM
    events in READY. These AgentEvents reach the caller; [E] marks synchronous
    notification through the separate on_event() hook.

    Tool execution errors become failed ToolMessages and still run after_tool().
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
