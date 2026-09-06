"""Lifecycle hook Mixins and the aggregate AgentExtension contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .events import AgentEvent, AgentEventType

if TYPE_CHECKING:
    from .agent import AgentContext
    from .extension_events import ExtensionEvent
    from .external_events import ExternalEvent
    from .messages import AssistantMessage, ToolCall, ToolMessage
    from .model import ModelResponse


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

    def accept(self, event: ExternalEvent) -> bool:
        """Handle one external event and report whether it was accepted.

        The Agent broadcasts each event to every registered extension. Override
        this synchronous hook when an extension waits for input from another
        thread, HTTP request, TUI, or Web UI. The default ignores the event.
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

    async def before_tool_events(self, context: AgentContext, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Stream CUSTOM events after before_tool and before each tool starts.

        Hooks run in registration order while the request remains READY. Hook
        failures abort the request before tool execution. Closing the consumer
        closes this iterator so its finally blocks can release resources.

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
         |    +----------------------------------------------+
         |                            |
         |                            |
         |                            v
         |    +----------------------------------------------+         +----------------------------------+
         |    | HAS TOOL CALLS?                              |-- no -->| SUCCESS                          |
         |    +----------------------------------------------+         | after_run()                      |
         |                            | yes                            | on_success()                     |
         |                            |                                | COMPLETED [E]                    |
         |                            v                                | emit RUN_COMPLETED               |
         |    +----------------------------------------------+         +----------------------------------+
         |    | FOR EACH TOOL CALL                           |
         |    | before_tool()                                |
         |    | before_tool_events()                         |
         |    | RUNNING_TOOL [E]                             |
         |    | execute tool                                 |
         |    | READY [E]                                    |
         |    | append ToolMessage [E]                       |
         |    | after_tool()                                 |
         |    | emit TOOL_COMPLETED / TOOL_FAILED            |
         |    +----------------------------------------------+
         |                            |
         |    all tools done          v
         +----------------------------+


              +-------------------------------------------------------------------------------------------+
              | [E] context.publish(event) -> on_event() for every extension, in registration order.      |
              | Phase changes and message appends publish events; output timing publishes boundaries.     |
              +-------------------------------------------------------------------------------------------+

              +-------------------------------------------------------------------------------------------+
              | ExternalEvent -> Agent.emit_external_event() -> accept() on every registered extension.   |
              +-------------------------------------------------------------------------------------------+

    All on_tool() hooks finish before any on_message() hook starts. Each hook
    runs in extension registration order. The left return line runs only after
    every tool in the model response has been processed.

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
