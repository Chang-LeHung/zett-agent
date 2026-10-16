"""What the mixins of :class:`~zett_agent.agent.Agent` expect of each other, declared once for the type checker.

Each mixin implements one concern and calls into the others, and reads state that ``Agent.__init__``
assigns. A mixin cannot see those from where it is written, so every one of them would have to
re-declare what it borrows. They inherit this class instead.

Everything below exists only for the type checker. At runtime the class is empty, so it adds nothing to
``Agent``, takes no part in method resolution, and a method written in a mixin is always the one that
runs. Where a mixin implements something declared here, the checker compares the two signatures, so a
declaration that drifts from its implementation is reported instead of silently believed.

Read the docstrings here as the *meaning* of each seam: what a caller may rely on. How a method does its
work is written where it is implemented, in the mixin named in the section heading.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from ..events import AgentPhaseTransitionMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from threading import RLock

    from ..agent import AgentRunConfig, AgentRunContext, AgentState
    from ..extensions.base import AgentExtension
    from ..extensions.internal_message import InternalMessageExtension
    from ..extensions.steering import SteeringExtension
    from ..messages import AgentMessage, ToolCall, UserMessage
    from ..model import AgentModel, ModelEvent, ModelRequest, ModelResponse, ReasoningEffort
    from ..tools.base import AgentTool

_Method = TypeVar("_Method", bound=Callable[..., Any])

#: Set on every method :func:`mixin_api` marks.
MIXIN_API_MARK = "__mixin_api__"


def mixin_api(method: _Method) -> _Method:
    """Mark a method that another mixin of ``Agent`` (or ``Agent`` itself) calls.

    A private method of a mixin is one of two things: a helper only its own mixin uses, or a seam the
    other mixins build on. The two look alike, so the second kind carries this mark. It changes nothing at
    runtime; ``tests/test_agent_mixin_api.py`` reads the source and holds the code to it:

    * a method that another file calls must be marked, and a marked one must be called from another file;
    * a marked method that another *mixin* calls is also declared on :class:`AgentContract`, where the
      type checker learns of it, and a private method declared there is marked and called by a mixin.

    A method without the mark is therefore free to change, because nothing outside its mixin may call it.
    """
    setattr(method, MIXIN_API_MARK, True)
    return method


class AgentContract(AgentPhaseTransitionMixin):
    """The attributes ``Agent`` holds and the methods its mixins call on one another."""

    if TYPE_CHECKING:
        # ================================================================== state
        # All of it is assigned once, in ``Agent.__init__``; no mixin creates state of its own.

        #: The default provider adapter for model calls. A request may bring its own; a request that has
        #: neither fails. ``None`` when the agent was built without one.
        model: AgentModel | None
        #: The tools registered on the agent, by name. Each request's context starts with a copy of
        #: these, and its extensions may add more to that copy.
        tools: dict[str, AgentTool]
        #: The system prompt used when a request does not bring its own.
        system_prompt: str
        #: The default reasoning effort, used when a request does not choose one.
        reasoning_effort: ReasoningEffort
        #: The default for whether a model may ask for several tools at once, and whether tools marked
        #: parallel run together.
        parallel_tool_call: bool
        #: The most recently started request's state; use ``get_state`` to select by session.
        state: AgentState
        #: Every extension, built-ins included, in ascending priority. Hooks are called in this order.
        extensions: tuple[AgentExtension, ...]
        #: How many model calls one input may use before the request fails with an iteration-limit
        #: error. Steering input starts its own count.
        max_iterations: int
        #: How many internal messages (instructions an extension queues for the model) one request may
        #: take up. Zero turns them off.
        max_internal_messages: int
        #: How many paused model steps in a row are continued automatically before the pause is treated
        #: as the end of the answer.
        max_pause_continuations: int
        #: The runtime state of each session this agent has seen, by session id.
        _states: dict[str, AgentState]
        #: The configuration of each request that is running now, by session id. A session in here is
        #: owned by that request, which is what ``is_conflict`` reports.
        _active_configs: dict[str, AgentRunConfig]
        #: Guards ``_states`` and ``_active_configs`` while a request claims or releases its session.
        _sessions_lock: RLock
        #: The configuration ``initialize`` bound, which gives requests without one their session.
        #: ``None`` until the agent is initialized.
        _initialized_config: AgentRunConfig | None
        #: The built-in extension that holds each session's queue of internal messages. It is also in
        #: ``extensions``; it is kept here because the loop reads it directly.
        _internal_message_extension: InternalMessageExtension
        #: The built-in extension that holds each session's queue of steering messages (new user input
        #: that arrives while a request is running). Also in ``extensions``.
        _steering_extension: SteeringExtension

        # ================================================================== Agent
        def is_conflict(self, session_id: str) -> bool:
            """Whether another request currently owns ``session_id``.

            Sessions are exclusive: one request at a time runs on a session. Requests on different
            sessions do not conflict.
            """
            ...

        # ================================================================== hooks (hooks.py)
        async def _broadcast(self, hook: str, context: AgentRunContext, *arguments: object) -> None:
            """Call the lifecycle hook named ``hook`` (such as ``"before_run"``) on every extension.

            Extensions are called one after another in priority order, each with ``context`` and then
            ``arguments``. An error from one stops the broadcast and propagates; later extensions are
            not called and what earlier ones did is not undone. ``hook`` is a name, so it must be one
            that :class:`~zett_agent.extensions.base.AgentExtension` defines.
            """
            ...

        def _stream_model_with_middleware(
            self, context: AgentRunContext, request: ModelRequest, model: AgentModel
        ) -> AsyncIterator[ModelEvent]:
            """Start one model call and return its events, passed through every extension's request hook.

            Each extension's ``on_model_request`` wraps the next one and the first extension is the
            outermost, so it sees the request first and the events last. The innermost call is
            ``model.stream``. Nothing runs until the returned iterator is consumed.
            """
            ...

        # ================================================================== lifecycle (lifecycle.py)
        async def _restore_context(self, context: AgentRunContext) -> None:
            """Begin loading a session: let extensions register their tools and restore its history.

            Calls the ``on_tool`` and ``on_state`` hooks. Messages the hooks add are only queued; they
            are written when :meth:`_settle_context` runs, so the caller can add its own first.
            """
            ...

        async def _settle_context(self, context: AgentRunContext) -> None:
            """Finish loading a session: write out what the setup hooks queued, in the order they were added.

            The second half of :meth:`_restore_context`. After it the history in ``context`` matches what
            is stored, and the session is ready for a message to be appended.
            """
            ...

        # ================================================================== recovery (recovery.py)
        async def _open_resume(self, context: AgentRunContext) -> None:
            """Open a request that carries on an interrupted one, instead of starting from new input.

            Restores the history, works out from its tail where the previous request stopped, makes the
            history valid to send to the model again (unanswered tool calls get a failed result; a reply
            that was cut off or interrupted gets a note telling the model to continue or retry), and
            calls ``before_run``. No user message is appended.
            """
            ...

        async def _close_unanswered_tool_calls(self, context: AgentRunContext, content: str) -> None:
            """Give every call in the latest tool batch that has no result a failed result saying ``content``.

            A provider rejects a history in which a tool call has no answer, so this runs before anything
            new is appended after a request that died mid-batch. The calls are never run again: the
            result only records that the outcome is unknown. Only the latest batch is closed; an older
            broken batch is left as it is.
            """
            ...

        # ================================================================== turns (turns.py)
        async def _run_loop(
            self, context: AgentRunContext, reasoning_effort: ReasoningEffort, parallel_tool_call: bool
        ) -> None:
            """Drive a request until it has its answer: a model call, then its tools or its final reply.

            Repeats while the model asks for tools, handles steering and internal input, continues paused
            steps, and stops when an answer is final. Raises when ``max_iterations`` is exceeded. All
            progress is emitted as events on ``context``; nothing is returned.
            """
            ...

        async def _start_steering(
            self,
            context: AgentRunContext,
            message: UserMessage,
            pending_calls: Sequence[ToolCall] = (),
            active_input: AgentMessage | UserMessage | None = None,
        ) -> None:
            """Switch the request to a steering ``message`` that arrived while it was working.

            ``pending_calls`` are tool calls the model asked for that will now not run; each gets a failed
            "superseded" result so the history stays valid. ``active_input`` is the input being worked on,
            which is reported as interrupted. Then ``message`` is appended and the switch is announced.
            """
            ...

        # ================================================================== model step (model_step.py)
        async def _run_model_step(
            self,
            context: AgentRunContext,
            reasoning_effort: ReasoningEffort,
            parallel_tool_call: bool,
            *,
            may_pause: bool = False,
        ) -> ModelResponse:
            """Make one model call: build the request, stream the reply, store the assistant message.

            Runs the ``before_model`` hooks, sends the history and tools through the model, relays the
            streamed events to the caller, checks the stream is well formed, and appends the reply to the
            history. ``may_pause`` says a paused reply will be continued by the caller, so the pause is
            not written on the stored message. Returns the response, whose ``stop`` says why it ended.
            """
            ...

        async def _announce_stop(self, context: AgentRunContext, response: ModelResponse) -> None:
            """Emit a ``MODEL_STOPPED`` event when ``response`` ended for a reason other than a normal finish.

            Says nothing for a reply that finished or asked for tools.
            """
            ...

        # ================================================================== tool runner (tool_runner.py)
        async def _execute_tools(
            self,
            context: AgentRunContext,
            calls: Sequence[ToolCall],
            parallel_tool_call: bool,
            active_input: AgentMessage | UserMessage | None = None,
        ) -> UserMessage | None:
            """Run the tool calls of one model reply and append a result for each.

            Tools marked parallel run together first, then the rest one by one. Steering input that
            arrives in between skips the calls not yet started, each getting a failed result, and is
            returned. Returns ``None`` when every call ran.
            """
            ...
