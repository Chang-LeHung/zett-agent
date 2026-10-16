"""Calling the extensions: one lifecycle point at a time, and as middleware around a model call."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from ..model import AgentModel, ModelEvent, ModelRequest
from .contract import AgentContract, mixin_api

if TYPE_CHECKING:
    from ..agent import AgentRunContext
    from ..extensions.base import AgentExtension, ModelRequestNext
    from ..extensions.internal_message import InternalMessageExtension
    from ..extensions.steering import SteeringExtension


class ExtensionHooksMixin(AgentContract):
    """Own the extension list and every way the runtime calls into it."""

    @staticmethod
    @mixin_api
    def _assemble_extensions(
        extensions: Sequence[AgentExtension] | None,
    ) -> tuple[tuple[AgentExtension, ...], InternalMessageExtension, SteeringExtension]:
        """Validate the caller's extensions, add the built-ins, and order them by priority.

        Returns the ordered extensions and the two built-ins the runtime talks to directly. A caller
        may not supply either built-in, and every name, built-ins included, must be unique.
        """
        from ..extensions.internal_message import InternalMessageExtension
        from ..extensions.memory import InMemoryMessageAccumulator
        from ..extensions.steering import SteeringExtension
        from ..extensions.tool_guidelines import ToolGuidelinesExtension

        configured = list(
            (InMemoryMessageAccumulator(), ToolGuidelinesExtension()) if extensions is None else tuple(extensions)
        )
        if any(isinstance(extension, InternalMessageExtension) for extension in configured):
            raise ValueError("InternalMessageExtension is an Agent built-in and cannot be supplied in extensions")
        internal = InternalMessageExtension()
        configured.append(internal)
        if any(isinstance(extension, SteeringExtension) for extension in configured):
            raise ValueError("SteeringExtension is an Agent built-in and cannot be supplied in extensions")
        steering = SteeringExtension()
        configured.append(steering)
        names: set[str] = set()
        for extension in configured:
            name = extension.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Extension name must be a non-empty string")
            if name in names:
                raise ValueError(f"Duplicate extension name: {name!r}")
            names.add(name)
        # Python's sort is stable, so extensions sharing a priority preserve the caller's
        # registration order. Lifecycle and middleware paths both use the resulting tuple, giving
        # extensions one shared and predictable ordering.
        return tuple(sorted(configured, key=lambda extension: extension.priority)), internal, steering

    @mixin_api
    async def _broadcast(self, hook: str, context: AgentRunContext, *arguments: Any) -> None:
        """Call one lifecycle hook, such as ``"before_run"``, on every extension in priority order.

        An error from one extension stops the broadcast and propagates; the extensions after it are
        not called, and work already done is not rolled back.
        """
        for extension in self.extensions:
            await getattr(extension, hook)(context, *arguments)

    @mixin_api
    def _stream_model_with_middleware(
        self,
        context: AgentRunContext,
        request: ModelRequest,
        model: AgentModel,
    ) -> AsyncIterator[ModelEvent]:
        """Build one request middleware chain with registration order outside-in."""
        call_next: ModelRequestNext = model.stream
        for extension in reversed(self.extensions):
            inner = call_next

            def wrapped(
                next_request: ModelRequest,
                *,
                hook: AgentExtension = extension,
                next_handler: ModelRequestNext = inner,
            ) -> AsyncIterator[ModelEvent]:
                return hook.on_model_request(context, next_request, next_handler)

            call_next = wrapped
        return call_next(request)
