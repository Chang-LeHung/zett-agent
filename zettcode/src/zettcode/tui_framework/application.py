"""Single-loop component application with differential repainting."""

from __future__ import annotations

import asyncio
import base64

import pyperclip

from .components import Component, Rect
from .events import EventType, InputEvent
from .screen import Canvas, DifferentialRenderer
from .terminal import AsyncInput, Terminal


class Application:
    """Own terminal state, focus, input dispatch, and frame scheduling."""

    def __init__(self, root: Component, *, terminal: Terminal | None = None) -> None:
        self.root = root
        self.terminal = terminal or Terminal()
        self.renderer = DifferentialRenderer(self.terminal.output)
        self.queue: asyncio.Queue[InputEvent] = asyncio.Queue()
        self.focused: Component | None = None
        self.running = False
        self._render_pending = False

    async def run(self) -> None:
        self.running = True
        with self.terminal:
            input_reader = AsyncInput(self.terminal, self.publish)
            input_reader.start()
            try:
                self.invalidate()
                while self.running:
                    event = await self.queue.get()
                    if event.type == EventType.RENDER:
                        self._render_pending = False
                        self._render()
                        continue
                    if event.type == EventType.RESIZE:
                        self.renderer.reset()
                    handled = self.root.handle(event, self)
                    if handled or event.type == EventType.RESIZE:
                        self.invalidate()
            finally:
                input_reader.close()

    def publish(self, event: InputEvent) -> None:
        self.queue.put_nowait(event)

    def invalidate(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self.publish(InputEvent(EventType.RENDER))

    def focus(self, component: Component) -> None:
        if self.focused is component:
            return
        if self.focused is not None and hasattr(self.focused, "focused"):
            self.focused.focused = False
        self.focused = component
        if hasattr(component, "focused"):
            component.focused = True

    def copy(self, text: str) -> None:
        if not text:
            return
        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException:
            encoded = base64.b64encode(text.encode()).decode()
            self.terminal.write(f"\x1b]52;c;{encoded}\x07")

    def refresh(self) -> None:
        """Force a complete repaint, matching the conventional Ctrl-L action."""
        self.renderer.reset()
        self.invalidate()

    def exit(self) -> None:
        self.running = False
        self.publish(InputEvent(EventType.RENDER))

    def _render(self) -> None:
        width, height = self.terminal.size
        canvas = Canvas(width, height)
        self.root.layout(Rect(0, 0, width, height))
        self.root.render(canvas)
        cursor = self.focused.cursor() if self.focused is not None else self.root.cursor()
        self.renderer.render(canvas, cursor=cursor)
