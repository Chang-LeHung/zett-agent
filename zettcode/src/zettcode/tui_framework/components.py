"""Composable terminal widgets with no rendering side effects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import monotonic
from typing import Protocol

from wcwidth import wcwidth

from .events import EventType, InputEvent, MouseAction
from .screen import DEFAULT_STYLE, Canvas, Span, Style, TextLine


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


class UIActions(Protocol):
    def focus(self, component: Component) -> None: ...

    def invalidate(self) -> None: ...

    def copy(self, text: str) -> None: ...

    def refresh(self) -> None: ...

    def exit(self) -> None: ...


class Component:
    """Base class for layout, painting, input, and cursor ownership."""

    def __init__(self) -> None:
        self.rect = Rect(0, 0, 0, 0)

    def layout(self, rect: Rect) -> None:
        self.rect = rect

    def render(self, canvas: Canvas) -> None:
        pass

    def handle(self, event: InputEvent, actions: UIActions) -> bool:
        return False

    def cursor(self) -> tuple[int, int] | None:
        return None

    def preferred_height(self, width: int) -> int:
        return 1


class Text(Component):
    """Single-line dynamic label."""

    def __init__(self, value: str | Callable[[], str], style: Style = DEFAULT_STYLE, *, align: str = "left") -> None:
        super().__init__()
        self.value = value
        self.style = style
        self.align = align

    def render(self, canvas: Canvas) -> None:
        value = self.value() if callable(self.value) else self.value
        width = sum(_character_width(character) for character in value)
        x = self.rect.x + max(0, self.rect.width - width) if self.align == "right" else self.rect.x
        canvas.draw_text(x, self.rect.y, value, self.style, max_width=self.rect.width)


class Rule(Component):
    """Horizontal separator."""

    def __init__(self, style: Style = DEFAULT_STYLE, character: str = "─") -> None:
        super().__init__()
        self.style = style
        self.character = character

    def render(self, canvas: Canvas) -> None:
        canvas.draw_text(self.rect.x, self.rect.y, self.character * self.rect.width, self.style)


@dataclass(frozen=True, slots=True)
class BoxChild:
    component: Component
    size: int | Callable[[int], int] | None = None
    flex: int = 0


class VBox(Component):
    """Vertical layout with fixed, dynamic, and flexible children."""

    def __init__(self, children: Sequence[BoxChild]) -> None:
        super().__init__()
        self.children = tuple(children)

    def layout(self, rect: Rect) -> None:
        super().layout(rect)
        fixed = [self._size(child, rect.width) for child in self.children]
        remaining = max(0, rect.height - sum(fixed))
        flex_total = sum(child.flex for child in self.children)
        y = rect.y
        distributed = 0
        for index, child in enumerate(self.children):
            height = fixed[index]
            if child.flex and flex_total:
                height += remaining * child.flex // flex_total
                distributed += remaining * child.flex // flex_total
            if index == len(self.children) - 1:
                height += remaining - distributed
            child.component.layout(Rect(rect.x, y, rect.width, max(0, height)))
            y += height

    def render(self, canvas: Canvas) -> None:
        for child in self.children:
            child.component.render(canvas)

    def handle(self, event: InputEvent, actions: UIActions) -> bool:
        return any(child.component.handle(event, actions) for child in self.children)

    def cursor(self) -> tuple[int, int] | None:
        for child in self.children:
            cursor = child.component.cursor()
            if cursor is not None:
                return cursor
        return None

    @staticmethod
    def _size(child: BoxChild, width: int) -> int:
        if callable(child.size):
            return max(0, child.size(width))
        if child.size is not None:
            return max(0, child.size)
        return 0


class HBox(Component):
    """Horizontal layout for compact header and status rows."""

    def __init__(self, children: Sequence[BoxChild]) -> None:
        super().__init__()
        self.children = tuple(children)

    def layout(self, rect: Rect) -> None:
        super().layout(rect)
        fixed = [self._size(child, rect.height) for child in self.children]
        remaining = max(0, rect.width - sum(fixed))
        flex_total = sum(child.flex for child in self.children)
        x = rect.x
        distributed = 0
        for index, child in enumerate(self.children):
            width = fixed[index]
            if child.flex and flex_total:
                width += remaining * child.flex // flex_total
                distributed += remaining * child.flex // flex_total
            if index == len(self.children) - 1:
                width += remaining - distributed
            child.component.layout(Rect(x, rect.y, max(0, width), rect.height))
            x += width

    def render(self, canvas: Canvas) -> None:
        for child in self.children:
            child.component.render(canvas)

    def handle(self, event: InputEvent, actions: UIActions) -> bool:
        return any(child.component.handle(event, actions) for child in self.children)

    @staticmethod
    def _size(child: BoxChild, height: int) -> int:
        if callable(child.size):
            return max(0, child.size(height))
        if child.size is not None:
            return max(0, child.size)
        return 0


@dataclass(frozen=True, slots=True)
class _VisualLine:
    spans: tuple[Span, ...]
    metadata: object | None

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)


class ScrollableText(Component):
    """Scrollable rich text with selection and metadata-aware interaction."""

    def __init__(
        self,
        lines: Callable[[int], Sequence[TextLine]],
        *,
        on_click: Callable[[object | None], bool] | None = None,
        on_hover: Callable[[object | None], bool] | None = None,
        on_selection: Callable[[str], None] | None = None,
        clock: Callable[[], float] = monotonic,
        double_click_seconds: float = 0.4,
    ) -> None:
        super().__init__()
        self.lines_provider = lines
        self.on_click = on_click
        self.on_hover = on_hover
        self.on_selection = on_selection
        self.clock = clock
        self.double_click_seconds = double_click_seconds
        self.scroll_top = 0
        self.follow_tail = True
        self._visual_lines: list[_VisualLine] = []
        self._selection_start: tuple[int, int] | None = None
        self._selection_end: tuple[int, int] | None = None
        self._dragging = False
        self._mouse_down: tuple[int, int] | None = None
        self._mouse_down_at = float("-inf")
        self._double_click = False
        self._last_click_at = float("-inf")
        self._last_click_line: int | None = None

    def render(self, canvas: Canvas) -> None:
        self._visual_lines = _wrap_lines(self.lines_provider(max(1, self.rect.width)), max(1, self.rect.width))
        maximum = max(0, len(self._visual_lines) - self.rect.height)
        if self.follow_tail:
            self.scroll_top = maximum
        else:
            self.scroll_top = min(self.scroll_top, maximum)
        for local_y in range(self.rect.height):
            index = self.scroll_top + local_y
            if index >= len(self._visual_lines):
                break
            line = self._visual_lines[index]
            if line.spans and line.spans[0].style.background:
                canvas.fill(self.rect.x, self.rect.y + local_y, self.rect.width, 1, line.spans[0].style)
            self._draw_line(canvas, line, index, local_y)

    def handle(self, event: InputEvent, actions: UIActions) -> bool:
        if event.type == EventType.KEY:
            if event.key == "page_up":
                self.scroll(-max(1, self.rect.height - 2))
                return True
            if event.key == "page_down":
                self.scroll(max(1, self.rect.height - 2))
                return True
            if getattr(actions, "focused", None) is not self:
                return False
            if event.key == "up":
                self.scroll(-1)
                return True
            if event.key == "down":
                self.scroll(1)
                return True
            if event.key in ("home", "ctrl_home"):
                self.scroll_top = 0
                self.follow_tail = False
                return True
            if event.key in ("end", "ctrl_end"):
                self.scroll_top = max(0, len(self._visual_lines) - self.rect.height)
                self.follow_tail = True
                return True
            return False
        if event.type != EventType.MOUSE or event.action is None:
            return False
        inside = self.rect.contains(event.x, event.y)
        if event.action == MouseAction.SCROLL_UP and inside:
            self.scroll(-3)
            return True
        if event.action == MouseAction.SCROLL_DOWN and inside:
            self.scroll(3)
            return True
        point = self._point(event.x, event.y) if inside else None
        metadata = self._metadata(point)
        if event.action == MouseAction.MOVE:
            changed = self.on_hover(metadata) if self.on_hover is not None else False
            if self._dragging:
                if event.y < self.rect.y:
                    self.scroll(-1)
                elif event.y >= self.rect.y + self.rect.height:
                    self.scroll(1)
                point = self._point_clamped(event.x, event.y)
                self._selection_end = point
                changed = True
            return changed
        if event.action == MouseAction.DOWN and inside:
            actions.focus(self)
            self._mouse_down = point
            assert point is not None
            now = self.clock()
            self._double_click = (
                self._last_click_line == point[0] and now - self._last_click_at <= self.double_click_seconds
            )
            self._mouse_down_at = now
            if self._double_click:
                self._select_line(point[0])
                self._dragging = False
            else:
                if not event.shift or self._selection_start is None:
                    self._selection_start = point
                self._selection_end = point
                self._dragging = True
            return True
        if event.action == MouseAction.UP and (self._dragging or self._double_click):
            was_double_click = self._double_click
            self._double_click = False
            if self._dragging:
                self._dragging = False
                if point is not None:
                    self._selection_end = point
                clicked = point == self._mouse_down
                if clicked and self.on_click is not None and self.on_click(metadata):
                    self.clear_selection()
                if clicked and point is not None:
                    self._last_click_at = self._mouse_down_at
                    self._last_click_line = point[0]
                else:
                    # Dragging is a selection gesture, not the first half of a
                    # double click.
                    self._last_click_at = float("-inf")
                    self._last_click_line = None
            if was_double_click:
                # Start a fresh click chain after a completed double click so a
                # third click does not unexpectedly replace the line selection.
                self._last_click_at = float("-inf")
                self._last_click_line = None
            self._copy_selection(actions)
            return True
        return False

    def scroll(self, amount: int) -> None:
        maximum = max(0, len(self._visual_lines) - self.rect.height)
        self.scroll_top = min(max(0, self.scroll_top + amount), maximum)
        self.follow_tail = self.scroll_top >= maximum

    def selected_text(self) -> str:
        bounds = self._selection_bounds()
        if bounds is None:
            return ""
        (start_line, start_column), (end_line, end_column) = bounds
        selected: list[str] = []
        for index in range(start_line, end_line + 1):
            text = self._visual_lines[index].text
            left = start_column if index == start_line else 0
            right = end_column if index == end_line else _display_width(text)
            selected.append(_display_slice(text, left, right))
        return "\n".join(selected)

    def clear_selection(self) -> None:
        self._selection_start = None
        self._selection_end = None

    def _select_line(self, line: int) -> None:
        if not 0 <= line < len(self._visual_lines):
            return
        self._selection_start = (line, 0)
        self._selection_end = (line, _display_width(self._visual_lines[line].text))

    def _copy_selection(self, actions: UIActions) -> None:
        selected = self.selected_text()
        if not selected:
            return
        actions.copy(selected)
        if self.on_selection is not None:
            self.on_selection(selected)

    def _point_clamped(self, x: int, y: int) -> tuple[int, int]:
        clamped_x = min(max(x, self.rect.x), max(self.rect.x, self.rect.x + self.rect.width - 1))
        clamped_y = min(max(y, self.rect.y), max(self.rect.y, self.rect.y + self.rect.height - 1))
        return self._point(clamped_x, clamped_y)

    def _draw_line(self, canvas: Canvas, line: _VisualLine, index: int, local_y: int) -> None:
        x = self.rect.x
        column = 0
        bounds = self._selection_bounds()
        for span in line.spans:
            for character in span.text:
                width = _character_width(character)
                selected = False
                if bounds is not None:
                    (start_line, start_column), (end_line, end_column) = bounds
                    selected = (start_line, start_column) <= (index, column) < (end_line, end_column)
                style = replace(span.style, reverse=not span.style.reverse) if selected else span.style
                canvas.set_cell(x + column, self.rect.y + local_y, character, style)
                column += width

    def _point(self, x: int, y: int) -> tuple[int, int]:
        line = min(max(0, self.scroll_top + y - self.rect.y), max(0, len(self._visual_lines) - 1))
        column = (
            min(max(0, x - self.rect.x), _display_width(self._visual_lines[line].text)) if self._visual_lines else 0
        )
        return line, column

    def _metadata(self, point: tuple[int, int] | None) -> object | None:
        return self._visual_lines[point[0]].metadata if point is not None and self._visual_lines else None

    def _selection_bounds(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if self._selection_start is None or self._selection_end is None:
            return None
        start, end = sorted((self._selection_start, self._selection_end))
        if start == end:
            return None
        return start, end


def _wrap_lines(lines: Sequence[TextLine], width: int) -> list[_VisualLine]:
    result: list[_VisualLine] = []
    for line in lines:
        spans = _expand_tabs(line.spans)
        current: list[Span] = []
        used = 0
        for span in spans:
            chunk = ""
            for character in span.text:
                char_width = _character_width(character)
                if used and used + char_width > width:
                    if chunk:
                        current.append(Span(chunk, span.style))
                    result.append(_VisualLine(tuple(current), line.metadata))
                    current, chunk, used = [], "", 0
                chunk += character
                used += char_width
            if chunk:
                current.append(Span(chunk, span.style))
        result.append(_VisualLine(tuple(current), line.metadata))
    return result or [_VisualLine((), None)]


def _expand_tabs(spans: tuple[Span, ...], *, tab_size: int = 4) -> tuple[Span, ...]:
    """Expand tabs across span boundaries while preserving their styles."""
    expanded: list[Span] = []
    column = 0
    for span in spans:
        chunk = ""
        for character in span.text:
            if character == "\t":
                spaces = tab_size - (column % tab_size)
                chunk += " " * spaces
                column += spaces
            else:
                chunk += character
                column += _character_width(character)
        if chunk:
            expanded.append(Span(chunk, span.style))
    return tuple(expanded)


def _display_width(value: str) -> int:
    return sum(_character_width(character) for character in value)


def _display_slice(value: str, start: int, end: int) -> str:
    output = []
    column = 0
    for character in value:
        width = _character_width(character)
        if column >= end:
            break
        if column + width > start:
            output.append(character)
        column += width
    return "".join(output)


def _character_width(character: str) -> int:
    width = wcwidth(character)
    return width if width >= 0 else 1
