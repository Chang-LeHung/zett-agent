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


class TextInput(Component):
    """Multiline Readline-style editor with history and visible cursor."""

    def __init__(
        self,
        submit: Callable[[str], bool | None],
        *,
        prompt: str = "› ",
        max_height: int = 7,
        completions: Sequence[str] = (),
        history_limit: int = 200,
    ) -> None:
        super().__init__()
        self.submit = submit
        self.prompt = prompt
        self.max_height = max_height
        self.completions = tuple(completions)
        self.history_limit = history_limit
        self.text = ""
        self.position = 0
        self.focused = True
        self._cursor_screen: tuple[int, int] | None = None
        self._kill_buffer = ""
        self._undo: list[tuple[str, int]] = []
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._completion_state: tuple[str, tuple[str, ...], int] | None = None

    def preferred_height(self, width: int) -> int:
        lines, _ = _layout_input(self.text, self.position, width, _display_width(self.prompt))
        return min(self.max_height, max(1, len(lines)))

    def render(self, canvas: Canvas) -> None:
        prompt_width = _display_width(self.prompt)
        lines, (cursor_row, cursor_column) = _layout_input(
            self.text,
            self.position,
            self.rect.width,
            prompt_width,
        )
        first_visible = max(0, cursor_row - self.rect.height + 1)
        for local_row, line in enumerate(lines[first_visible : first_visible + self.rect.height]):
            source_row = first_visible + local_row
            x = self.rect.x + (prompt_width if source_row == 0 else 0)
            if source_row == 0:
                canvas.draw_text(
                    self.rect.x,
                    self.rect.y + local_row,
                    self.prompt,
                    Style(foreground="#9bddad", bold=True),
                )
            canvas.draw_text(
                x,
                self.rect.y + local_row,
                line,
                Style(foreground="#e6e9e7"),
                max_width=self.rect.width - (x - self.rect.x),
            )
        cursor_x = self.rect.x + (prompt_width if cursor_row == 0 else 0) + cursor_column
        self._cursor_screen = (cursor_x, self.rect.y + cursor_row - first_visible)

    def handle(self, event: InputEvent, actions: UIActions) -> bool:
        if event.type == EventType.MOUSE and self.rect.contains(event.x, event.y) and event.action == MouseAction.DOWN:
            actions.focus(self)
            return True
        if not self.focused:
            return False
        if event.type in (EventType.TEXT, EventType.PASTE):
            self._insert(event.text)
            return True
        if event.type != EventType.KEY:
            return False
        if event.key not in ("tab", "backtab"):
            self._completion_state = None
        if event.alt:
            return self._handle_alt(event.key)
        match event.key:
            case "left" | "ctrl_b":
                self.position = max(0, self.position - 1)
            case "right" | "ctrl_f":
                self.position = min(len(self.text), self.position + 1)
            case "home" | "ctrl_a":
                self.position = self._line_start()
            case "end" | "ctrl_e":
                self.position = self._line_end()
            case "ctrl_home":
                self.position = 0
            case "ctrl_end":
                self.position = len(self.text)
            case "ctrl_left" | "alt_left":
                self.position = _previous_word(self.text, self.position)
            case "ctrl_right" | "alt_right":
                self.position = _next_word(self.text, self.position)
            case "up":
                if not self._move_vertical(-1):
                    self._move_history(-1)
            case "down":
                if not self._move_vertical(1):
                    self._move_history(1)
            case "ctrl_p":
                self._move_history(-1)
            case "ctrl_n":
                self._move_history(1)
            case "backspace":
                if self.position:
                    self._delete(self.position - 1, self.position)
            case "delete" | "ctrl_d":
                if self.position < len(self.text):
                    self._delete(self.position, self.position + 1)
            case "ctrl_w" | "alt_backspace":
                self._delete(_previous_word(self.text, self.position), self.position, kill=True)
            case "ctrl_u":
                self._delete(self._line_start(), self.position, kill=True)
            case "ctrl_k":
                self._delete(self.position, self._line_end(), kill=True)
            case "ctrl_y":
                if self._kill_buffer:
                    self._insert(self._kill_buffer)
            case "ctrl_z" | "ctrl_underscore":
                self._undo_once()
            case "ctrl_g" | "escape":
                self._leave_history()
            case "ctrl_r":
                self._reverse_history_search()
            case "tab":
                self._complete(1)
            case "backtab":
                self._complete(-1)
            case "alt_enter":
                self._insert("\n")
            case "enter":
                value = self.text.strip()
                if value and self.submit(value) is not False:
                    if not self._history or self._history[-1] != value:
                        self._history.append(value)
                        del self._history[: -self.history_limit]
                    self.clear()
            case _:
                return False
        return True

    def cursor(self) -> tuple[int, int] | None:
        return self._cursor_screen if self.focused else None

    def _insert(self, value: str) -> None:
        if not value:
            return
        # Tabs have no glyph of their own. Store their visual expansion so
        # cursor movement, wrapping, selection, and rendering share one width.
        value = value.expandtabs(4)
        self._remember_undo()
        self._leave_history()
        self.text = self.text[: self.position] + value + self.text[self.position :]
        self.position += len(value)

    def clear(self) -> None:
        """Reset the current draft and transient editing modes."""
        self.text = ""
        self.position = 0
        self._undo.clear()
        self._history_index = None
        self._history_draft = ""
        self._completion_state = None

    def _handle_alt(self, key: str) -> bool:
        match key.lower():
            case "b":
                self.position = _previous_word(self.text, self.position)
            case "f":
                self.position = _next_word(self.text, self.position)
            case "d":
                self._delete(self.position, _next_word(self.text, self.position), kill=True)
            case "<":
                self.position = 0
            case ">":
                self.position = len(self.text)
            case _:
                return False
        return True

    def _line_start(self) -> int:
        return self.text.rfind("\n", 0, self.position) + 1

    def _line_end(self) -> int:
        end = self.text.find("\n", self.position)
        return len(self.text) if end < 0 else end

    def _move_vertical(self, direction: int) -> bool:
        start = self._line_start()
        column = self.position - start
        if direction < 0:
            if start == 0:
                return False
            previous_end = start - 1
            previous_start = self.text.rfind("\n", 0, previous_end) + 1
            self.position = min(previous_start + column, previous_end)
            return True
        end = self._line_end()
        if end == len(self.text):
            return False
        next_start = end + 1
        next_end = self.text.find("\n", next_start)
        if next_end < 0:
            next_end = len(self.text)
        self.position = min(next_start + column, next_end)
        return True

    def _delete(self, start: int, end: int, *, kill: bool = False) -> None:
        start, end = max(0, start), min(len(self.text), end)
        if start >= end:
            return
        self._remember_undo()
        removed = self.text[start:end]
        if kill:
            self._kill_buffer = removed
        self.text = self.text[:start] + self.text[end:]
        self.position = start
        self._leave_history()

    def _remember_undo(self) -> None:
        state = (self.text, self.position)
        if not self._undo or self._undo[-1] != state:
            self._undo.append(state)
            del self._undo[:-100]

    def _undo_once(self) -> None:
        if self._undo:
            self.text, self.position = self._undo.pop()
            self._leave_history()

    def _move_history(self, direction: int) -> None:
        if not self._history:
            return
        if self._history_index is None:
            if direction > 0:
                return
            self._history_draft = self.text
            self._history_index = len(self._history) - 1
        else:
            next_index = self._history_index + direction
            if next_index >= len(self._history):
                self.text = self._history_draft
                self.position = len(self.text)
                self._history_index = None
                return
            self._history_index = max(0, next_index)
        self.text = self._history[self._history_index]
        self.position = len(self.text)

    def _leave_history(self) -> None:
        self._history_index = None

    def _reverse_history_search(self) -> None:
        if not self._history:
            return
        query = self.text if self._history_index is None else self._history_draft
        start = len(self._history) if self._history_index is None else self._history_index
        match = next((index for index in range(start - 1, -1, -1) if query in self._history[index]), None)
        if match is None:
            return
        if self._history_index is None:
            self._history_draft = query
        self._history_index = match
        self.text = self._history[match]
        self.position = len(self.text)

    def _complete(self, direction: int) -> None:
        state = self._completion_state
        if state is None:
            start = self.text.rfind("\n", 0, self.position) + 1
            token = self.text[start : self.position]
            if not token.startswith("/") or any(character.isspace() for character in token):
                return
            matches = tuple(value for value in self.completions if value.startswith(token))
            if not matches:
                return
            state = (token, matches, -1 if direction > 0 else 0)
        original, matches, index = state
        index = (index + direction) % len(matches)
        start = self.text.rfind("\n", 0, self.position) + 1
        current = self.text[start : self.position]
        self._remember_undo()
        self.text = self.text[:start] + matches[index] + self.text[self.position :]
        self.position += len(matches[index]) - len(current)
        self._completion_state = (original, matches, index)


def _previous_word(value: str, position: int) -> int:
    index = position
    while index and value[index - 1].isspace():
        index -= 1
    if index and _word_character(value[index - 1]):
        while index and _word_character(value[index - 1]):
            index -= 1
    else:
        while index and not value[index - 1].isspace() and not _word_character(value[index - 1]):
            index -= 1
    return index


def _next_word(value: str, position: int) -> int:
    index = position
    while index < len(value) and value[index].isspace():
        index += 1
    if index < len(value) and _word_character(value[index]):
        while index < len(value) and _word_character(value[index]):
            index += 1
    else:
        while index < len(value) and not value[index].isspace() and not _word_character(value[index]):
            index += 1
    return index


def _word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


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


def _layout_input(text: str, position: int, width: int, prompt_width: int) -> tuple[list[str], tuple[int, int]]:
    """Wrap editable text while preserving a source-index cursor."""
    width = max(1, width)
    rows = [""]
    row = 0
    column = 0
    cursor = (0, 0)
    for index, character in enumerate(text):
        if index == position:
            available = max(1, width - (prompt_width if row == 0 else 0))
            cursor = (row + 1, 0) if column >= available else (row, column)
        if character == "\n":
            rows.append("")
            row += 1
            column = 0
            continue
        character_width = _character_width(character)
        available = max(1, width - (prompt_width if row == 0 else 0))
        if column and column + character_width > available:
            rows.append("")
            row += 1
            column = 0
        rows[row] += character
        column += character_width
    if position == len(text):
        available = max(1, width - (prompt_width if row == 0 else 0))
        if column >= available:
            rows.append("")
            cursor = (row + 1, 0)
        else:
            cursor = (row, column)
    return rows, cursor


def _character_width(character: str) -> int:
    width = wcwidth(character)
    return width if width >= 0 else 1
