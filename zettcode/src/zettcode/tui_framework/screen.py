"""Cell canvas and ANSI differential renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import TextIO

from wcwidth import wcwidth


@dataclass(frozen=True, slots=True)
class Style:
    """Terminal style independent from ANSI encoding."""

    foreground: str | None = None
    background: str | None = None
    bold: bool = False
    italic: bool = False
    dim: bool = False
    reverse: bool = False


DEFAULT_STYLE = Style()


@dataclass(frozen=True, slots=True)
class Span:
    """One styled text fragment."""

    text: str
    style: Style = DEFAULT_STYLE


@dataclass(frozen=True, slots=True)
class TextLine:
    """One logical line with optional component-owned metadata."""

    spans: tuple[Span, ...] = ()
    metadata: object | None = None

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)


@dataclass(frozen=True, slots=True)
class Cell:
    """One terminal column; wide glyph tails use continuation=True."""

    character: str = " "
    style: Style = DEFAULT_STYLE
    continuation: bool = False


@dataclass(slots=True)
class Canvas:
    """Clipped two-dimensional cell buffer."""

    width: int
    height: int
    cells: list[list[Cell]] = field(init=False)

    def __post_init__(self) -> None:
        self.cells = [[Cell() for _ in range(self.width)] for _ in range(self.height)]

    def set_cell(self, x: int, y: int, character: str, style: Style = DEFAULT_STYLE) -> int:
        """Draw one printable glyph and return its display width."""
        character_width = wcwidth(character)
        # Never let model or tool output inject terminal control sequences into
        # the renderer. Combining marks are the only zero-width input retained.
        if character_width < 0:
            character = "�"
            character_width = 1
        if not (0 <= y < self.height) or x >= self.width:
            return character_width
        width = character_width
        if width == 0:
            if x > 0:
                previous = self.cells[y][x - 1]
                self.cells[y][x - 1] = Cell(previous.character + character, previous.style, previous.continuation)
            return 0
        if x < 0 or x + width > self.width:
            return width
        self.cells[y][x] = Cell(character, style)
        for offset in range(1, width):
            self.cells[y][x + offset] = Cell("", style, continuation=True)
        return width

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        style: Style = DEFAULT_STYLE,
        *,
        max_width: int | None = None,
        tab_size: int = 4,
    ) -> int:
        """Draw one unwrapped line and return consumed columns."""
        used = 0
        limit = self.width - x if max_width is None else min(max_width, self.width - x)
        for character in text:
            if character in "\r\n":
                break
            if character == "\t":
                spaces = tab_size - ((x + used) % tab_size)
                spaces = min(spaces, max(0, limit - used))
                self.fill(x + used, y, spaces, 1, style)
                used += spaces
                continue
            width = wcwidth(character)
            width = width if width >= 0 else 1
            if used + width > limit:
                break
            self.set_cell(x + used, y, character, style)
            used += width
        return used

    def draw_spans(self, x: int, y: int, spans: tuple[Span, ...], *, max_width: int | None = None) -> int:
        """Draw styled fragments without crossing the requested width."""
        used = 0
        limit = self.width - x if max_width is None else min(max_width, self.width - x)
        for span in spans:
            if used >= limit:
                break
            used += self.draw_text(x + used, y, span.text, span.style, max_width=limit - used)
        return used

    def fill(self, x: int, y: int, width: int, height: int, style: Style, character: str = " ") -> None:
        """Fill one clipped rectangle."""
        for row in range(max(0, y), min(self.height, y + height)):
            for column in range(max(0, x), min(self.width, x + width)):
                self.cells[row][column] = Cell(character, style)


class DifferentialRenderer:
    """Write only changed cell ranges since the previous frame."""

    def __init__(self, output: TextIO) -> None:
        self.output = output
        self.previous: Canvas | None = None

    def reset(self) -> None:
        self.previous = None

    def render(self, canvas: Canvas, *, cursor: tuple[int, int] | None = None) -> None:
        full = self.previous is None or (self.previous.width, self.previous.height) != (canvas.width, canvas.height)
        stream = StringIO()
        if full:
            stream.write("\x1b[2J")
        for y, row in enumerate(canvas.cells):
            previous_row = None if full or self.previous is None else self.previous.cells[y]
            if previous_row == row:
                continue
            start, end = _changed_bounds(row, previous_row)
            stream.write(f"\x1b[{y + 1};{start + 1}H")
            active_style: Style | None = None
            for cell in row[start : end + 1]:
                if cell.continuation:
                    continue
                if cell.style != active_style:
                    stream.write(_ansi_style(cell.style))
                    active_style = cell.style
                stream.write(cell.character or " ")
            stream.write("\x1b[0m")
        if cursor is None:
            stream.write("\x1b[?25l")
        else:
            x, y = cursor
            stream.write(f"\x1b[{y + 1};{x + 1}H\x1b[?25h")
        self.output.write(stream.getvalue())
        self.output.flush()
        self.previous = canvas


def _changed_bounds(row: list[Cell], previous: list[Cell] | None) -> tuple[int, int]:
    """Return a glyph-safe inclusive range containing every changed cell."""
    if previous is None:
        return 0, len(row) - 1

    changed = [index for index, cell in enumerate(row) if cell != previous[index]]
    start, end = changed[0], changed[-1]

    # A wide glyph occupies a leading cell plus one or more continuation cells.
    # Include its leader so replacing either half never leaves a stale terminal glyph.
    while start > 0 and (row[start].continuation or previous[start].continuation):
        start -= 1
    while end + 1 < len(row) and (row[end + 1].continuation or previous[end + 1].continuation):
        end += 1
    return start, end


def _ansi_style(style: Style) -> str:
    codes = ["0"]
    if style.bold:
        codes.append("1")
    if style.dim:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.reverse:
        codes.append("7")
    if style.foreground:
        codes.append(_truecolor(style.foreground, foreground=True))
    if style.background:
        codes.append(_truecolor(style.background, foreground=False))
    return f"\x1b[{';'.join(codes)}m"


def _truecolor(value: str, *, foreground: bool) -> str:
    value = value.removeprefix("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"{'38' if foreground else '48'};2;{red};{green};{blue}"
