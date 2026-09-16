"""ZettCode component composition on the internal differential TUI framework."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace

from wcwidth import wcwidth
from zett_agent import ToolMessage

from .tui_framework import Span, Style, TextLine

GREEN = "#79b88b"


LIGHT_GREEN = "#9bddad"


MUTED = "#747d77"


DETAIL = "#a3ada6"


FOREGROUND = "#e6e9e7"


MAX_STORED_TOOL_OUTPUT = 64_000


MAX_MARKDOWN_TABLE_ROWS = 100


STYLE_NORMAL = Style(foreground=FOREGROUND)


STYLE_MUTED = Style(foreground=MUTED)


STYLE_DETAIL = Style(foreground=DETAIL)


STYLE_GREEN = Style(foreground=GREEN, bold=True)


def _tool_output(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content or "Completed"
    return message.text or "Multimodal result"


def _tool_preview(message: ToolMessage, *, limit: int = 180) -> str:
    value = " ".join(_tool_output(message).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _arguments_preview(arguments: Mapping[str, object], *, limit: int = 90) -> str:
    if not arguments:
        return ""
    value = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    value = value if len(value) <= limit else value[: limit - 1] + "…"
    return f" {value}"


def _serialized_output(value: object) -> str:
    if value is None:
        return "Completed"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "done"
    if seconds < 1:
        return f"{round(seconds * 1000)} ms"
    return f"{seconds:.1f} s"


def _activity_icon(frame: int) -> str:
    """Return a compact green-theme activity glyph with a visible flash phase."""
    return ("✦", "✧", "·", "✧")[(frame // 3) % 4]


def _activity_spans(value: str, frame: int, *, background: str | None = None) -> tuple[Span, ...]:
    """Sweep one narrow shimmer over otherwise stable activity text."""
    spans: list[Span] = []
    center = frame % (len(value) + 8) - 3
    for index, character in enumerate(value):
        distance = abs(index - center)
        wave = math.exp(-(distance**2) / 5.5) if distance < 7 else 0.0
        spans.append(
            Span(
                character,
                Style(
                    foreground=_blend_color("#617268", "#d2f8dc", wave),
                    background=background,
                    bold=wave > 0.68,
                ),
            )
        )
    return tuple(spans)


def _blend_color(start: str, end: str, ratio: float) -> str:
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    values = (round(left + (right - left) * ratio) for left, right in zip(start_rgb, end_rgb, strict=True))
    return "#" + "".join(f"{value:02x}" for value in values)


def _limit_stored_output(value: str) -> str:
    """Bound transcript memory while retaining useful context from both ends."""
    if len(value) <= MAX_STORED_TOOL_OUTPUT:
        return value
    half = MAX_STORED_TOOL_OUTPUT // 2
    omitted = len(value) - half * 2
    return f"{value[:half]}\n… {omitted:,} characters omitted from tool output …\n{value[-half:]}"


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")


def _terminal_safe(value: str) -> str:
    value = _ANSI_ESCAPE.sub("", value).expandtabs(4)
    return "".join(character if character in "\n" or character.isprintable() else "�" for character in value)


def _bounded_visual_lines(value: str, width: int, limit: int) -> tuple[list[str], int]:
    """Wrap safe tool output and return a bounded viewport plus omitted-row count."""
    visual: list[str] = []
    for source in _terminal_safe(value).splitlines() or [""]:
        current = ""
        used = 0
        for character in source:
            character_width = wcwidth(character)
            character_width = character_width if character_width >= 0 else 1
            if current and used + character_width > width:
                visual.append(current)
                current = ""
                used = 0
            current += character
            used += character_width
        visual.append(current)
    omitted = max(0, len(visual) - limit)
    return visual[:limit], omitted


_INLINE_MARKDOWN = re.compile(
    r"(\*\*.+?\*\*|__.+?__|`[^`]+`|\[[^]]+\]\([^)]+\)|(?<!\*)\*[^*]+\*(?!\*)|(?<!_)_[^_]+_(?!_))"
)


def _markdown_block_lines(value: str, width: int) -> list[TextLine]:
    """Render Markdown blocks, including streaming-safe GFM-style tables."""
    source = value.split("\n")
    result: list[TextLine] = []
    index = 0
    in_code = False
    while index < len(source):
        if not in_code and index + 1 < len(source):
            header = _split_table_row(source[index])
            alignments = _table_alignments(source[index + 1])
            if header is not None and alignments is not None and len(header) == len(alignments):
                rows: list[list[str]] = []
                cursor = index + 2
                while cursor < len(source):
                    row = _split_table_row(source[cursor])
                    if row is None:
                        break
                    rows.append((row + [""] * len(header))[: len(header)])
                    cursor += 1
                result.extend(_render_markdown_table(header, alignments, rows, width))
                index = cursor
                continue
        rendered, in_code = _markdown_line(source[index], in_code=in_code)
        if rendered is not None:
            result.append(TextLine(tuple(rendered)))
        index += 1
    return result


def _split_table_row(value: str) -> list[str] | None:
    """Split table cells while preserving escaped and inline-code pipes."""
    stripped = value.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    in_code = False
    for character in stripped:
        if escaped:
            current.append(character if character in ("|", "\\") else f"\\{character}")
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "`":
            in_code = not in_code
            current.append(character)
        elif character == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells if len(cells) >= 2 else None


def _table_alignments(value: str) -> list[str] | None:
    cells = _split_table_row(value)
    if cells is None:
        return None
    alignments: list[str] = []
    for cell in cells:
        marker = cell.replace(" ", "")
        if re.fullmatch(r":-{3,}:", marker):
            alignments.append("center")
        elif re.fullmatch(r"-{3,}:", marker):
            alignments.append("right")
        elif re.fullmatch(r":?-{3,}", marker):
            alignments.append("left")
        else:
            return None
    return alignments


def _render_markdown_table(
    header: list[str], alignments: list[str], rows: list[list[str]], width: int
) -> list[TextLine]:
    original_columns = len(header)
    column_count = original_columns
    while column_count > 1 and width - (3 * column_count + 1) < 3 * column_count:
        column_count -= 1
    header = header[:column_count]
    alignments = alignments[:column_count]
    rows = [row[:column_count] for row in rows]
    if column_count < original_columns:
        header[-1] = "…"
        rows = [row[:-1] + ["…"] for row in rows]

    natural = []
    for column in range(column_count):
        values = [header[column], *(row[column] for row in rows[:MAX_MARKDOWN_TABLE_ROWS])]
        natural.append(max(3, min(32, max(_text_width(_plain_inline(value)) for value in values))))
    budget = max(column_count * 3, width - (3 * column_count + 1))
    column_widths = natural[:]
    while sum(column_widths) > budget:
        largest = max(range(column_count), key=column_widths.__getitem__)
        if column_widths[largest] <= 3:
            break
        column_widths[largest] -= 1

    border_style = Style(foreground="#587062", dim=True)
    lines = [TextLine((Span(_table_border("╭", "┬", "╮", column_widths), border_style),))]
    lines.append(_table_content_line(header, column_widths, ["center"] * column_count, header=True))
    lines.append(TextLine((Span(_table_border("├", "┼", "┤", column_widths), border_style),)))
    visible_rows = rows[:MAX_MARKDOWN_TABLE_ROWS]
    lines.extend(_table_content_line(row, column_widths, alignments) for row in visible_rows)
    if len(rows) > len(visible_rows):
        omitted = [f"… {len(rows) - len(visible_rows)} rows omitted", *([""] * (column_count - 1))]
        lines.append(_table_content_line(omitted, column_widths, ["left"] * column_count))
    lines.append(TextLine((Span(_table_border("╰", "┴", "╯", column_widths), border_style),)))
    return lines


def _table_border(left: str, junction: str, right: str, widths: list[int]) -> str:
    return left + junction.join("─" * (width + 2) for width in widths) + right


def _table_content_line(
    cells: list[str], widths: list[int], alignments: list[str], *, header: bool = False
) -> TextLine:
    border_style = Style(foreground="#587062", dim=True)
    base_style = Style(foreground=FOREGROUND, bold=header)
    spans: list[Span] = [Span("│", border_style)]
    for value, cell_width, alignment in zip(cells, widths, alignments, strict=True):
        content = _clip_inline_spans(value, cell_width, base_style)
        used = sum(_text_width(span.text) for span in content)
        remaining = max(0, cell_width - used)
        left = remaining if alignment == "right" else remaining // 2 if alignment == "center" else 0
        right = remaining - left
        spans.extend((Span(" " * (left + 1), base_style), *content, Span(" " * (right + 1), base_style)))
        spans.append(Span("│", border_style))
    return TextLine(tuple(spans))


def _clip_inline_spans(value: str, width: int, base_style: Style) -> list[Span]:
    value = value.expandtabs(4)
    source = _inline_markdown(value, base_style=base_style)
    needs_ellipsis = _text_width(_plain_inline(value)) > width
    clipped: list[Span] = []
    used = 0
    truncated = False
    for span in source:
        chunk = ""
        for character in span.text:
            character_width = wcwidth(character)
            character_width = character_width if character_width >= 0 else 1
            reserve = 1 if needs_ellipsis else 0
            if used + character_width > width - reserve:
                truncated = True
                break
            chunk += character
            used += character_width
        if chunk:
            clipped.append(Span(chunk, span.style))
        if truncated:
            break
    if truncated:
        clipped.append(Span("…", base_style))
    return clipped


def _plain_inline(value: str) -> str:
    return "".join(span.text for span in _inline_markdown(value))


def _text_width(value: str) -> int:
    return sum(max(0, wcwidth(character)) for character in value)


def _markdown_line(line: str, *, in_code: bool) -> tuple[list[Span] | None, bool]:
    stripped = line.strip()
    if stripped.startswith("```"):
        if in_code:
            return None, False
        language = stripped[3:].strip()
        label = f"  {language}" if language else ""
        return ([Span(label, Style(foreground=MUTED, background="#222725"))] if label else None), True
    if in_code:
        return [Span(f"  {line}", Style(foreground="#d9dfdb", background="#222725"))], True

    heading = re.match(r"^(#{1,3})\s+(.+)$", line)
    if heading:
        level = len(heading.group(1))
        return _inline_markdown(
            heading.group(2), base_style=Style(foreground=FOREGROUND, bold=True, dim=level > 1)
        ), False

    bullet = re.match(r"^(\s*)[-+*]\s+(.+)$", line)
    if bullet:
        prefix = [Span(f"{bullet.group(1)}• ", Style(foreground=GREEN, bold=True))]
        return prefix + _inline_markdown(bullet.group(2)), False

    quote = re.match(r"^\s*>\s?(.*)$", line)
    if quote:
        return [Span("│ ", STYLE_GREEN)] + _inline_markdown(quote.group(1), base_style=STYLE_DETAIL), False

    return _inline_markdown(line), False


def _inline_markdown(value: str, *, base_style: Style = STYLE_NORMAL) -> list[Span]:
    fragments: list[Span] = []
    position = 0
    for match in _INLINE_MARKDOWN.finditer(value):
        if match.start() > position:
            fragments.append(Span(value[position : match.start()], base_style))
        token = match.group(0)
        if token.startswith(("**", "__")):
            fragments.append(Span(token[2:-2], replace(base_style, bold=True)))
        elif token.startswith("`"):
            fragments.append(Span(token[1:-1], Style(foreground="#d9dfdb", background="#303633")))
        elif token.startswith("["):
            label, _, target = token[1:].partition("](")
            fragments.append(Span(label, Style(foreground=LIGHT_GREEN)))
            fragments.append(Span(f" <{target[:-1]}>", STYLE_MUTED))
        else:
            fragments.append(Span(token[1:-1], replace(base_style, italic=True)))
        position = match.end()
    if position < len(value) or not fragments:
        fragments.append(Span(value[position:], base_style))
    return fragments
