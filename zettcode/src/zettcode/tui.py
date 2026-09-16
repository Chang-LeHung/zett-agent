"""ZettCode component composition on the internal differential TUI framework."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from wcwidth import wcwidth
from zett_agent import ToolMessage

from .tui_framework import Span, Style

MAX_STORED_TOOL_OUTPUT = 64_000


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
