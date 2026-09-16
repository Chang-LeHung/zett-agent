"""ZettCode's internally implemented terminal UI framework."""

from .events import EventType, InputEvent, MouseAction
from .screen import Canvas, DifferentialRenderer, Span, Style, TextLine
from .terminal import Terminal

__all__ = [
    "Canvas",
    "DifferentialRenderer",
    "EventType",
    "InputEvent",
    "MouseAction",
    "Span",
    "Style",
    "Terminal",
    "TextLine",
]
