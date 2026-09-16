"""Framework input events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EventType(StrEnum):
    KEY = "key"
    TEXT = "text"
    PASTE = "paste"
    MOUSE = "mouse"
    RESIZE = "resize"
    RENDER = "render"


class MouseAction(StrEnum):
    DOWN = "down"
    UP = "up"
    MOVE = "move"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"


@dataclass(frozen=True, slots=True)
class InputEvent:
    type: EventType
    key: str = ""
    text: str = ""
    x: int = 0
    y: int = 0
    button: int = 0
    action: MouseAction | None = None
    shift: bool = False
    alt: bool = False
    control: bool = False
