"""Incremental terminal byte-stream decoder."""

from __future__ import annotations

import re

from .events import EventType, InputEvent, MouseAction

_MOUSE = re.compile(rb"^\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_KEYS = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1b[H": "home",
    b"\x1b[F": "end",
    b"\x1b[1~": "home",
    b"\x1b[4~": "end",
    b"\x1b[5~": "page_up",
    b"\x1b[6~": "page_down",
    b"\x1b[3~": "delete",
    b"\x1b[2~": "insert",
    b"\x1b[Z": "backtab",
    b"\x1b[1;5D": "ctrl_left",
    b"\x1b[1;5C": "ctrl_right",
    b"\x1b[1;3D": "alt_left",
    b"\x1b[1;3C": "alt_right",
    b"\x1b[1;5H": "ctrl_home",
    b"\x1b[1;5F": "ctrl_end",
}
_CONTROL_KEYS = {
    0x01: "ctrl_a",
    0x02: "ctrl_b",
    0x03: "ctrl_c",
    0x04: "ctrl_d",
    0x05: "ctrl_e",
    0x06: "ctrl_f",
    0x07: "ctrl_g",
    0x09: "tab",
    0x0A: "enter",
    0x0B: "ctrl_k",
    0x0C: "ctrl_l",
    0x0D: "enter",
    0x0E: "ctrl_n",
    0x10: "ctrl_p",
    0x12: "ctrl_r",
    0x14: "ctrl_t",
    0x15: "ctrl_u",
    0x17: "ctrl_w",
    0x19: "ctrl_y",
    0x1A: "ctrl_z",
    0x1F: "ctrl_underscore",
    0x7F: "backspace",
    0x08: "backspace",
}


class InputDecoder:
    """Decode fragmented UTF-8, keyboard, paste, and SGR mouse input."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[InputEvent]:
        self.buffer.extend(data)
        events: list[InputEvent] = []
        while self.buffer:
            event, consumed, incomplete = self._next()
            if incomplete:
                break
            if consumed == 0:
                consumed = 1
            del self.buffer[:consumed]
            if event is not None:
                events.append(event)
        return events

    def flush_escape(self) -> InputEvent | None:
        """Resolve an isolated ESC after the terminal's sequence timeout."""
        if self.buffer == b"\x1b":
            self.buffer.clear()
            return InputEvent(EventType.KEY, key="escape")
        return None

    def _next(self) -> tuple[InputEvent | None, int, bool]:
        data = bytes(self.buffer)
        if data.startswith(b"\x1b[200~"):
            end = data.find(b"\x1b[201~", 6)
            if end < 0:
                return None, 0, True
            return InputEvent(EventType.PASTE, text=data[6:end].decode("utf-8", "replace")), end + 6, False

        mouse = _MOUSE.match(data)
        if mouse:
            code, x, y = (int(mouse.group(index)) for index in (1, 2, 3))
            suffix = mouse.group(4)
            action = _mouse_action(code, suffix)
            return (
                InputEvent(
                    EventType.MOUSE,
                    x=max(0, x - 1),
                    y=max(0, y - 1),
                    button=code & 3,
                    action=action,
                    shift=bool(code & 4),
                    alt=bool(code & 8),
                    control=bool(code & 16),
                ),
                mouse.end(),
                False,
            )

        for sequence, key in sorted(_KEYS.items(), key=lambda item: len(item[0]), reverse=True):
            if data.startswith(sequence):
                return InputEvent(EventType.KEY, key=key), len(sequence), False
            if sequence.startswith(data):
                return None, 0, True

        first = data[0]
        if first in _CONTROL_KEYS:
            return InputEvent(EventType.KEY, key=_CONTROL_KEYS[first]), 1, False
        if first == 0x1B:
            if len(data) == 1:
                return None, 0, True
            if data[1] in (0x0A, 0x0D):
                return InputEvent(EventType.KEY, key="alt_enter"), 2, False
            if data[1] in (0x08, 0x7F):
                return InputEvent(EventType.KEY, key="alt_backspace"), 2, False
            if 0x20 <= data[1] < 0x7F:
                return InputEvent(EventType.KEY, key=chr(data[1]), alt=True), 2, False
            return InputEvent(EventType.KEY, key="escape"), 1, False
        if first < 0x20:
            return None, 1, False

        length = _utf8_length(first)
        if len(data) < length:
            return None, 0, True
        try:
            text = data[:length].decode("utf-8")
        except UnicodeDecodeError:
            text = "�"
            length = 1
        return InputEvent(EventType.TEXT, text=text), length, False


def _utf8_length(first: int) -> int:
    if first < 0x80:
        return 1
    if first & 0xE0 == 0xC0:
        return 2
    if first & 0xF0 == 0xE0:
        return 3
    if first & 0xF8 == 0xF0:
        return 4
    return 1


def _mouse_action(code: int, suffix: bytes) -> MouseAction:
    if code & 64:
        return MouseAction.SCROLL_DOWN if code & 1 else MouseAction.SCROLL_UP
    if code & 32:
        return MouseAction.MOVE
    return MouseAction.UP if suffix == b"m" else MouseAction.DOWN
