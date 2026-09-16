"""TTY ownership and asynchronous input."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import termios
import tty
from collections.abc import Callable
from types import TracebackType
from typing import TextIO

from .events import EventType, InputEvent
from .input import InputDecoder


class Terminal:
    """Own raw mode and restore the user's terminal on every exit path."""

    def __init__(self, *, input_fd: int | None = None, output: TextIO | None = None) -> None:
        self.input_fd = sys.stdin.fileno() if input_fd is None else input_fd
        self.output = sys.stdout if output is None else output
        self._attributes: list | None = None

    @property
    def size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size((80, 24))
        return max(20, size.columns), max(8, size.lines)

    def __enter__(self) -> Terminal:
        if not os.isatty(self.input_fd) or not self.output.isatty():
            raise RuntimeError("ZettCode TUI requires an interactive terminal")
        self._attributes = termios.tcgetattr(self.input_fd)
        tty.setraw(self.input_fd)
        self.write("\x1b[?1049h\x1b[?7l\x1b[?25l\x1b[?2004h\x1b[?1000h\x1b[?1003h\x1b[?1006h")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.write("\x1b[0m\x1b[?25h\x1b[?7h\x1b[?1000l\x1b[?1003l\x1b[?1006l\x1b[?2004l\x1b[?1049l")
        if self._attributes is not None:
            termios.tcsetattr(self.input_fd, termios.TCSADRAIN, self._attributes)
            self._attributes = None

    def write(self, value: str) -> None:
        self.output.write(value)
        self.output.flush()


class AsyncInput:
    """Publish decoded terminal input into one application queue."""

    def __init__(self, terminal: Terminal, publish: Callable[[InputEvent], None]) -> None:
        self.terminal = terminal
        self.publish = publish
        self.decoder = InputDecoder()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._previous_resize_handler = None
        self._escape_handle: asyncio.TimerHandle | None = None

    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.loop.add_reader(self.terminal.input_fd, self._read_ready)
        self._previous_resize_handler = signal.getsignal(signal.SIGWINCH)
        self.loop.add_signal_handler(signal.SIGWINCH, self._resize)

    def close(self) -> None:
        if self.loop is None:
            return
        self.loop.remove_reader(self.terminal.input_fd)
        self.loop.remove_signal_handler(signal.SIGWINCH)
        if self._escape_handle is not None:
            self._escape_handle.cancel()
            self._escape_handle = None
        if callable(self._previous_resize_handler):
            signal.signal(signal.SIGWINCH, self._previous_resize_handler)
        self.loop = None

    def _read_ready(self) -> None:
        try:
            data = os.read(self.terminal.input_fd, 65536)
        except OSError:
            return
        if self._escape_handle is not None:
            self._escape_handle.cancel()
            self._escape_handle = None
        for event in self.decoder.feed(data):
            self.publish(event)
        if self.decoder.buffer == b"\x1b" and self.loop is not None:
            self._escape_handle = self.loop.call_later(0.03, self._flush_escape)

    def _resize(self) -> None:
        self.publish(InputEvent(EventType.RESIZE))

    def _flush_escape(self) -> None:
        self._escape_handle = None
        event = self.decoder.flush_escape()
        if event is not None:
            self.publish(event)
