"""Shims for standard-library features that are newer than Python 3.10.

The runtime supports Python 3.10 and newer, but a few pieces of it build on
features that only exist from 3.11 onward. Collecting them here keeps the
version checks in one place instead of repeating them across the package.

This module is an implementation detail and is not part of the public API.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager

if sys.version_info >= (3, 11):  # pragma: no cover - selected by the interpreter
    from asyncio import Runner, timeout
    from builtins import BaseExceptionGroup, ExceptionGroup
    from datetime import UTC
    from enum import StrEnum
    from typing import Self

    def add_note(error: BaseException, note: str) -> None:
        """Attach a diagnostic note to one exception."""
        error.add_note(note)

    @contextmanager
    def owned_event_loop() -> Iterator[asyncio.AbstractEventLoop]:
        """Yield a fresh event loop for this thread and close it again on exit.

        ``asyncio.Runner`` releases pending async generators and default-executor
        threads before the loop closes.
        """
        with Runner() as runner:
            yield runner.get_loop()

else:  # pragma: no cover - selected by the interpreter
    from datetime import timezone
    from enum import Enum

    from async_timeout import timeout
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup
    from typing_extensions import Self

    UTC = timezone.utc

    def add_note(error: BaseException, note: str) -> None:
        """Attach a diagnostic note to one exception without changing its text."""
        error.__notes__ = [*getattr(error, "__notes__", []), note]

    class StrEnum(str, Enum):
        """``enum.StrEnum`` for Python 3.10: members are their own string value."""

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            return str.__format__(str(self), format_spec)

    @contextmanager
    def owned_event_loop() -> Iterator[asyncio.AbstractEventLoop]:
        """Yield a fresh event loop for this thread and close it again on exit.

        Python 3.10 has no ``asyncio.Runner``, so this reproduces its shutdown
        order: pending async generators and default-executor threads are
        released before the loop closes.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield loop
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()


if sys.version_info >= (3, 12):  # pragma: no cover - selected by the interpreter
    from typing import TypeAliasType
else:  # pragma: no cover - selected by the interpreter
    from typing_extensions import TypeAliasType

__all__ = [
    "BaseExceptionGroup",
    "ExceptionGroup",
    "Self",
    "StrEnum",
    "TypeAliasType",
    "UTC",
    "add_note",
    "owned_event_loop",
    "timeout",
]
