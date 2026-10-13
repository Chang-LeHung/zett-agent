"""Shims for standard-library features that are newer than Python 3.10.

The runtime supports Python 3.10 and newer, but a few pieces of it build on
features that only exist from 3.11 onward. Collecting them here keeps the
version checks in one place instead of repeating them across the package.

This module is an implementation detail and is not part of the public API.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncio

if sys.version_info >= (3, 11):  # pragma: no cover - selected by the interpreter
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
        from asyncio import Runner

        with Runner() as runner:
            yield runner.get_loop()

else:  # pragma: no cover - selected by the interpreter
    from datetime import timezone
    from enum import Enum

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
        import asyncio

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


def __getattr__(name: str) -> object:
    """Resolve the expensive shims only when an importer asks for them.

    Importing asyncio costs about 18 ms and uuid6 about 20 ms, while most
    modules that need this file only want ``StrEnum``, ``UTC``, or a type alias.
    Resolved values are cached in the module namespace, so later lookups are
    ordinary attribute reads.
    """
    if name == "timeout":
        if sys.version_info >= (3, 11):  # pragma: no cover - selected by the interpreter
            from asyncio import timeout as value
        else:  # pragma: no cover - selected by the interpreter
            from async_timeout import timeout as value
    elif name == "uuid7":
        if sys.version_info >= (3, 14):  # pragma: no cover - selected by the interpreter
            from uuid import uuid7 as value
        else:  # pragma: no cover - selected by the interpreter
            from uuid6 import uuid7 as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "BaseExceptionGroup",
    "ExceptionGroup",
    "Self",
    "StrEnum",
    "TypeAliasType",
    "UTC",
    "add_note",
    "owned_event_loop",
]
#: ``timeout`` and ``uuid7`` are intentionally omitted from ``__all__``; the
#: module-level ``__getattr__`` above resolves them so importing asyncio and
#: uuid6 stays optional.
