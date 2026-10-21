"""Lazy re-exports for the facade packages (PEP 562).

A facade package lists ``name -> defining module`` and gets a module ``__getattr__`` and
``__dir__`` from :func:`lazy_exports`. Importing the package imports nothing: the defining module
loads on the first access to one of its names, then the name is cached on the package. Each facade
also repeats its imports under ``TYPE_CHECKING`` so type checkers and editors see the real objects.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from importlib import import_module


def lazy_exports(package: str, exports: Mapping[str, str]) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Return the ``__getattr__`` and ``__dir__`` of ``package``.

    Args:
        package: Dotted name of the facade package, normally ``__name__``.
        exports: Maps each public name to the dotted module that defines it.
    """

    def module_getattr(name: str) -> object:
        module = exports.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(module), name)
        vars(sys.modules[package])[name] = value
        return value

    def module_dir() -> list[str]:
        return sorted({*vars(sys.modules[package]), *exports})

    return module_getattr, module_dir
