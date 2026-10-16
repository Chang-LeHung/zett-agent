"""The rules, each reading one parsed module and reporting where it breaks them.

A rule looks at syntax only, so it needs neither the package imported nor its dependencies installed. The
three here are the ones neither Ruff nor mypy enforces:

``SC001``
    A module-level name bound to a type expression is declared as an alias. ``Foo = dict[str, int]`` is an
    ordinary assignment to a type checker's eyes in many editors, so it is not highlighted or offered as a
    type; ``Foo: TypeAlias = dict[str, int]`` or ``Foo = TypeAliasType("Foo", dict[str, int])`` is.
``SC002``
    ``Any`` does not appear in the provider modules. It switches the checker off for everything it
    touches, and the wire formats those modules read and write can be described exactly.
``SC003``
    A module does not import an underscore name from another module of the package. Such a name is
    private to its module; needing it elsewhere means it should be public, with a name and a contract.

A line ends the conversation with ``# static-check: ignore[SC001] reason``; the reason is required by
convention, not by the tool.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TypeAlias

PACKAGE = PurePosixPath("src/zett_agent")
PROVIDERS = PACKAGE / "providers"
IGNORE_MARKER = re.compile(r"#\s*static-check:\s*ignore\[([A-Z0-9, ]+)\]")

#: Generic names whose subscript, at module level on the right of ``=``, is a type and not a value.
GENERIC_TYPES = frozenset(
    {
        "Annotated",
        "AsyncGenerator",
        "AsyncIterator",
        "Awaitable",
        "Callable",
        "Collection",
        "Coroutine",
        "Iterable",
        "Iterator",
        "Literal",
        "Mapping",
        "MutableMapping",
        "Optional",
        "Sequence",
        "Union",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
        "type",
    }
)
BUILTIN_TYPES = frozenset({"bool", "bytes", "float", "int", "object", "str"})
#: ``JsonValue`` and ``ChatMessagePayload`` but not ``READ`` (a constant) or ``X`` (a type variable).
CLASS_STYLE_NAME = re.compile(r"^_?[A-Z][a-z0-9]\w*$")


@dataclass(frozen=True, slots=True)
class Violation:
    """One place a rule is broken.

    ``key`` is what a baseline counts: the same for every occurrence of one allowed thing, so a baseline
    can say "this file may keep three of these".
    """

    rule: str
    path: str
    line: int
    key: str
    message: str


Rule: TypeAlias = Callable[[str, ast.Module], Iterator[Violation]]


def in_providers(path: str) -> bool:
    """Whether ``path`` (relative to the repository root, with ``/``) is a provider module."""
    return PurePosixPath(path).is_relative_to(PROVIDERS)


def module_level_statements(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements that run when the module is imported, looking inside ``if`` and ``try`` but not defs."""
    for statement in body:
        yield statement
        if isinstance(statement, ast.If):
            yield from module_level_statements(statement.body)
            yield from module_level_statements(statement.orelse)
        elif isinstance(statement, ast.Try):
            yield from module_level_statements(statement.body)
            for handler in statement.handlers:
                yield from module_level_statements(handler.body)
            yield from module_level_statements(statement.orelse)
            yield from module_level_statements(statement.finalbody)


def is_type_expression(node: ast.expr) -> bool:
    """Whether ``node`` can only be a type: a subscripted generic, or ``|`` between types."""
    if isinstance(node, ast.Subscript):
        base = node.value
        name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
        return name in GENERIC_TYPES
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return all(_is_union_member(side) for side in (node.left, node.right))
    return False


def _is_union_member(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.Name):
        return node.id in BUILTIN_TYPES or CLASS_STYLE_NAME.match(node.id) is not None
    return is_type_expression(node)


def undeclared_type_alias(path: str, tree: ast.Module) -> Iterator[Violation]:
    """SC001: ``Name = <type expression>`` at module level without ``TypeAlias`` or ``TypeAliasType``."""
    for statement in module_level_statements(tree.body):
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1):
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and CLASS_STYLE_NAME.match(target.id) and is_type_expression(statement.value):
            yield Violation(
                "SC001",
                path,
                statement.lineno,
                path,
                f'{target.id} is a type alias; declare it as `{target.id} = TypeAliasType("{target.id}", ...)` '
                f"(the package's form, from `_compat`) or, outside the package, `{target.id}: TypeAlias = ...`",
            )


def explicit_any(path: str, tree: ast.Module) -> Iterator[Violation]:
    """SC002: ``Any`` written in a provider module, in an annotation, an alias or a ``cast``."""
    if not in_providers(path):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute):
            written = (node.id if isinstance(node, ast.Name) else node.attr) == "Any"
        else:
            written = isinstance(node, ast.Call) and _casts_to_any(node)
        if written and isinstance(node, ast.expr):
            yield Violation(
                "SC002",
                path,
                node.lineno,
                path,
                "`Any` in a provider module; describe the shape with a TypedDict, a Protocol or the SDK's own type",
            )


def _casts_to_any(node: ast.Call) -> bool:
    """Whether ``node`` is ``cast("... Any ...", value)``, whose type is a string and so not a Name."""
    if not node.args:
        return False
    function = node.func
    name = (
        function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
    )
    first = node.args[0]
    return (
        name == "cast"
        and isinstance(first, ast.Constant)
        and isinstance(first.value, str)
        and re.search(r"\bAny\b", first.value) is not None
    )


def private_name_import(path: str, tree: ast.Module) -> Iterator[Violation]:
    """SC003: ``from .other import _name`` where ``other`` is a module of this package.

    Only the package itself is held to this. Tests import private names on purpose, to test them.
    """
    if not PurePosixPath(path).is_relative_to(PACKAGE):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        ours = node.level > 0 or node.module == "zett_agent" or node.module.startswith("zett_agent.")
        if not ours:
            continue
        for imported in node.names:
            if imported.name.startswith("_") and not imported.name.startswith("__"):
                source = "." * node.level + node.module
                yield Violation(
                    "SC003",
                    path,
                    node.lineno,
                    f"{path}: from {source} import {imported.name}",
                    f"{imported.name} is private to {source}; make it public there or do not use it here",
                )


RULES: dict[str, Rule] = {
    "SC001": undeclared_type_alias,
    "SC002": explicit_any,
    "SC003": private_name_import,
}


def check_source(path: str, source: str) -> list[Violation]:
    """Every violation in one module, except on lines that carry an ignore marker for that rule."""
    lines = source.splitlines()
    tree = ast.parse(source)
    found: list[Violation] = []
    for rule in RULES.values():
        for violation in rule(path, tree):
            if not _ignored(lines, violation):
                found.append(violation)
    return sorted(found, key=lambda violation: (violation.line, violation.rule))


def _ignored(lines: Sequence[str], violation: Violation) -> bool:
    if not 1 <= violation.line <= len(lines):
        return False
    marker = IGNORE_MARKER.search(lines[violation.line - 1])
    return marker is not None and violation.rule in {part.strip() for part in marker.group(1).split(",")}
