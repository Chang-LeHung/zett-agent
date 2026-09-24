"""Validate JSON-only data shared by AgentRunContext and Raw Log persistence.

This module defines the data shape accepted by ``Agent.run(metadata=..., tags=...)``.
Those values are extension context, not model-message fields. The Agent validates
them before lifecycle hooks run, and SQLite validates them again before encoding
them into each Raw Log record::

    caller Mapping
         |
         | json_object()
         v
    AgentRunContext.metadata / AgentRunContext.tags
         |
         | extensions may inspect or modify the dictionaries
         v
    RawLogMessageModel.metadata_json / tags_json

Validation is deliberately stricter than ``json.dumps(default=str)``. Unsupported
Python values are rejected rather than converted into lossy strings, so restored
context has the same JSON structure that was supplied originally.
"""

from collections.abc import Mapping
from math import isfinite

from ._compat import TypeAliasType

# Recursive, provider-neutral values that JSON can represent without custom
# encoders. Tuples, sets, bytes, datetime objects, NaN, and infinity are excluded.
JsonValue = TypeAliasType(
    "JsonValue",
    "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]",
)


def json_object(
    value: Mapping[str, JsonValue] | None,
    *,
    field_name: str,
    nonempty_keys: bool = False,
) -> dict[str, JsonValue]:
    """Return a validated top-level copy of one JSON object.

    Args:
        value: Caller-owned mapping to validate. ``None`` becomes an empty
            dictionary. Nested values must conform to ``JsonValue``.
        field_name: Human-readable field label included in validation errors,
            such as ``"Context metadata"`` or ``"Message tags"``.
        nonempty_keys: Require every top-level key to contain non-whitespace
            characters. Tags enable this because an empty tag cannot classify
            anything; metadata only requires string keys.

    Returns:
        A new top-level dictionary. Nested lists and dictionaries are validated
        but are not deep-copied.

    Raises:
        ValueError: A key violates the selected rule, or a nested value is not
            an exact JSON value. No value is converted automatically.

    This function is used at both trust boundaries: Agent creation protects
    extensions from malformed caller data, while storage validation protects
    the Raw Log if an extension mutated the context later.
    """
    result = dict(value or {})
    if any(not isinstance(key, str) or (nonempty_keys and not key.strip()) for key in result):
        qualifier = "non-empty string" if nonempty_keys else "string"
        raise ValueError(f"{field_name} keys must be {qualifier}s")
    try:
        _validate_json_value(result)
    except ValueError as error:
        raise ValueError(f"{field_name} must be JSON serializable") from error
    return result


def _validate_json_value(value: object) -> None:
    """Recursively verify one value against the ``JsonValue`` definition.

    Scalars terminate recursion. Finite floats are accepted, while NaN and
    infinity are rejected because they are not valid JSON numbers. Lists are
    checked in order, dictionaries require string keys, and every nested value
    is validated recursively. The function returns ``None`` on success and
    raises ``ValueError`` at the first unsupported value.
    """
    match value:
        case None | bool() | int() | str():
            return
        case float() if isfinite(value):
            return
        case list():
            for item in value:
                _validate_json_value(item)
        case dict():
            if any(not isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            for item in value.values():
                _validate_json_value(item)
        case _:
            raise ValueError("Value must contain only JSON-compatible types")
