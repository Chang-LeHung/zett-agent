"""Read JSON that came off the wire without trusting its shape.

Providers receive JSON they do not control: a field may be missing, null, or of another type. These
helpers return a value of the type asked for, or the neutral one, so the code that reads a stream is
free of ``isinstance`` ladders and a hook written for one service reads like the default.
"""

from __future__ import annotations

from ..json_types import JsonValue

__all__ = ["JsonObject", "read_int", "read_object", "read_objects", "read_str"]

#: One JSON object as it goes to or comes from the wire.
JsonObject = dict[str, JsonValue]


def read_str(value: JsonValue | None) -> str:
    """Return ``value`` if it is a string, else an empty one: what a missing or null field reads as."""
    return value if isinstance(value, str) else ""


def read_int(value: JsonValue | None) -> int:
    """Return ``value`` if it is an integer (a boolean is not one), else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def read_object(value: JsonValue | None) -> JsonObject:
    """Return ``value`` if it is an object, else an empty one."""
    return value if isinstance(value, dict) else {}


def read_objects(value: JsonValue | None) -> list[JsonObject]:
    """Return the objects in ``value`` if it is an array; anything else in it is skipped."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
