"""Time-ordered identifiers for persisted runtime records."""


def new_uuid7() -> str:
    """Return a time-ordered UUIDv7 string for persisted runtime records."""
    from ._compat import uuid7

    return str(uuid7())
