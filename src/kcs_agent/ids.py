from uuid6 import uuid7


def new_uuid7() -> str:
    """Return a time-ordered UUIDv7 string for persisted runtime records."""
    return str(uuid7())
