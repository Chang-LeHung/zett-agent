"""Runtime message codec and lazy access to the SQLite session storage.

Everything a caller can import here is cheap: the message kind codes and the
JSON codec only need the runtime's own message types. ``SQLiteSessionStorage``
resolves on first attribute access, so importing this module never imports
SQLAlchemy unless the storage class is actually requested.
"""

import base64
import json
from collections.abc import Sequence
from dataclasses import asdict
from enum import IntEnum
from functools import cache
from typing import Any

from .extensions.compaction import CompactedMessage
from .messages import AgentMessage, AnyMessage, AssistantMessage, SystemMessage, ToolMessage, UserMessage

#: Session type this runtime stores for the sessions it creates itself.
#: Applications define every other code they need.
_DEFAULT_SESSION_TYPE = 0


class MessageKind(IntEnum):
    """Numeric types for lossless runtime messages and checkpoints."""

    SYSTEM = 1
    USER = 2
    ASSISTANT = 3
    TOOL = 4
    CHECKPOINT = 5
    AGENT = 6


MESSAGE_TYPES = {
    MessageKind.AGENT: AgentMessage,
    MessageKind.SYSTEM: SystemMessage,
    MessageKind.USER: UserMessage,
    MessageKind.ASSISTANT: AssistantMessage,
    MessageKind.TOOL: ToolMessage,
    MessageKind.CHECKPOINT: CompactedMessage,
}


@cache
def _message_adapter(kind: MessageKind) -> Any:
    """Return the cached validator for one message kind.

    Pydantic is imported here so decoding builds each adapter once per process
    instead of once per decoded message, and so importing this module never
    imports pydantic by itself.
    """
    from pydantic import TypeAdapter

    return TypeAdapter(MESSAGE_TYPES[kind])


def encode_messages(messages: Sequence[AnyMessage]) -> str:
    """Encode complete typed messages, including binary images and signed replay blocks."""

    def encode_bytes(value: object) -> object:
        if isinstance(value, bytes):
            return {"__kcs_bytes__": base64.b64encode(value).decode("ascii")}
        raise TypeError(f"Unsupported message value: {type(value).__name__}")

    return json.dumps(
        [
            {
                "kind": int(next(kind for kind, cls in MESSAGE_TYPES.items() if type(message) is cls)),
                "data": asdict(message),
            }
            for message in messages
        ],
        default=encode_bytes,
        ensure_ascii=False,
        allow_nan=False,
    )


def decode_messages(payload: str) -> list[AnyMessage]:
    """Validate stored message envelopes and reconstruct their concrete types."""

    def decode_bytes(value: dict) -> object:
        if set(value) == {"__kcs_bytes__"}:
            return base64.b64decode(value["__kcs_bytes__"], validate=True)
        return value

    return [
        _message_adapter(MessageKind(item["kind"])).validate_python(item["data"])
        for item in json.loads(payload, object_hook=decode_bytes)
    ]


def __getattr__(name: str) -> object:
    """Import SQLAlchemy-backed storage only when someone asks for it.

    The ORM models must be declared at import time, which costs about 90 ms in
    SQLAlchemy imports. Resolving ``SQLiteSessionStorage`` here keeps that cost
    on applications that actually open a database.
    """
    if name == "SQLiteSessionStorage":
        from ._sqlite_storage import SQLiteSessionStorage

        return SQLiteSessionStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
