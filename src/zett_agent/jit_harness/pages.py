"""Shared output budgets and exact fragments for workflow readers."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from .results import ToolTextPage

MAX_RESPONSE_CHARS = 8000

#: Character measurements stop at the model-facing budget: past it, extra
#: precision costs dispatch time without changing a decision, and a reported
#: value equal to this limit means "at least this many".
MAX_MEASURED_CHARS = MAX_RESPONSE_CHARS


def fits_response(value: Any) -> bool:
    # Final-page `null` can cost three characters more than a numeric cursor.
    return len(json.dumps(value, ensure_ascii=False)) <= MAX_RESPONSE_CHARS - 4


def validate_offset(offset: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")


def validate_text_bounds(max_chars: int) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= 4000:
        raise ValueError("max_chars must be between 1 and 4000")


def text_page(name: str, content: str, offset: int, max_chars: int, path: str) -> ToolTextPage:
    """Return an exact fragment whose serialized JSON fits the response budget."""
    validate_offset(offset)
    validate_text_bounds(max_chars)
    page: ToolTextPage = {
        "name": name,
        "path": path,
        "content": "",
        "total_chars": len(content),
        "next_offset": None,
        "content_id": sha256(content.encode()).hexdigest(),
    }
    # Quotes, control characters and newlines can multiply JSON length.
    low, high = 0, len(content[offset : offset + max_chars])
    while low < high:
        middle = (low + high + 1) // 2
        page["content"] = content[offset : offset + middle]
        page["next_offset"] = offset + middle
        if fits_response(page):
            low = middle
        else:
            high = middle - 1
    page["content"] = content[offset : offset + low]
    end = offset + low
    page["next_offset"] = end if end < len(content) else None
    if not fits_response(page) or end < len(content) and low == 0:
        raise ValueError("Text metadata exceeds the response budget")
    return page
