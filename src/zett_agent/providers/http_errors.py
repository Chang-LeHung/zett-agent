"""Read what a provider's HTTP failure actually said, including when it is a web page.

An API client usually sees an error as JSON with a message in it. In front of a real
provider there is often something else: a reverse proxy, a gateway, a firewall, a captive
portal. When it fails it answers with an HTML page, a bare line of text, or nothing, and
all of those used to be reduced to "Provider stream request failed". The HTTP status, the
provider's own explanation, and the fact that it was a web page at all are what a person
needs to fix the problem, so they are read here and put in the error message.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus

__all__ = ["HttpFailure", "failure_message", "read_failure"]

#: Longest stretch of a provider's own words that goes into an error message.
_MESSAGE_LIMIT = 300
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HEADING = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_BLOCKS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class HttpFailure:
    """What an HTTP error response or a response that was not an API reply turned out to say.

    Attributes:
        reason: The provider's own explanation when the body was a JSON error; None when it
            was not.
        is_html: Whether the body was a web page.
        title: The page's title or first heading, for an HTML body.
        preview: The start of the body as plain text, whitespace collapsed and limited in
            length, for the cases where there is no ``reason``.
    """

    reason: str | None
    is_html: bool
    title: str | None
    preview: str


def _plain(text: str) -> str:
    """Return ``text`` as one line without markup or entities."""
    return _SPACE.sub(" ", html.unescape(_TAGS.sub(" ", _BLOCKS.sub(" ", text)))).strip()


def _shorten(text: str, limit: int = _MESSAGE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _json_reason(parsed: object) -> str | None:
    """Find the message in the error shapes providers use.

    OpenAI and most copies send ``{"error": {"message": ...}}``; Anthropic nests the same
    object; some gateways send ``{"message": ...}``, ``{"detail": ...}`` or a bare string.
    The SDKs hand over the inner ``error`` object, so that is accepted as well.
    """
    if isinstance(parsed, str):
        return parsed.strip() or None
    if not isinstance(parsed, Mapping):
        return None
    inner = parsed.get("error")
    if isinstance(inner, str) and inner.strip():
        return inner.strip()
    for source in (inner, parsed):
        if isinstance(source, Mapping):
            for key in ("message", "detail", "msg", "error_description"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def read_failure(*, content_type: str | None, text: str, parsed: object = None) -> HttpFailure:
    """Work out what a failed or unexpected response body is.

    Args:
        content_type: The ``Content-Type`` header, or None.
        text: The body as text; may be empty.
        parsed: The body already decoded as JSON, when the caller has it.
    """
    kind = (content_type or "").lower()
    head = text.lstrip()[:200].lower()
    is_html = "html" in kind or head.startswith(("<!doctype html", "<html", "<head", "<body"))
    if is_html:
        found = _TITLE.search(text) or _HEADING.search(text)
        title = _plain(found.group(1)) if found else None
        return HttpFailure(None, True, _shorten(title, 120) if title else None, _shorten(_plain(text)))
    reason = _json_reason(parsed)
    if reason is None and text.strip().startswith(("{", "[")):
        try:
            reason = _json_reason(json.loads(text))
        except ValueError:
            reason = None
    return HttpFailure(_shorten(_plain(reason)) if reason else None, False, None, _shorten(_plain(text)))


def failure_message(provider: str, status_code: int, failure: HttpFailure) -> str:
    """Return one sentence that says what the provider answered and what that usually means.

    Args:
        provider: How to name the provider in the sentence.
        status_code: The HTTP status of the response.
        failure: What :func:`read_failure` made of the body.
    """
    try:
        phrase = HTTPStatus(status_code).phrase
    except ValueError:
        phrase = ""
    if failure.is_html:
        titled = f' ("{failure.title}")' if failure.title else ""
        return (
            f"{provider} answered HTTP {status_code} with a web page{titled} instead of an API response. "
            "A proxy, gateway, firewall or login page in front of the API usually does this; "
            "check the base_url and the network path to it."
        )
    if failure.reason:
        return f"{provider} returned HTTP {status_code}: {failure.reason}"
    if failure.preview:
        return f"{provider} returned HTTP {status_code}: {failure.preview}"
    return f"{provider} returned HTTP {status_code}" + (f" ({phrase})" if phrase else " with an empty body")
