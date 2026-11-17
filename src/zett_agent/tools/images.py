"""Recognising the image files ``read`` hands to a model, by their first bytes, never by their name.

``view_image`` also covers an image that is not in the working directory at all: an HTTP(S) URL. It is the one
reader here that needs the network, so its client is imported only when a URL is asked for.
"""

from __future__ import annotations

import asyncio
import ssl
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field

from ..messages import ImageBytesSource, ImageContent
from .base import tool

#: The largest image file a reader returns.
MAX_IMAGE_BYTES = 10_000_000

#: Bytes needed to recognise every signature below.
SIGNATURE_BYTES = 12


def image_media_type(header: bytes) -> str | None:
    """Return the media type of a PNG, JPEG, GIF or WebP from the first bytes of a file, or ``None`` for anything else.

    A file's name says what its author meant; its first bytes say what it is, and a model rejects a picture
    announced as the wrong type. A file that is not one of the four is text as far as ``read`` is concerned.

    Args:
        header: At least :data:`SIGNATURE_BYTES` bytes from the start of the file (fewer for a short file).
    """
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _read_local_image(path: str, max_bytes: int) -> bytes:
    """Read one bounded image from the local filesystem."""
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Image is not a regular file: {path}")
    with target.open("rb") as source:
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes}-byte limit")
    return data


async def _read_remote_image(url: str, max_bytes: int) -> bytes:
    """Download one bounded image from an HTTP(S) URL."""
    # Imported here because only remote URLs need them; local files should not
    # pay for an HTTP client and a trust store at import time.
    import httpx
    import truststore

    try:
        async with httpx.AsyncClient(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            trust_env=True,
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length is not None and content_length.isdecimal() and int(content_length) > max_bytes:
                    raise ValueError(f"Image exceeds the {max_bytes}-byte limit")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError(f"Image exceeds the {max_bytes}-byte limit")
    except ValueError:
        raise
    except httpx.HTTPError as error:
        raise ValueError(f"Unable to download image from URL: {error}") from error
    return bytes(data)


@tool
async def view_image(
    path: str,
    max_bytes: Annotated[int, Field(ge=1, le=20_000_000)] = MAX_IMAGE_BYTES,
) -> ImageContent:
    """View a PNG, JPEG, GIF, or WebP from a local path or HTTP(S) URL.

    Args:
        path: HTTP(S) URL, absolute path, or path relative to the current working directory.
        max_bytes: Maximum encoded image size; oversized files are rejected.

    Returns:
        The image itself, which only a vision-capable model can read.

    Snippet:
        view_image(path="https://example.com/chart.png")

    Guidelines:
        - Use this tool to inspect screenshots and images with a vision-capable model.
        - It accepts a local path or an HTTP(S) URL; remote images are downloaded with redirects enabled.
        - Use read_file for text. PDF and SVG files are not supported by this tool.
        - Images are returned intact; resize large images before viewing them.
    """
    parsed = urlparse(path)
    if parsed.scheme in {"http", "https"}:
        data = await _read_remote_image(path, max_bytes)
        alt_text = Path(parsed.path).name or parsed.netloc or "remote image"
    elif "://" in path:
        raise ValueError(f"Unsupported image URL scheme: {parsed.scheme or 'unknown'}")
    else:
        target = Path(path).expanduser().resolve()
        data = await asyncio.to_thread(_read_local_image, path, max_bytes)
        alt_text = target.name
    media_type = image_media_type(data)
    if media_type is None:
        raise ValueError("Unsupported image signature; expected PNG, JPEG, GIF, or WebP")
    return ImageContent(
        source=ImageBytesSource(data=data, media_type=media_type),
        alt_text=alt_text,
    )


__all__ = ["MAX_IMAGE_BYTES", "SIGNATURE_BYTES", "image_media_type", "view_image"]
