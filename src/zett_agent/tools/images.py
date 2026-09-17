"""View encoded image files or remote URLs as model-facing content."""

import asyncio
import ssl
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import httpx
import truststore
from pydantic import Field

from ..messages import ImageBytesSource, ImageContent
from .base import tool


def _image_media_type(data: bytes) -> str:
    """Return the media type for one supported image signature."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image signature; expected PNG, JPEG, GIF, or WebP")


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
    max_bytes: Annotated[int, Field(ge=1, le=20_000_000)] = 10_000_000,
) -> ImageContent:
    """View a PNG, JPEG, GIF, or WebP from a local path or HTTP(S) URL.

    Args:
        path: HTTP(S) URL, absolute path, or path relative to the current working directory.
        max_bytes: Maximum encoded image size; oversized files are rejected.

    Snippet:
        view_image(path="https://example.com/chart.png")

    Guidelines:
        - Use this tool to inspect screenshots and images with a vision-capable model.
        - Use read_file for text. PDF and SVG files are not supported by this tool.
        - Remote images must use HTTP or HTTPS and are downloaded with redirects enabled.
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
    return ImageContent(
        source=ImageBytesSource(data=data, media_type=_image_media_type(data)),
        alt_text=alt_text,
    )
