"""Read encoded image files without converting them to model-facing JSON."""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from ..messages import ImageBytesSource, ImageContent
from .base import tool


@tool
def read_image(
    path: str,
    max_bytes: Annotated[int, Field(ge=1, le=20_000_000)] = 10_000_000,
) -> ImageContent:
    """Read a PNG, JPEG, GIF, or WebP file as model-visible image content.

    Args:
        path: Absolute path or path relative to the current working directory.
        max_bytes: Maximum encoded file size; oversized files are rejected.

    Snippet:
        read_image(path="screenshots/page.png")

    Guidelines:
        - Use this tool to inspect screenshots and images with a vision-capable model.
        - Use read_file for text. PDF and SVG files are not supported by this tool.
        - Images are returned intact; resize large images before reading them.
    """
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Image is not a regular file: {path}")
    with target.open("rb") as source:
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes}-byte limit")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        media_type = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise ValueError("Unsupported image signature; expected PNG, JPEG, GIF, or WebP")
    return ImageContent(source=ImageBytesSource(data=data, media_type=media_type), alt_text=target.name)
