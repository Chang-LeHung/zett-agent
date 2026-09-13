"""Adapt image tool results for APIs whose tool role only accepts text."""

from collections.abc import Sequence
from dataclasses import replace

from ..messages import AnyMessage, ImageContent, TextContent, ToolMessage, UserMessage


def expand_tool_images(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Send images after the complete tool-result group, preserving call pairing.

    Storage retains the original multimodal ToolMessage. Only provider wire
    requests use companion user image blocks, which all vision adapters support.
    Deferring these blocks until every adjacent tool result has been emitted is
    essential for parallel calls: user input cannot interrupt a tool-result group.
    """
    result: list[AnyMessage] = []
    pending: list[TextContent | ImageContent] = []
    for message in messages:
        if not isinstance(message, ToolMessage) and pending:
            result.append(UserMessage(content=pending))
            pending = []
        if isinstance(message, ToolMessage) and isinstance(message.content, list):
            text = "\n".join(part.text for part in message.content if isinstance(part, TextContent))
            images = [part for part in message.content if isinstance(part, ImageContent)]
            result.append(replace(message, content=text or "Image result supplied in the following image blocks."))
            if images:
                pending.extend([TextContent(text=f"Tool result: {message.name} ({message.tool_call_id})"), *images])
        else:
            result.append(message)
    if pending:
        result.append(UserMessage(content=pending))
    return result
