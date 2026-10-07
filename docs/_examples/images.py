"""Send typed image bytes to a deterministic model without network access."""

import asyncio
import base64
from collections.abc import AsyncIterator

from zett_agent.client import create_agent
from zett_agent.messages import AssistantMessage, ImageBytesSource, ImageContent, TextContent, UserMessage
from zett_agent.model import ModelEvent, ModelRequest, ModelResponse, RetryOptions

# A complete 1x1 PNG; no image is read from the user's filesystem.
PIXEL = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII=")


class ImageModel:
    retry = RetryOptions(max_retries=0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        message = request.messages[-1]
        assert isinstance(message, UserMessage)
        assert isinstance(message.parts[0], TextContent)
        assert isinstance(message.parts[1], ImageContent)
        assert isinstance(message.parts[1].source, ImageBytesSource)
        assert message.parts[1].source.data == PIXEL
        assert message.text == "Describe this image."
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="Received text and image in order.")))


async def main() -> None:
    client = await create_agent(ImageModel())
    message = UserMessage(
        content=[
            TextContent("Describe this image."),
            ImageContent(source=ImageBytesSource(data=PIXEL, media_type="image/png")),
        ]
    )
    reply = await client.run(message)
    assert reply.content == "Received text and image in order."
    print(reply.content)


if __name__ == "__main__":
    asyncio.run(main())
