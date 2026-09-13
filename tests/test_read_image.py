"""Real file reads, model-loop delivery, and lossless image tool history."""

import base64

import pytest

from zett_agent import (
    Agent,
    AgentRunConfig,
    AssistantMessage,
    FileSystemExtension,
    ImageContent,
    ModelEvent,
    ModelResponse,
    SQLiteSessionStorage,
    ToolCall,
    ToolMessage,
    UserMessage,
    read_image,
)
from zett_agent.providers.base import _message_to_openai_payload
from zett_agent.providers.responses import responses_input
from zett_agent.providers.tool_images import expand_tool_images
from zett_agent.storage import decode_messages, encode_messages

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


async def test_image_tool_reaches_next_model_request_and_round_trips(tmp_path):
    path = tmp_path / "pixel.png"
    path.write_bytes(PNG)
    requests = []

    class Model:
        async def stream(self, request):
            requests.append(request)
            message = (
                AssistantMessage(tool_calls=(ToolCall("image-1", "read_image", {"path": str(path)}),))
                if len(requests) == 1
                else AssistantMessage(content="seen")
            )
            yield ModelEvent.completed(ModelResponse(message))

    agent = await Agent.create(
        Model(), config=AgentRunConfig("image-test"), extensions=[FileSystemExtension(read_only=True)]
    )
    assert (await agent.run("Inspect image")).content == "seen"
    result = next(message for message in requests[-1].messages if isinstance(message, ToolMessage))
    assert result.success
    assert isinstance(result.content[0], ImageContent)
    assert result.content[0].source.data == PNG
    assert decode_messages(encode_messages([result])) == [result]


async def test_image_validation_and_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "image.dat").write_bytes(PNG)
    assert (await read_image({"path": "image.dat"})).source.media_type == "image/png"
    with pytest.raises(ValueError, match="limit"):
        await read_image({"path": "image.dat", "max_bytes": 4})
    (tmp_path / "fake.png").write_text("not an image")
    with pytest.raises(ValueError, match="signature"):
        await read_image({"path": "fake.png"})
    with pytest.raises(FileNotFoundError):
        await read_image({"path": "missing.png"})


async def test_parallel_tool_images_follow_all_tool_results(tmp_path):
    path = tmp_path / "pixel.png"
    path.write_bytes(PNG)
    image = await read_image({"path": str(path)})
    messages = [
        ToolMessage(tool_call_id="1", name="read_image", content=[image]),
        ToolMessage(tool_call_id="2", name="other", content="done"),
    ]
    wire = expand_tool_images(messages)
    assert [message.role for message in wire] == ["tool", "tool", "user"]
    assert isinstance(wire[-1], UserMessage)
    assert _message_to_openai_payload(wire[-1])["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,")
    output = responses_input(messages, provider="openai", model="vision")
    assert output[-1]["content"][-1]["type"] == "input_image"
    assert messages[0].content == [image]


async def test_image_result_survives_sqlite_reopen(tmp_path):
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(PNG)
    image = await read_image({"path": str(image_path)})
    result = ToolMessage(tool_call_id="image", name="read_image", content=[image])
    database = tmp_path / "history.sqlite3"
    storage = SQLiteSessionStorage(database)
    try:
        await storage.append("session", "request", result)
    finally:
        storage.close()
    reopened = SQLiteSessionStorage(database)
    try:
        view = await reopened.load("session")
        assert view.messages == [result]
        assert view.raw_tail[0].message == result
    finally:
        reopened.close()
