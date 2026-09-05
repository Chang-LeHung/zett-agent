import pytest

from kcs_agent import (
    AssistantMessage,
    ImageBytesSource,
    ImageContent,
    ImageDetail,
    ImageUrlSource,
    MessageRole,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def test_message_fields_are_direct_and_role_specific():
    call = ToolCall("c1", "lookup", {"query": "agents"})
    messages = [
        SystemMessage(content="Rules"),
        UserMessage(content="Question"),
        AssistantMessage(reasoning="Checking", tool_calls=(call,)),
        ToolMessage(content="Result", tool_call_id=call.id, name=call.name),
    ]
    assert [message.role for message in messages] == list(MessageRole)
    assert messages[1].content == "Question"
    assert messages[2].tool_calls == (call,)
    assert messages[3].success is True
    with pytest.raises(TypeError):
        UserMessage(content="Question", tool_calls=(call,))


def test_user_text_and_image_parts():
    image = ImageContent(ImageUrlSource("https://example.com/a.png"), ImageDetail.HIGH)
    message = UserMessage(content=[TextContent("Explain"), image, TextContent("Focus on arrows")])
    assert message.parts == [TextContent("Explain"), image, TextContent("Focus on arrows")]
    assert message.text == "Explain\nFocus on arrows"
    assert UserMessage(content="Hi").parts == [TextContent("Hi")]


@pytest.mark.parametrize(
    "source",
    [
        lambda: ImageUrlSource("  "),
        lambda: ImageUrlSource("file:///tmp/private.png"),
        lambda: ImageBytesSource(b"", "image/png"),
        lambda: ImageBytesSource(b"image", "application/octet-stream"),
    ],
)
def test_invalid_image_sources_are_rejected(source):
    with pytest.raises(ValueError):
        source()
