import pytest

from zett_agent import (
    AgentMessage,
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


def test_message_attributes_are_independent_and_available_to_every_role():
    first = UserMessage(content="first")
    second = UserMessage(content="second")
    first.attributes["source"] = "clipboard"

    assert first.attributes == {"source": "clipboard"}
    assert second.attributes == {}
    assert SystemMessage(content="system", attributes={"scope": "request"}).attributes == {"scope": "request"}
    assert AssistantMessage(content="answer", attributes={"trace_id": "trace"}).attributes == {"trace_id": "trace"}
    assert ToolMessage(tool_call_id="call", name="tool", content="result", attributes={"cached": True}).attributes == {
        "cached": True
    }


def test_message_storage_and_restore_flags_are_role_aware():
    system = SystemMessage(content="Rules")
    user = UserMessage(content="Question")

    assert system.persist is True
    assert system.include_in_messages is False
    assert user.persist is True
    assert user.include_in_messages is True
    assert UserMessage(content="Transient", persist=False).persist is False


def test_message_fields_are_direct_and_role_specific():
    call = ToolCall("c1", "lookup", {"query": "agents"})
    messages = [
        SystemMessage(content="Rules"),
        UserMessage(content="Question"),
        AssistantMessage(reasoning="Checking", tool_calls=(call,)),
        ToolMessage(content="Result", tool_call_id=call.id, name=call.name),
        AgentMessage(content="Check the result"),
    ]
    assert [message.role for message in messages] == list(MessageRole)
    assert messages[1].content == "Question"
    assert messages[2].tool_calls == (call,)
    assert messages[3].success is True
    with pytest.raises(TypeError):
        UserMessage(content="Question", tool_calls=(call,))


def test_message_protocol_does_not_accept_extension_context_data():
    with pytest.raises(TypeError):
        UserMessage(content="Question", metadata={"source": "clipboard"})
    with pytest.raises(TypeError):
        AssistantMessage(content="Answer", tags={"kind": "internal"})


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
