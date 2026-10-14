"""End-to-end Google provider tests against a local Gemini server.

The real ``google-genai`` SDK talks to ``tests/fake_gemini.py`` over HTTP, so
these tests cover the wire format, SSE streaming, error mapping, and the rules
the live service enforces that unit tests with mocked SDK objects cannot reach.
"""

import base64

import pytest

from fake_gemini import API_KEY, SIGNATURE, FakeGemini
from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.messages import ImageBytesSource, ImageContent, TextContent, ToolMessage, UserMessage
from zett_agent.model import ModelEventType, ModelRequest, RetryOptions, ToolDefinition
from zett_agent.providers.base import ProviderAuthError, ProviderResponseError
from zett_agent.providers.google import GoogleProvider
from zett_agent.storage import decode_messages, encode_messages
from zett_agent.tools.base import tool

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
WEATHER = ToolDefinition("get_weather", "Get weather", {"type": "object", "properties": {"city": {"type": "string"}}})
FAST_RETRY = RetryOptions(max_retries=2, base_delay=0.001, max_delay=0.01)


@pytest.fixture
def server():
    with FakeGemini() as running:
        yield running


@pytest.fixture
async def provider_for(server):
    created: list[GoogleProvider] = []

    def build(model: str = "gemini-2.5-flash", **options) -> GoogleProvider:
        options.setdefault("api_key", API_KEY)
        options.setdefault("retry", FAST_RETRY)
        created.append(GoogleProvider(model, base_url=server.url, **options))
        return created[-1]

    yield build
    for provider in created:
        await provider.aclose()


async def collect(provider, request):
    events = [event async for event in provider.stream(request)]
    return events, events[-1].response


@tool(guidelines="Use to look up the weather.")
async def get_weather(city: str) -> str:
    """Get the weather.

    Args:
        city: City name.
    """
    if city == "Atlantis":
        raise RuntimeError("no such city")
    return f"sunny in {city}"


async def test_streams_text_and_usage_over_http(provider_for):
    events, response = await collect(provider_for(), ModelRequest([UserMessage(content="hello there")]))

    assert response.message.content == "Echo: hello there"
    assert response.finish_reason == "STOP"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (11, 5)
    # Text arrived as separate deltas before the completed response.
    assert [event.delta for event in events if event.type is ModelEventType.TEXT_DELTA] == ["Echo: ", "hello there"]


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3-pro-preview"])
async def test_parallel_tool_calls_return_one_function_response_turn(provider_for, server, model):
    provider = provider_for(model)
    agent = await Agent.create(provider, config=AgentRunConfig("parallel"), tools=[get_weather])

    result = await agent.run("weather in Paris and Rome")

    assert 'get_weather={"content": "sunny in Paris"}; get_weather={"content": "sunny in Rome"}' in result.content
    answer = server.requests[-1]["contents"][-1]
    assert answer["role"] == "user"
    # Both results travel in one turn, which the live service requires.
    assert [part["functionResponse"]["name"] for part in answer["parts"]] == ["get_weather", "get_weather"]
    assert len(server.requests) == 2


async def test_failed_tool_is_reported_to_gemini_as_an_error(provider_for, server):
    provider = provider_for()
    agent = await Agent.create(provider, config=AgentRunConfig("failure"), tools=[get_weather])

    await agent.run("weather in Atlantis and Rome")

    responses = [part["functionResponse"]["response"] for part in server.requests[-1]["contents"][-1]["parts"]]
    assert "error" in responses[0] and "no such city" in responses[0]["error"]
    assert responses[1] == {"content": "sunny in Rome"}


async def test_thought_signature_survives_storage_between_turns(provider_for):
    provider = provider_for("gemini-3-pro-preview")
    user = UserMessage(content="weather in Paris and Rome")
    _, first = await collect(provider, ModelRequest([user], tools=(WEATHER,)))
    stored = decode_messages(encode_messages([first.message]))[0]
    assert stored.replay_blocks[0]["thought_signature"] == base64.b64decode(SIGNATURE)

    results = [
        ToolMessage(tool_call_id=call.id, name=call.name, content=f"sunny in {call.arguments['city']}")
        for call in stored.tool_calls
    ]
    # The server rejects the request when the restored turn lost its signature.
    _, second = await collect(provider, ModelRequest([user, stored, *results], tools=(WEATHER,)))

    assert second.message.content.startswith("Results: ")


async def test_a_rejected_key_raises_the_provider_auth_error(provider_for):
    provider = provider_for(api_key="wrong-key")

    with pytest.raises(ProviderAuthError):
        await collect(provider, ModelRequest([UserMessage(content="hi")]))


async def test_transient_server_errors_are_retried_before_any_output(provider_for, server):
    server.failures = [503, 503]

    _, response = await collect(provider_for(), ModelRequest([UserMessage(content="again")]))

    assert response.message.content == "Echo: again"
    assert len(server.requests) == 3


async def test_a_client_error_is_a_provider_error_and_is_not_retried(provider_for, server):
    server.failures = [400]

    with pytest.raises(ProviderResponseError, match="transient failure"):
        await collect(provider_for(), ModelRequest([UserMessage(content="x")]))

    assert len(server.requests) == 1


async def test_extra_headers_and_body_reach_the_server(provider_for, server):
    provider = provider_for(
        extra_headers={"x-tenant": "acme"},
        extra_body={"labels": {"team": "agents"}},
    )

    await collect(provider, ModelRequest([UserMessage(content="hi")]))

    assert server.headers[0]["x-tenant"] == "acme"
    assert server.headers[0]["x-goog-api-key"] == API_KEY
    assert server.requests[0]["labels"] == {"team": "agents"}


async def test_an_image_in_a_user_message_is_sent_as_inline_data(provider_for, server):
    message = UserMessage(
        content=[
            TextContent(text="what is this?"),
            ImageContent(source=ImageBytesSource(data=PNG, media_type="image/png")),
        ]
    )

    _, response = await collect(provider_for(), ModelRequest([message]))

    assert response.message.content == f"I can see 1 image(s) of {len(PNG)} bytes."
    assert server.requests[0]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"


async def test_an_image_returned_by_a_tool_reaches_gemini(provider_for, tmp_path, monkeypatch):
    from zett_agent.messages import AssistantMessage, ToolCall
    from zett_agent.tools.coding import read

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pixel.png").write_bytes(PNG)
    outcome = await read({"path": "pixel.png"})
    assert isinstance(outcome[1], ImageContent)
    request = ModelRequest(
        [
            UserMessage(content="look"),
            AssistantMessage(tool_calls=(ToolCall("1", "read", {"path": "pixel.png"}),)),
            ToolMessage(tool_call_id="1", name="read", content=outcome, success=True),
        ]
    )

    _, response = await collect(provider_for(), request)

    assert response.message.content == f"I can see 1 image(s) of {len(PNG)} bytes."
