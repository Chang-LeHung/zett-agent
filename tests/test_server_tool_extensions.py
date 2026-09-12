"""Tests for provider-specific hosted-tool extension registration."""

from __future__ import annotations

import pytest

from zett_agent import (
    Agent,
    AgentConfig,
    AgentExtension,
    AnthropicServerToolExtension,
    AssistantMessage,
    DeepSeekServerToolExtension,
    GoogleServerToolExtension,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    OpenAIServerToolExtension,
    OpenRouterServerToolExtension,
    ServerToolDefinition,
    ServerToolExtension,
)


class RecordingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))


@pytest.mark.parametrize(
    ("extension", "tool_type", "configuration"),
    [
        (OpenAIServerToolExtension(), "web_search", {}),
        (DeepSeekServerToolExtension(), "web_search", {}),
        (
            GoogleServerToolExtension(),
            ("google_search", "url_context"),
            ({}, {}),
        ),
        (
            AnthropicServerToolExtension(),
            ("web_search_20260318", "web_fetch_20260318"),
            ({"name": "web_search"}, {"name": "web_fetch"}),
        ),
        (
            OpenRouterServerToolExtension(),
            ("openrouter:web_search", "openrouter:web_fetch"),
            ({}, {}),
        ),
    ],
)
async def test_provider_extension_registers_its_default_server_tool(
    extension: AgentExtension,
    tool_type: str | tuple[str, ...],
    configuration: dict[str, object] | tuple[dict[str, object], ...],
) -> None:
    model = RecordingModel()
    agent = await Agent.create(
        model,
        config=AgentConfig(session_id=f"session-{tool_type}"),
        extensions=[extension],
    )

    await agent.run("Use the hosted capability")

    assert len(model.requests) == 1
    expected_types = (tool_type,) if isinstance(tool_type, str) else tool_type
    expected_configuration = (configuration,) if isinstance(configuration, dict) else configuration
    assert tuple(tool.type for tool in model.requests[0].server_tools) == expected_types
    assert tuple(tool.configuration for tool in model.requests[0].server_tools) == expected_configuration


async def test_multiple_google_server_tools_keep_registration_order() -> None:
    model = RecordingModel()
    extension = GoogleServerToolExtension(
        [
            "url_context",
            "code_execution",
        ]
    )
    agent = await Agent.create(model, config=AgentConfig("google-tools"), extensions=[extension])

    await agent.run("Read and calculate")

    assert [tool.type for tool in model.requests[0].server_tools] == ["url_context", "code_execution"]
    assert extension.name == "GoogleServerToolExtension"


async def test_provider_extension_preserves_opaque_configuration() -> None:
    configuration = {
        "parameters": {
            "max_uses": 4,
            "allowed_domains": ["python.org"],
        }
    }
    extension = OpenRouterServerToolExtension([ServerToolDefinition("openrouter:web_search", configuration)])
    model = RecordingModel()
    agent = await Agent.create(model, config=AgentConfig("openrouter-tools"), extensions=[extension])

    await agent.run("Search")

    assert model.requests[0].server_tools[0].configuration == configuration
    assert model.requests[0].server_tools[0].configuration is not configuration


def test_base_server_tool_extension_requires_an_explicit_type() -> None:
    with pytest.raises(ValueError, match="at least one definition"):
        ServerToolExtension()


def test_provider_extension_accepts_new_unrecognized_wire_types() -> None:
    extension = OpenAIServerToolExtension([ServerToolDefinition("future_tool_20270101", {"option": True})])

    assert extension.server_tools[0].type == "future_tool_20270101"
    assert extension.server_tools[0].configuration == {"option": True}
    assert extension.name == "OpenAIServerToolExtension"


def test_anthropic_infers_only_known_required_names() -> None:
    web_fetch = AnthropicServerToolExtension([ServerToolDefinition("web_fetch_20260318")])
    custom = AnthropicServerToolExtension([ServerToolDefinition("advisor_20260301", {"name": "advisor"})])

    assert web_fetch.server_tools[0].configuration == {"name": "web_fetch"}
    assert custom.server_tools[0].configuration == {"name": "advisor"}


def test_extension_rejects_duplicate_server_tool_types_before_a_request() -> None:
    with pytest.raises(ValueError, match="duplicate types"):
        GoogleServerToolExtension(
            [
                ServerToolDefinition("google_search"),
                ServerToolDefinition("google_search", {"exclude_domains": ["example.com"]}),
            ]
        )


def test_explicit_empty_sequence_does_not_enable_provider_default() -> None:
    with pytest.raises(ValueError, match="at least one definition"):
        GoogleServerToolExtension([])


async def test_extension_accepts_mixed_type_strings_and_definitions() -> None:
    model = RecordingModel()
    extension = GoogleServerToolExtension(
        [
            "google_search",
            ServerToolDefinition("file_search", {"file_search_store_names": ["stores/notes"]}),
            "code_execution",
        ]
    )
    agent = await Agent.create(model, config=AgentConfig("mixed-server-tools"), extensions=[extension])

    await agent.run("Search and calculate")

    assert [(tool.type, tool.configuration) for tool in model.requests[0].server_tools] == [
        ("google_search", {}),
        ("file_search", {"file_search_store_names": ["stores/notes"]}),
        ("code_execution", {}),
    ]
