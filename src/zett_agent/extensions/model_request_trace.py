"""Attach provider-visible model request metadata to the resulting response."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from copy import deepcopy
from typing import Literal, TypedDict, cast

from ..agent import AgentRunContext
from ..json_types import JsonValue
from ..model import ModelEvent, ModelEventType, ModelRequest, ServerToolDefinition, ToolDefinition
from .base import AgentExtension, ModelRequestNext

MODEL_REQUEST_TRACE_ATTRIBUTE = "model_request_trace"
MODEL_REQUEST_TRACE_SCHEMA_VERSION: Literal[1] = 1


class ToolDefinitionTrace(TypedDict):
    """JSON-safe local tool definition persisted beside one model response."""

    name: str
    description: str
    parameters: dict[str, JsonValue]
    deferred: bool


class ServerToolDefinitionTrace(TypedDict):
    """JSON-safe provider-hosted tool definition persisted beside one response."""

    type: str
    configuration: dict[str, JsonValue]


class ModelRequestTrace(TypedDict):
    """Exact tool metadata supplied to one provider request."""

    schema_version: Literal[1]
    tools: list[ToolDefinitionTrace]
    server_tools: list[ServerToolDefinitionTrace]


def _tool_definition_payload(tool: ToolDefinition) -> ToolDefinitionTrace:
    """Project one local tool into JSON-safe trace metadata."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": cast(dict[str, JsonValue], deepcopy(dict(tool.parameters))),
        "deferred": tool.deferred,
    }


def _server_tool_definition_payload(tool: ServerToolDefinition) -> ServerToolDefinitionTrace:
    """Project one provider-hosted tool into JSON-safe trace metadata."""
    return {
        "type": tool.type,
        "configuration": cast(dict[str, JsonValue], deepcopy(dict(tool.configuration))),
    }


def model_request_trace(request: ModelRequest) -> ModelRequestTrace:
    """Return the exact local and provider tool definitions sent to the model."""
    return {
        "schema_version": MODEL_REQUEST_TRACE_SCHEMA_VERSION,
        "tools": [_tool_definition_payload(tool) for tool in request.tools],
        "server_tools": [_server_tool_definition_payload(tool) for tool in request.server_tools],
    }


class ModelRequestTraceExtension(AgentExtension):
    """Persist exact tool definitions through the assistant response attributes.

    The extension is intentionally placed closest to the provider in the
    middleware chain. It snapshots the final request, then enriches the
    assistant response before the runtime appends and persists that message.
    This keeps request metadata out of the provider-visible message content.
    """

    async def on_model_request(
        self,
        context: AgentRunContext,
        request: ModelRequest,
        call_next: ModelRequestNext,
    ) -> AsyncIterator[ModelEvent]:
        """Attach one immutable request snapshot to the completed response."""
        snapshot = model_request_trace(request)
        async with aclosing(call_next(request)) as events:
            async for event in events:
                if event.type is ModelEventType.RESPONSE and event.response is not None:
                    event.response.message.attributes[MODEL_REQUEST_TRACE_ATTRIBUTE] = deepcopy(snapshot)
                yield event
