"""Harness result types have one defining module and stable public exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

import zett_agent.jit_harness as jit_harness
from zett_agent.jit_harness import results
from zett_agent.jit_harness.errors import ToolErrorRecord
from zett_agent.jit_harness.profiling import ToolCallProfile, ToolProfileStats


@pytest.mark.parametrize(
    "name",
    [
        "ToolMutationResult",
        "ToolCatalogEntry",
        "ToolInspection",
        "ToolListPage",
        "ToolDefinitionView",
        "ToolTextPage",
        "ToolCallDetails",
        "ToolCallsPage",
        "ToolErrorsPage",
    ],
)
def test_public_result_types_keep_their_identity_and_defining_module(name):
    result_type = getattr(results, name)
    assert result_type.__module__ == "zett_agent.jit_harness.results"
    assert getattr(jit_harness, name) is result_type
    assert name in jit_harness.__all__


def test_rename_has_no_catalog_compatibility_module():
    assert not Path(results.__file__).with_name("catalog.py").exists()
    assert jit_harness.ToolStatus is results.ToolStatus


@pytest.mark.parametrize("status", ["loaded", "unloaded", "missing", "unavailable"])
def test_tool_status_accepts_the_consistent_unload_vocabulary(status):
    assert TypeAdapter(results.ToolStatus).validate_python(status) == status


def test_tool_status_rejects_the_old_unload_state():
    with pytest.raises(ValidationError):
        TypeAdapter(results.ToolStatus).validate_python("offloaded")


def test_result_shapes_stay_minimal_and_drop_internal_telemetry():
    """Guards the deliberate simplification: no sequence, session, or version IDs."""
    assert set(ToolCallProfile.model_fields) == {
        "call_id",
        "tool_name",
        "origin",
        "status",
        "duration_ms",
        "input_chars",
        "output_chars",
        "caller_tool_name",
        "error_type",
    }
    assert set(ToolProfileStats.model_fields) == {
        "tool_name",
        "call_count",
        "completed_count",
        "failed_count",
        "cancelled_count",
        "rejected_count",
        "average_duration_ms",
        "average_input_chars",
        "average_output_chars",
    }
    assert set(ToolErrorRecord.model_fields) == {
        "tool_name",
        "error_type",
        "message",
        "message_truncated",
        "occurrences",
        "last_seen_at",
        "last_call_id",
        "status",
        "caller_tool_name",
    }
    assert results.ToolInspection.__required_keys__ | results.ToolInspection.__optional_keys__ == {
        "name",
        "origin",
        "status",
        "description",
        "source",
        "entrypoint",
        "parameters",
        "guidelines",
        "files",
    }
    # Optional context stays optional without weakening the required core fields.
    assert results.ToolCallDetails.__required_keys__ == {
        "call_id",
        "status",
        "duration_ms",
        "input_chars",
        "output_chars",
    }
    assert results.ToolCallDetails.__optional_keys__ == {"caller_tool_name", "error_type"}
    for removed in ("implementation_id", "session_id", "request_id", "parent_call_id", "sequence", "error_id"):
        assert removed not in ToolCallProfile.model_fields
        assert removed not in ToolErrorRecord.model_fields
