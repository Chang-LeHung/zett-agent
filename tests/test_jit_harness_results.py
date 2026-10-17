"""Harness result types have one defining module and stable public exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

import zett_agent.jit_harness as jit_harness
from zett_agent.jit_harness import results
from zett_agent.jit_harness.enums import ToolOrigin
from zett_agent.jit_harness.errors import ToolErrorRecord
from zett_agent.jit_harness.profiling import ToolCallProfile, ToolProfileStats

#: Result types of the removed inspection tiers; nothing may advertise them again.
REMOVED_RESULTS = (
    "ToolInspection",
    "ToolListPage",
    "ToolDefinitionView",
    "ToolCallDetails",
    "ToolCallsPage",
    "ToolErrorsPage",
)

#: One workflow fragment page whose cursor ends on the last character.
PAGE_CONTENT = "# Investigate a failing test\n"


@pytest.mark.parametrize(
    "name",
    [
        "ToolMutationResult",
        "ToolCatalogEntry",
        "RepoResult",
        "ToolTextPage",
    ],
)
def test_public_result_types_keep_their_identity_and_defining_module(name):
    result_type = getattr(results, name)
    assert result_type.__module__ == "zett_agent.jit_harness.results"
    assert getattr(jit_harness, name) is result_type
    assert name in jit_harness.__all__


@pytest.mark.parametrize("name", REMOVED_RESULTS)
def test_removed_result_types_are_gone_from_the_module_and_the_facade(name):
    """The repository reports paths and statuses, not snapshots or call pages."""
    assert not hasattr(results, name)
    assert not hasattr(jit_harness, name)
    assert name not in jit_harness.__all__


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
    assert set(results.ToolMutationResult.__annotations__) == {"name", "status"}
    assert set(results.ToolCatalogEntry.__annotations__) == {"name", "origin", "status", "description"}
    assert set(results.RepoResult.__annotations__) == {"repo", "path", "status", "tools"}
    assert set(results.ToolTextPage.__annotations__) == {
        "name",
        "path",
        "content",
        "total_chars",
        "next_offset",
        "content_id",
    }
    for removed in ("implementation_id", "session_id", "request_id", "parent_call_id", "sequence", "error_id"):
        assert removed not in ToolCallProfile.model_fields
        assert removed not in ToolErrorRecord.model_fields


@pytest.mark.parametrize(
    ("result_type", "payload"),
    [
        (results.ToolMutationResult, {"name": "repo_load", "status": "loaded"}),
        (
            results.ToolCatalogEntry,
            {
                "name": "repo_load",
                "origin": ToolOrigin.MANAGEMENT,
                "status": "loaded",
                "description": "Import the repository and register what it defines.",
            },
        ),
        (
            results.RepoResult,
            {"repo": "repo", "path": "/sessions/session/repo", "status": "created", "tools": []},
        ),
        (
            results.ToolTextPage,
            {
                "name": "investigate",
                "path": "workflows/investigate.md",
                "content": PAGE_CONTENT,
                "total_chars": len(PAGE_CONTENT),
                "next_offset": None,
                "content_id": "0" * 64,
            },
        ),
    ],
    ids=["mutation", "catalog-entry", "repository", "text-page"],
)
def test_every_declared_result_field_is_required(result_type, payload):
    """Each acknowledgement is complete, so the model can act without another call."""
    adapter = TypeAdapter(result_type)
    assert adapter.validate_python(payload) == payload
    for field in payload:
        with pytest.raises(ValidationError):
            adapter.validate_python({name: value for name, value in payload.items() if name != field})


def test_repository_status_is_not_the_catalog_unload_vocabulary():
    """A repository reports what happened to it; a catalog row reports unload state."""
    repo = {"repo": "repo", "path": "/sessions/session/repo", "tools": []}
    for status in ("created", "loaded", "unloaded"):
        assert TypeAdapter(results.RepoResult).validate_python({**repo, "status": status})["status"] == status
    with pytest.raises(ValidationError):
        TypeAdapter(results.RepoResult).validate_python({**repo, "status": "missing"})
    entry = {"name": "repo_load", "origin": ToolOrigin.MANAGEMENT, "description": "Import the repository."}
    for status in ("loaded", "unloaded", "missing", "unavailable"):
        assert TypeAdapter(results.ToolCatalogEntry).validate_python({**entry, "status": status})["status"] == status
    with pytest.raises(ValidationError):
        TypeAdapter(results.ToolCatalogEntry).validate_python({**entry, "status": "created"})
    with pytest.raises(ValidationError):
        TypeAdapter(results.ToolMutationResult).validate_python({"name": "repo_load", "status": "created"})
