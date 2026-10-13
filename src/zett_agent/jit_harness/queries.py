"""Small, purpose-specific model-facing views of a harness's full data."""

from __future__ import annotations

import json

from .enums import ToolOrigin
from .pages import (
    MAX_RESPONSE_CHARS as MAX_RESPONSE_CHARS,
)
from .pages import (
    fits_response,
    text_page,
    validate_page_bounds,
    validate_text_bounds,
)
from .profiling import ToolProfileStats
from .results import (
    ToolCallDetails,
    ToolCallsPage,
    ToolCatalogEntry,
    ToolDefinitionView,
    ToolErrorsPage,
    ToolErrorView,
    ToolListPage,
    ToolTextPage,
)
from .runtime import JITHarness


def _preview(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ToolQueries:
    """Internal: bounded model-facing views over one harness.

    Page counts are maximums, not promises: pages stop at the JSON character
    budget and expose a continuation offset. Definition schemas are never cut
    into invalid JSON: oversized schemas are explicitly omitted and available
    through the exact schema-fragment reader.
    """

    def __init__(self, harness: JITHarness) -> None:
        self.harness = harness

    def list_tools(self, offset: int = 0, limit: int = 20) -> ToolListPage:
        validate_page_bounds(offset, limit)
        catalog = self.harness.list_tools()
        page: ToolListPage = {"tools": [], "total_tools": len(catalog), "next_offset": None}
        for entry in catalog[offset : offset + limit]:
            bounded: ToolCatalogEntry = {**entry, "description": _preview(entry["description"], 200)}
            page["tools"].append(bounded)
            page["next_offset"] = offset + len(page["tools"])
            if not fits_response(page):
                page["tools"].pop()
                break
        end = offset + len(page["tools"])
        page["next_offset"] = end if end < len(catalog) else None
        return page

    def inspect_tool(self, name: str) -> ToolDefinitionView:
        self._known(name)
        definition = self.harness.definition(name)
        view: ToolDefinitionView = {
            "name": definition["name"],
            "origin": definition["origin"],
            "status": definition["status"],
            "description": _preview(definition["description"], 500),
        }
        if "entrypoint" in definition:
            view["entrypoint"] = definition["entrypoint"]
        if "files" in definition:
            view["files"] = definition["files"]
        if "parameters" in definition:
            view["parameters"] = definition["parameters"]
            if not fits_response(view):
                view.pop("parameters")
                view["parameters_omitted"] = True
                view["parameters_chars"] = len(self._schema(name))
        if not fits_response(view):
            raise ValueError("Tool definition metadata exceeds the response budget; read its source instead")
        # Guidelines already accompany callable tools in the model request;
        # source, profiling, snippets and history are separate query concerns.
        return view

    def read_source(self, name: str, path: str | None = None, offset: int = 0, max_chars: int = 2000) -> ToolTextPage:
        """Read exact Python or Markdown source of one file, including unloaded definitions."""
        self._known(name)
        return self._text_page(name, self.harness.source(name, path), offset, max_chars, path)

    def read_schema(self, name: str, offset: int = 0, max_chars: int = 2000) -> ToolTextPage:
        self._known(name)
        return self._text_page(name, self._schema(name), offset, max_chars, None)

    def _schema(self, name: str) -> str:
        definition = self.harness.definition(name)
        if "parameters" not in definition:
            raise ValueError(f"Missing base tool schema: {name}")
        return json.dumps(definition["parameters"], ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _text_page(name: str, content: str, offset: int, max_chars: int, path: str | None = None) -> ToolTextPage:
        return text_page(name, content, offset, max_chars, path)

    def stats(self, name: str) -> ToolProfileStats:
        # Aggregates do not need to sort/materialize the full call history.
        self._known(name)
        return self.harness.profile_stats(name)

    def calls(self, name: str, offset: int = 0, limit: int = 5) -> ToolCallsPage:
        validate_page_bounds(offset, limit)
        self._known(name)
        report = self.harness.profile(name, offset=offset, limit=limit)
        page: ToolCallsPage = {"name": name, "calls": [], "total_calls": report.total_calls, "next_offset": None}
        for call in report.calls:
            item: ToolCallDetails = {
                "call_id": call.call_id,
                "status": call.status,
                "duration_ms": call.duration_ms,
                "input_chars": call.input_chars,
                "output_chars": call.output_chars,
            }
            if call.caller_tool_name is not None:
                item["caller_tool_name"] = call.caller_tool_name
            if call.error_type is not None:
                item["error_type"] = call.error_type
            page["calls"].append(item)
            page["next_offset"] = offset + len(page["calls"])
            if not fits_response(page):
                page["calls"].pop()
                if not page["calls"]:
                    raise ValueError("Call metadata exceeds the response budget; use the application profile API")
                break
        end = offset + len(page["calls"])
        page["next_offset"] = end if end < report.total_calls else None
        return page

    def errors(self, name: str | None = None, offset: int = 0, limit: int = 3, max_chars: int = 512) -> ToolErrorsPage:
        validate_page_bounds(offset, limit)
        validate_text_bounds(max_chars)
        report = self.harness.tool_errors(name, offset=offset, limit=limit)
        page: ToolErrorsPage = {"errors": [], "total_matches": report.total_matches, "next_offset": None}
        for error in report.errors:
            item: ToolErrorView = {
                "tool_name": error.tool_name,
                "error_type": error.error_type,
                "message": error.message[:max_chars] if error.message is not None else None,
                "message_truncated": error.message_truncated
                or (error.message is not None and len(error.message) > max_chars),
                "occurrences": error.occurrences,
                "last_call_id": error.last_call_id,
            }
            # Even one message full of escaped control chars must fit.
            while not fits_response(
                {"errors": [item], "total_matches": report.total_matches, "next_offset": offset + 1}
            ):
                if not item["message"]:
                    raise ValueError("Error metadata exceeds the response budget")
                item["message"] = item["message"][: len(item["message"]) // 2]
                item["message_truncated"] = True
            page["errors"].append(item)
            page["next_offset"] = offset + len(page["errors"])
            if not fits_response(page):
                page["errors"].pop()
                break
        end = offset + len(page["errors"])
        page["next_offset"] = end if end < report.total_matches else None
        return page

    def _known(self, name: str) -> None:
        if self.harness.origin(name) is ToolOrigin.UNKNOWN:
            raise ValueError("Unknown tool; inspect list_tools before querying measurements")
