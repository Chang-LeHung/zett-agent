"""Expose Markdown workflows as bounded, non-executable tool reads."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated

from pydantic import Field

from ..tools.base import AgentTool, tool
from .pages import text_page
from .repo import Workflow
from .results import ToolTextPage


def load_text_tool(workflow: Workflow) -> AgentTool:
    """Build a reader bound to one workflow version, without eval/exec."""

    @tool(name=workflow.name)
    async def read_workflow(
        offset: Annotated[int, Field(strict=True, ge=0)] = 0,
        max_chars: Annotated[int, Field(strict=True, ge=1, le=4000)] = 2000,
    ) -> ToolTextPage:
        """Read a reusable Markdown workflow; follow its steps using available tools.

        Args:
            offset: Character offset in the workflow.
            max_chars: Maximum text characters, from 1 to 4000.

        Guidelines:
            - Read relevant steps before following them; continue with next_offset and compare content_id.
        """
        return text_page(workflow.name, workflow.content, offset, max_chars, workflow.path)

    return replace(read_workflow, description=f"Read workflow: {workflow.description}")
