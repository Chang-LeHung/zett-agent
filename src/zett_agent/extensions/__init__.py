"""Built-in extensions.

Each extension imports straight from the package, for example
``from zett_agent.extensions import CodingExtension``. A name loads its defining module the first
time it is used, so importing one extension never imports the SDKs or storage engines the others
need, and importing this package imports none of them. The defining modules, such as
``zett_agent.extensions.coding``, stay importable for the events, storage records, and helpers the
package does not list.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .._lazy import lazy_exports as _lazy_exports

_MODULES = {
    "agents_md": ("AgentsMdExtension",),
    "ask_user": ("AskUserExtension",),
    "async_subagent": ("AsyncSubAgentExtension",),
    "base": ("AgentExtension",),
    "bash": ("BashExtension",),
    "coding": ("CodingExtension",),
    "compaction": ("CompactionExtension",),
    "external": ("ExternalEventExtension",),
    "file_system": (
        "DeleteFileExtension",
        "EditExtension",
        "FileSystemExtension",
        "GlobExtension",
        "GrepExtension",
        "ReadExtension",
        "ReadFileExtension",
        "ViewImageExtension",
        "WriteFileExtension",
    ),
    "goal": ("GoalExtension",),
    "internal_message": ("InternalMessageExtension",),
    "jsonl": ("JSONLExtension",),
    "mcp": ("McpExtension",),
    "model_request_trace": ("ModelRequestTraceExtension",),
    "persistence": ("BaseSessionPersistenceExtension",),
    "plan_mode": ("PlanModeExtension",),
    "refusal_replay": ("RefusalReplayExtension",),
    "session_persistence": ("SessionPersistenceExtension",),
    "shell_approval": ("ShellApprovalExtension",),
    "skill": ("SkillExtension",),
    "sqlite": ("SQLiteSessionExtension",),
    "steering": ("SteeringExtension",),
    "subagent": ("SubAgentExtension",),
    "todo": ("TodoWriteExtension",),
    "tool": ("ToolExtension",),
    "tool_guidelines": ("ToolGuidelinesExtension",),
    "tool_search": ("ToolSearchExtension",),
    "usage_activity": ("UsageActivityExtension",),
}
_EXPORTS = {name: f"{__name__}.{module}" for module, names in _MODULES.items() for name in names}

__all__ = tuple(_EXPORTS)
__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS)

if _TYPE_CHECKING:
    from .agents_md import AgentsMdExtension as AgentsMdExtension
    from .ask_user import AskUserExtension as AskUserExtension
    from .async_subagent import AsyncSubAgentExtension as AsyncSubAgentExtension
    from .base import AgentExtension as AgentExtension
    from .bash import BashExtension as BashExtension
    from .coding import CodingExtension as CodingExtension
    from .compaction import CompactionExtension as CompactionExtension
    from .external import ExternalEventExtension as ExternalEventExtension
    from .file_system import DeleteFileExtension as DeleteFileExtension
    from .file_system import EditExtension as EditExtension
    from .file_system import FileSystemExtension as FileSystemExtension
    from .file_system import GlobExtension as GlobExtension
    from .file_system import GrepExtension as GrepExtension
    from .file_system import ReadExtension as ReadExtension
    from .file_system import ReadFileExtension as ReadFileExtension
    from .file_system import ViewImageExtension as ViewImageExtension
    from .file_system import WriteFileExtension as WriteFileExtension
    from .goal import GoalExtension as GoalExtension
    from .internal_message import InternalMessageExtension as InternalMessageExtension
    from .jsonl import JSONLExtension as JSONLExtension
    from .mcp import McpExtension as McpExtension
    from .model_request_trace import ModelRequestTraceExtension as ModelRequestTraceExtension
    from .persistence import BaseSessionPersistenceExtension as BaseSessionPersistenceExtension
    from .plan_mode import PlanModeExtension as PlanModeExtension
    from .refusal_replay import RefusalReplayExtension as RefusalReplayExtension
    from .session_persistence import SessionPersistenceExtension as SessionPersistenceExtension
    from .shell_approval import ShellApprovalExtension as ShellApprovalExtension
    from .skill import SkillExtension as SkillExtension
    from .sqlite import SQLiteSessionExtension as SQLiteSessionExtension
    from .steering import SteeringExtension as SteeringExtension
    from .subagent import SubAgentExtension as SubAgentExtension
    from .todo import TodoWriteExtension as TodoWriteExtension
    from .tool import ToolExtension as ToolExtension
    from .tool_guidelines import ToolGuidelinesExtension as ToolGuidelinesExtension
    from .tool_search import ToolSearchExtension as ToolSearchExtension
    from .usage_activity import UsageActivityExtension as UsageActivityExtension
