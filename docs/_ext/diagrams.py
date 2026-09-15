"""Render source-readable ASCII diagrams as Mermaid in generated API pages."""

from dataclasses import dataclass

from sphinx.errors import ExtensionError


@dataclass(frozen=True)
class Diagram:
    """One browser rendering selected by a ``zett-diagram`` source marker."""

    caption: str
    source: str


DIAGRAMS = {
    "agent-state": Diagram(
        "A minimal request phase cycle",
        """flowchart LR
    created[CREATED] --> loading[LOADING_CONTEXT]
    loading --> ready[READY]
    ready --> generating[GENERATING]
    generating --> ready""",
    ),
    "agent-loop": Diagram(
        "The model-tool loop",
        """flowchart TD
    user[UserMessage] --> model[model]
    model -- ToolCall --> tool[tool]
    tool -- ToolMessage --> model
    model -- final answer --> assistant[AssistantMessage]""",
    ),
    "model-stream": Diagram(
        "A successful model stream",
        """flowchart LR
    text[text deltas] --> response[one terminal ModelResponse]
    reasoning[reasoning deltas] --> response
    calls[tool-call JSON deltas] --> response
    response --> complete[complete message, usage, finish reason]""",
    ),
    "session-view": Diagram(
        "Restoring one session view",
        """flowchart LR
    raw[Raw Log: immutable UI history]
    snapshot[Snapshot: compacted context + boundary]
    tail[Raw Log after boundary]
    view[SessionView]
    snapshot --> view
    raw --> tail
    tail --> view""",
    ),
    "ask-user": Diagram(
        "Ask-user pause and resume protocol",
        """sequenceDiagram
    participant Model
    participant Agent
    participant Extension
    participant UI
    Model->>Agent: ask_user ToolCall
    Agent->>Extension: before_tool() + context.emit()
    Extension-->>UI: AskUserEvent
    Note over Agent,Extension: waiting before TOOL_STARTED
    UI->>Extension: ExternalEvent via accept()
    Extension-->>Agent: resume
    Agent->>Extension: TOOL_STARTED and ask_user()
    Extension-->>Model: ToolMessage""",
    ),
    "plan-mode": Diagram(
        "Model-proposed Plan Mode entry",
        """sequenceDiagram
    participant Model
    participant Agent
    participant Extension
    participant UI
    Model->>Agent: enter_plan_mode(reason)
    Agent->>Extension: before_tool() + context.emit()
    Extension-->>UI: EnterPlanModeEvent
    UI->>Extension: ExternalEvent(approved)
    Extension-->>Agent: resume tool lifecycle
    Agent->>Extension: enter_plan_mode()
    Extension-->>Agent: PlanModeEnteredEvent
    Agent->>Model: next step with Plan Mode prompt""",
    ),
    "extension-lifecycle": Diagram(
        "Agent and extension lifecycle",
        """flowchart TD
    request[NEW REQUEST] --> setup["SETUP<br/>on_tool() / on_state()<br/>on_message() / append UserMessage<br/>before_run()"]
    setup --> premodel["PRE-MODEL<br/>before_model()<br/>optional compaction via context.emit()"]
    premodel --> model["MODEL STEP<br/>append AssistantMessage<br/>after_model() / MODEL_COMPLETED"]
    model --> calls[Has tool calls?]
    calls -- yes --> tools["TOOL STEP<br/>before_tool()<br/>execute / after_tool()<br/>append ToolMessage / TOOL_*"]
    tools --> premodel
    calls -- no --> inbox[Steering or internal input?]
    inbox -- yes --> continuation[Append selected input]
    continuation --> premodel
    inbox -- no --> success["SUCCESS<br/>after_run() / on_success()<br/>COMPLETED / RUN_COMPLETED"]
    active[ANY ACTIVE STAGE] -. exception .-> failed["FAILED<br/>on_error() / re-raise"]
    active -. cancellation .-> cancelled["CANCELLED<br/>RunCancelledEvent / re-raise"]
    request -. active scope .-> active
    external[ExternalEvent] -. emit_external_event .-> accept[accept hooks]""",
    ),
}


def render_source_diagrams(app, what, name, obj, options, lines):
    """Replace a marked ASCII block with its browser-oriented Mermaid form."""
    del app, what, obj, options
    index = 0
    while index < len(lines):
        marker = lines[index].strip()
        if not marker.startswith(".. zett-diagram::"):
            index += 1
            continue
        key = marker.removeprefix(".. zett-diagram::").strip()
        diagram = DIAGRAMS.get(key)
        if diagram is None:
            raise ExtensionError(f"Unknown zett diagram {key!r} in {name}")
        end = index + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
            end += 1
        replacement = [".. mermaid::", f"   :caption: {diagram.caption}", ""]
        replacement.extend(f"   {line}" for line in diagram.source.splitlines())
        replacement.append("")
        lines[index:end] = replacement
        index += len(replacement)


def setup(app):
    """Register ASCII-to-Mermaid conversion for autodoc content only."""
    app.connect("autodoc-process-docstring", render_source_diagrams)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
