"""ZettCode component composition on the internal differential TUI framework."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Mapping
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic

from wcwidth import wcwidth
from zett_agent import AgentEvent, AgentEventDispatcher, AgentRunConfig, ToolMessage

from .runtime import ZettCodeRuntime
from .tui_framework import (
    Application,
    BoxChild,
    Component,
    EventType,
    HBox,
    InputEvent,
    Rule,
    ScrollableText,
    Span,
    Style,
    Text,
    TextInput,
    TextLine,
    VBox,
)

SLASH_COMMANDS = ("/new", "/sessions", "/use", "/clear", "/help", "/quit", "/exit")

GREEN = "#79b88b"
LIGHT_GREEN = "#9bddad"
MUTED = "#747d77"
DETAIL = "#a3ada6"
FOREGROUND = "#e6e9e7"
THINKING_BACKGROUND = "#26342b"
ERROR = "#dc8178"
MAX_STORED_TOOL_OUTPUT = 64_000
TOOL_PREVIEW_ROWS = 5
TOOL_EXPANDED_ROWS = 40
MAX_MARKDOWN_TABLE_ROWS = 100

STYLE_NORMAL = Style(foreground=FOREGROUND)
STYLE_MUTED = Style(foreground=MUTED)
STYLE_DETAIL = Style(foreground=DETAIL)
STYLE_GREEN = Style(foreground=GREEN, bold=True)
STYLE_USER = Style(foreground="#d6ddd8", bold=True)
STYLE_THINKING = Style(foreground="#dce8df", background=THINKING_BACKGROUND, bold=True)
STYLE_THINKING_BODY = Style(foreground="#d1ddd4", background=THINKING_BACKGROUND)
STYLE_ERROR = Style(foreground=ERROR, bold=True)


@dataclass(frozen=True, slots=True)
class LineMetadata:
    """Interaction identity attached to one rendered transcript line."""

    thinking_id: int | None = None
    thinking_header: bool = False
    tool_id: int | None = None
    tool_header: bool = False


@dataclass(slots=True)
class TranscriptEntry:
    """One semantic terminal block kept in AgentClient event order."""

    id: int
    kind: str
    call_id: str = ""
    text: str = ""
    title: str = ""
    detail: str = ""
    status: str = ""
    expanded: bool = False
    started_at: float | None = None
    duration_seconds: float | None = None


@dataclass(slots=True)
class Transcript:
    """Structured transcript independent from terminal rendering mechanics."""

    entries: list[TranscriptEntry] = field(default_factory=list)
    _next_id: int = 1
    animation_frame: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines(120))

    def append(self, value: str) -> None:
        self._add("raw", text=value)

    def clear(self) -> None:
        self.entries.clear()

    def begin_turn(self, prompt: str) -> None:
        self._add("user", text=prompt)

    def start_thinking(self) -> TranscriptEntry:
        current = self._last("thinking")
        if current is not None and current.status == "running":
            return current
        return self._add("thinking", title="Thinking", status="running", started_at=monotonic())

    def append_thinking(self, delta: str) -> None:
        self.start_thinking().text += delta

    def complete_thinking(self) -> None:
        current = self._last("thinking")
        if current is None or current.status != "running":
            return
        current.status = "completed"
        if current.started_at is not None:
            current.duration_seconds = monotonic() - current.started_at

    def append_answer(self, delta: str) -> None:
        current = self.entries[-1] if self.entries else None
        if current is None or current.kind != "answer":
            current = self._add("answer")
        current.text += delta

    def start_tool(self, call_id: str, name: str, arguments: Mapping[str, object]) -> None:
        self.complete_thinking()
        self._add(
            "tool",
            call_id=call_id,
            title=name,
            detail=_arguments_preview(arguments),
            status="running",
            started_at=monotonic(),
        )

    def complete_tool(self, call_id: str, output: str, *, status: str = "completed") -> None:
        entry = self._tool(call_id)
        if entry is None:
            return
        entry.text = _limit_stored_output(output)
        entry.status = status
        if entry.started_at is not None:
            entry.duration_seconds = monotonic() - entry.started_at

    def start_server_tool(self, call_id: str, name: str) -> None:
        self.complete_thinking()
        self._add("server_tool", call_id=call_id, title=name, status="running", started_at=monotonic())

    def complete_server_tool(self, call_id: str, output: str, *, status: str = "completed") -> None:
        entry = next(
            (item for item in reversed(self.entries) if item.kind == "server_tool" and item.call_id == call_id),
            None,
        )
        if entry is None:
            return
        entry.text = _limit_stored_output(output)
        entry.status = status
        if entry.started_at is not None:
            entry.duration_seconds = monotonic() - entry.started_at

    def toggle_latest_thinking(self) -> bool:
        entry = self._last("thinking")
        if entry is None:
            return False
        entry.expanded = not entry.expanded
        return True

    def toggle_thinking(self, entry_id: int) -> bool:
        entry = self._entry(entry_id)
        if entry is None or entry.kind != "thinking":
            return False
        entry.expanded = not entry.expanded
        return True

    def toggle_tool(self, entry_id: int) -> bool:
        entry = self._entry(entry_id)
        if entry is None or entry.kind not in ("tool", "server_tool") or entry.status == "running" or not entry.text:
            return False
        entry.expanded = not entry.expanded
        return True

    def collapse_thinking_except(self, entry_id: int | None) -> bool:
        changed = False
        for entry in self.entries:
            if entry.kind == "thinking" and entry.expanded and entry.id != entry_id:
                entry.expanded = False
                changed = True
        return changed

    def lines(self, width: int) -> list[TextLine]:
        """Render semantic blocks into rich logical lines for ScrollableText."""
        result: list[TextLine] = []

        def blank() -> None:
            result.append(TextLine())

        def plain(value: str, style: Style = STYLE_NORMAL, metadata: object | None = None) -> None:
            for line in value.split("\n"):
                result.append(TextLine((Span(line, style),), metadata))

        for entry in self.entries:
            match entry.kind:
                case "welcome":
                    plain(entry.text, Style(foreground="#9fb6a6"))
                case "user":
                    blank()
                    prompt_lines = entry.text.splitlines() or [""]
                    plain(f"❯ {prompt_lines[0]}", STYLE_USER)
                    for line in prompt_lines[1:]:
                        plain(f"  {line}", STYLE_USER)
                case "thinking":
                    blank()
                    marker = (
                        _activity_icon(self.animation_frame)
                        if entry.status == "running"
                        else ("▾" if entry.expanded else "▸")
                    )
                    timing = _duration(entry.duration_seconds) if entry.status == "completed" else "working"
                    metadata = LineMetadata(entry.id, thinking_header=True)
                    style = STYLE_THINKING if entry.expanded else STYLE_GREEN
                    if entry.status == "running":
                        result.append(
                            TextLine(
                                _activity_spans(
                                    f"  {marker} Thinking  {timing}",
                                    self.animation_frame,
                                    background=style.background,
                                ),
                                metadata,
                            )
                        )
                    else:
                        plain(f"  {marker} Thinking  {timing}", style, metadata)
                    if entry.expanded:
                        body_metadata = LineMetadata(entry.id)
                        for line in entry.text.splitlines() or ["Waiting for reasoning…"]:
                            plain(f"    {line}", STYLE_THINKING_BODY, body_metadata)
                case "tool" | "server_tool":
                    blank()
                    symbol = (
                        _activity_icon(self.animation_frame)
                        if entry.status == "running"
                        else {"completed": "✓", "failed": "×", "skipped": "–"}.get(entry.status, "●")
                    )
                    timing = f"  {_duration(entry.duration_seconds)}" if entry.duration_seconds is not None else ""
                    style = STYLE_ERROR if entry.status == "failed" else STYLE_GREEN
                    metadata = LineMetadata(tool_id=entry.id, tool_header=True)
                    disclosure = (
                        "" if entry.status == "running" or not entry.text else (" ▾" if entry.expanded else " ▸")
                    )
                    label = f"  {symbol} {entry.title}{entry.detail}{timing}{disclosure}"
                    if entry.status == "running":
                        result.append(TextLine(_activity_spans(label, self.animation_frame), metadata))
                    else:
                        plain(label, style, metadata)
                    output = "Running…" if entry.status == "running" else entry.text
                    if output:
                        row_limit = TOOL_EXPANDED_ROWS if entry.expanded else TOOL_PREVIEW_ROWS
                        output_lines, omitted = _bounded_visual_lines(output, max(8, width - 6), row_limit)
                        plain(f"    └ {output_lines[0]}", STYLE_DETAIL)
                        for line in output_lines[1:]:
                            plain(f"      {line}", STYLE_DETAIL)
                        if omitted:
                            plain(f"      … {omitted} more rows · click tool to expand", STYLE_MUTED)
                case "answer":
                    blank()
                    result.extend(_markdown_block_lines(entry.text, width))
                case _:
                    plain(entry.text, STYLE_DETAIL)
        return result

    def _add(self, kind: str, **values) -> TranscriptEntry:
        entry = TranscriptEntry(self._next_id, kind, **values)
        self._next_id += 1
        self.entries.append(entry)
        return entry

    def _last(self, kind: str) -> TranscriptEntry | None:
        return next((entry for entry in reversed(self.entries) if entry.kind == kind), None)

    def _entry(self, entry_id: int) -> TranscriptEntry | None:
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def _tool(self, call_id: str) -> TranscriptEntry | None:
        return next(
            (entry for entry in reversed(self.entries) if entry.kind == "tool" and entry.call_id == call_id),
            None,
        )


class TUIEventDispatcher(AgentEventDispatcher):
    """Convert AgentClient callbacks into structured transcript blocks."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript

    def begin_turn(self, prompt: str) -> None:
        self.transcript.begin_turn(prompt)

    async def on_compaction_started_event(self, event: AgentEvent) -> None:
        self.transcript.append("\n  ◇ Compacting context…")

    async def on_compaction_completed_event(self, event: AgentEvent) -> None:
        status = "applied" if event.applied else "skipped"
        self.transcript.append(f"  Context compaction {status}")

    async def on_reasoning_started_event(self, event: AgentEvent) -> None:
        self.transcript.start_thinking()

    async def on_reasoning_delta_event(self, event: AgentEvent) -> None:
        self.transcript.append_thinking(event.delta)

    async def on_reasoning_completed_event(self, event: AgentEvent) -> None:
        self.transcript.complete_thinking()

    async def on_text_delta_event(self, event: AgentEvent) -> None:
        self.transcript.complete_thinking()
        self.transcript.append_answer(event.delta)

    async def on_tool_started_event(self, event: AgentEvent) -> None:
        for call in event.tool_calls:
            self.transcript.start_tool(call.id, call.name, call.arguments)

    async def on_tool_completed_event(self, event: AgentEvent) -> None:
        if isinstance(event.message, ToolMessage):
            self.transcript.complete_tool(event.message.tool_call_id, _tool_output(event.message))

    async def on_tool_failed_event(self, event: AgentEvent) -> None:
        for call in event.tool_calls:
            output = str(event.error) if event.error is not None else "Tool failed"
            self.transcript.complete_tool(call.id, output, status="failed")

    async def on_tool_skipped_event(self, event: AgentEvent) -> None:
        for call in event.tool_calls:
            output = _tool_output(event.message) if isinstance(event.message, ToolMessage) else "Skipped"
            self.transcript.complete_tool(call.id, output, status="skipped")

    async def on_server_tool_started_event(self, event: AgentEvent) -> None:
        if event.server_tool_call is not None:
            self.transcript.start_server_tool(event.server_tool_call.id, event.server_tool_call.name)

    async def on_server_tool_completed_event(self, event: AgentEvent) -> None:
        if event.server_tool_result is not None:
            self.transcript.complete_server_tool(
                event.server_tool_result.call_id,
                _serialized_output(event.server_tool_result.output),
            )

    async def on_server_tool_failed_event(self, event: AgentEvent) -> None:
        if event.server_tool_result is not None:
            output = event.server_tool_result.error_code or _serialized_output(event.server_tool_result.output)
            self.transcript.complete_server_tool(event.server_tool_result.call_id, output, status="failed")

    async def on_run_completed_event(self, event: AgentEvent) -> None:
        self.transcript.complete_thinking()


class ZettCodeRoot(Component):
    """Application-specific command routing around reusable framework widgets."""

    def __init__(self, tui: ZettCodeTUI, content: VBox) -> None:
        super().__init__()
        self.tui = tui
        self.content = content

    def layout(self, rect) -> None:
        super().layout(rect)
        self.content.layout(rect)

    def render(self, canvas) -> None:
        self.content.render(canvas)

    def handle(self, event: InputEvent, actions) -> bool:
        if event.type == EventType.KEY:
            match event.key:
                case "ctrl_c":
                    if actions.focused is self.tui.transcript_view:
                        selected = self.tui.transcript_view.selected_text()
                        if selected:
                            actions.copy(selected)
                            self.tui._status = "copied selection"
                        self.tui.transcript_view.clear_selection()
                        actions.focus(self.tui.composer)
                    elif self.tui.busy:
                        self.tui._status = "stopping"
                        assert self.tui._request_task is not None
                        self.tui._request_task.cancel()
                    else:
                        self.tui.composer.clear()
                    return True
                case "ctrl_d" if not self.tui.busy and not self.tui.composer.text:
                    actions.exit()
                    return True
                case "ctrl_l":
                    actions.refresh()
                    return True
                case "ctrl_t":
                    return self.tui.transcript.toggle_latest_thinking()
                case "escape" if actions.focused is self.tui.transcript_view:
                    actions.focus(self.tui.composer)
                    return True
        if event.type in (EventType.TEXT, EventType.PASTE) and actions.focused is self.tui.transcript_view:
            actions.focus(self.tui.composer)
            return self.tui.composer.handle(event, actions)
        return self.content.handle(event, actions)

    def cursor(self) -> tuple[int, int] | None:
        return self.content.cursor()


class ZettCodeTUI:
    """Compose AgentClient, transcript, and internal terminal components."""

    def __init__(self, runtime: ZettCodeRuntime) -> None:
        self.runtime = runtime
        self.transcript = Transcript()
        self.transcript._add(
            "welcome",
            text="  ╭───╮\n  │ ›_│  ZettCode\n  ╰─◆─╯  A focused coding agent\n\n  Type a task below, or /help for commands.",
        )
        self.event_dispatcher = TUIEventDispatcher(self.transcript)
        self.runtime.client.event_dispatcher = self.event_dispatcher
        self._request_task: asyncio.Task[None] | None = None
        self._animation_frame = 0
        self._status = "ready"

        self.transcript_view = ScrollableText(
            self.transcript.lines,
            on_click=self._click_transcript,
            on_hover=self._hover_transcript,
            on_selection=self._selection_copied,
        )
        self.composer = TextInput(self._submit, prompt="› ", max_height=7, completions=SLASH_COMMANDS)
        header = HBox(
            [
                BoxChild(Text(self._header, Style(foreground=LIGHT_GREEN, bold=True)), flex=1),
                BoxChild(Text(self._model, STYLE_MUTED, align="right"), size=32),
            ]
        )
        status = HBox(
            [
                BoxChild(Text(self._status_text, STYLE_MUTED), flex=1),
                BoxChild(Text("^T thinking  PgUp history  ^C stop  ", STYLE_MUTED, align="right"), size=40),
            ]
        )
        content = VBox(
            [
                BoxChild(header, size=1),
                BoxChild(Rule(Style(foreground="#343a36")), size=1),
                BoxChild(self.transcript_view, flex=1),
                BoxChild(Rule(Style(foreground="#60876b")), size=1),
                BoxChild(self.composer, size=lambda width: self.composer.preferred_height(width)),
                BoxChild(status, size=1),
            ]
        )
        self.root = ZettCodeRoot(self, content)
        self.application = Application(self.root)
        self.application.focus(self.composer)

    @property
    def busy(self) -> bool:
        return self._request_task is not None and not self._request_task.done()

    async def run(self) -> None:
        try:
            await self.application.run()
        finally:
            if self._request_task is not None and not self._request_task.done():
                self._request_task.cancel()
                await asyncio.gather(self._request_task, return_exceptions=True)

    def _submit(self, value: str) -> bool:
        if self.busy:
            self._status = "busy — Ctrl-C stops the current request"
            return False
        if value.startswith("/"):
            self._request_task = asyncio.create_task(self._run_command(value))
        else:
            self._request_task = asyncio.create_task(self._run_prompt(value))
        return True

    async def _run_prompt(self, prompt: str) -> None:
        self.event_dispatcher.begin_turn(prompt)
        self._status = "running"
        self.application.invalidate()
        animation = asyncio.create_task(self._animate_activity())
        try:
            async with aclosing(
                self.runtime.client.stream(prompt, config=AgentRunConfig(session_id=self.runtime.session_id))
            ) as events:
                async for _event in events:
                    self.application.invalidate()
        except asyncio.CancelledError:
            self.transcript.complete_thinking()
            self.transcript.append("\n  stopped")
        except Exception as error:
            self.transcript.complete_thinking()
            self.transcript.append(f"\n  error: {error}")
        finally:
            animation.cancel()
            await asyncio.gather(animation, return_exceptions=True)
            self.transcript.animation_frame = 0
            self._status = "ready"
            self.application.invalidate()

    async def _animate_activity(self) -> None:
        """Pulse active rows without coupling animation timing to provider events."""
        while True:
            await asyncio.sleep(0.12)
            self._animation_frame = (self._animation_frame + 1) % 1_000_000
            self.transcript.animation_frame = self._animation_frame
            self.application.invalidate()

    async def _run_command(self, value: str) -> None:
        command, _, argument = value.partition(" ")
        match command:
            case "/new":
                session_id = self.runtime.new_session()
                self.transcript.append(f"\nStarted session {session_id}.")
            case "/use" if argument.strip():
                self.runtime.use_session(argument)
                self.transcript.append(f"\nUsing session {self.runtime.session_id}.")
            case "/use":
                self.transcript.append("\nUsage: /use <session-id>")
            case "/sessions":
                sessions = await self.runtime.persistence.list_sessions(limit=30)
                if not sessions:
                    self.transcript.append("\nNo persisted sessions.")
                for session in sessions:
                    marker = "*" if session.session_id == self.runtime.session_id else " "
                    self.transcript.append(f"\n{marker} {session.session_id}  {session.message_count} messages")
            case "/clear":
                self.transcript.clear()
            case "/help":
                self.transcript.append(
                    "\n/new  new session\n/sessions  list sessions\n/use ID  switch session\n"
                    "/clear  clear display\n/quit  exit\n\n"
                    "Editing follows common Readline keys: Ctrl-A/E, Ctrl-U/K, Ctrl-W, Alt-B/F, history, and undo.\n"
                    "Click Thinking to inspect it; moving the pointer away collapses it. Ctrl-T works without a mouse."
                )
            case "/quit" | "/exit":
                self.application.exit()
            case _:
                self.transcript.append(f"\nUnknown command: {command}. Use /help.")
        self._status = "ready"
        self.application.invalidate()

    def _click_transcript(self, metadata: object | None) -> bool:
        if not isinstance(metadata, LineMetadata):
            return False
        if metadata.thinking_header and metadata.thinking_id is not None:
            return self.transcript.toggle_thinking(metadata.thinking_id)
        if metadata.tool_header and metadata.tool_id is not None:
            return self.transcript.toggle_tool(metadata.tool_id)
        return False

    def _hover_transcript(self, metadata: object | None) -> bool:
        thinking_id = metadata.thinking_id if isinstance(metadata, LineMetadata) else None
        return self.transcript.collapse_thinking_except(thinking_id)

    def _header(self) -> str:
        return f"  ◈ zettcode  {_compact_path(self.runtime.config.workspace)}"

    def _model(self) -> str:
        return f"{self.runtime.config.provider.value}/{self.runtime.config.model}  "

    def _status_text(self) -> str:
        icon = _activity_icon(self._animation_frame) if self.busy else "●"
        return f"  {icon} {self._status}  session {self.runtime.session_id[:8]}"

    def _selection_copied(self, value: str) -> None:
        self._status = f"copied {len(value)} characters"


def _tool_output(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content or "Completed"
    return message.text or "Multimodal result"


def _tool_preview(message: ToolMessage, *, limit: int = 180) -> str:
    value = " ".join(_tool_output(message).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _arguments_preview(arguments: Mapping[str, object], *, limit: int = 90) -> str:
    if not arguments:
        return ""
    value = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    value = value if len(value) <= limit else value[: limit - 1] + "…"
    return f" {value}"


def _serialized_output(value: object) -> str:
    if value is None:
        return "Completed"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "done"
    if seconds < 1:
        return f"{round(seconds * 1000)} ms"
    return f"{seconds:.1f} s"


def _activity_icon(frame: int) -> str:
    """Return a compact green-theme activity glyph with a visible flash phase."""
    return ("✦", "✧", "·", "✧")[(frame // 3) % 4]


def _activity_spans(value: str, frame: int, *, background: str | None = None) -> tuple[Span, ...]:
    """Sweep one narrow shimmer over otherwise stable activity text."""
    spans: list[Span] = []
    center = frame % (len(value) + 8) - 3
    for index, character in enumerate(value):
        distance = abs(index - center)
        wave = math.exp(-(distance**2) / 5.5) if distance < 7 else 0.0
        spans.append(
            Span(
                character,
                Style(
                    foreground=_blend_color("#617268", "#d2f8dc", wave),
                    background=background,
                    bold=wave > 0.68,
                ),
            )
        )
    return tuple(spans)


def _blend_color(start: str, end: str, ratio: float) -> str:
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    values = (round(left + (right - left) * ratio) for left, right in zip(start_rgb, end_rgb, strict=True))
    return "#" + "".join(f"{value:02x}" for value in values)


def _limit_stored_output(value: str) -> str:
    """Bound transcript memory while retaining useful context from both ends."""
    if len(value) <= MAX_STORED_TOOL_OUTPUT:
        return value
    half = MAX_STORED_TOOL_OUTPUT // 2
    omitted = len(value) - half * 2
    return f"{value[:half]}\n… {omitted:,} characters omitted from tool output …\n{value[-half:]}"


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")


def _terminal_safe(value: str) -> str:
    value = _ANSI_ESCAPE.sub("", value).expandtabs(4)
    return "".join(character if character in "\n" or character.isprintable() else "�" for character in value)


def _bounded_visual_lines(value: str, width: int, limit: int) -> tuple[list[str], int]:
    """Wrap safe tool output and return a bounded viewport plus omitted-row count."""
    visual: list[str] = []
    for source in _terminal_safe(value).splitlines() or [""]:
        current = ""
        used = 0
        for character in source:
            character_width = wcwidth(character)
            character_width = character_width if character_width >= 0 else 1
            if current and used + character_width > width:
                visual.append(current)
                current = ""
                used = 0
            current += character
            used += character_width
        visual.append(current)
    omitted = max(0, len(visual) - limit)
    return visual[:limit], omitted


_INLINE_MARKDOWN = re.compile(
    r"(\*\*.+?\*\*|__.+?__|`[^`]+`|\[[^]]+\]\([^)]+\)|(?<!\*)\*[^*]+\*(?!\*)|(?<!_)_[^_]+_(?!_))"
)


def _markdown_block_lines(value: str, width: int) -> list[TextLine]:
    """Render Markdown blocks, including streaming-safe GFM-style tables."""
    source = value.split("\n")
    result: list[TextLine] = []
    index = 0
    in_code = False
    while index < len(source):
        if not in_code and index + 1 < len(source):
            header = _split_table_row(source[index])
            alignments = _table_alignments(source[index + 1])
            if header is not None and alignments is not None and len(header) == len(alignments):
                rows: list[list[str]] = []
                cursor = index + 2
                while cursor < len(source):
                    row = _split_table_row(source[cursor])
                    if row is None:
                        break
                    rows.append((row + [""] * len(header))[: len(header)])
                    cursor += 1
                result.extend(_render_markdown_table(header, alignments, rows, width))
                index = cursor
                continue
        rendered, in_code = _markdown_line(source[index], in_code=in_code)
        if rendered is not None:
            result.append(TextLine(tuple(rendered)))
        index += 1
    return result


def _split_table_row(value: str) -> list[str] | None:
    """Split table cells while preserving escaped and inline-code pipes."""
    stripped = value.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    in_code = False
    for character in stripped:
        if escaped:
            current.append(character if character in ("|", "\\") else f"\\{character}")
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "`":
            in_code = not in_code
            current.append(character)
        elif character == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells if len(cells) >= 2 else None


def _table_alignments(value: str) -> list[str] | None:
    cells = _split_table_row(value)
    if cells is None:
        return None
    alignments: list[str] = []
    for cell in cells:
        marker = cell.replace(" ", "")
        if re.fullmatch(r":-{3,}:", marker):
            alignments.append("center")
        elif re.fullmatch(r"-{3,}:", marker):
            alignments.append("right")
        elif re.fullmatch(r":?-{3,}", marker):
            alignments.append("left")
        else:
            return None
    return alignments


def _render_markdown_table(
    header: list[str], alignments: list[str], rows: list[list[str]], width: int
) -> list[TextLine]:
    original_columns = len(header)
    column_count = original_columns
    while column_count > 1 and width - (3 * column_count + 1) < 3 * column_count:
        column_count -= 1
    header = header[:column_count]
    alignments = alignments[:column_count]
    rows = [row[:column_count] for row in rows]
    if column_count < original_columns:
        header[-1] = "…"
        rows = [row[:-1] + ["…"] for row in rows]

    natural = []
    for column in range(column_count):
        values = [header[column], *(row[column] for row in rows[:MAX_MARKDOWN_TABLE_ROWS])]
        natural.append(max(3, min(32, max(_text_width(_plain_inline(value)) for value in values))))
    budget = max(column_count * 3, width - (3 * column_count + 1))
    column_widths = natural[:]
    while sum(column_widths) > budget:
        largest = max(range(column_count), key=column_widths.__getitem__)
        if column_widths[largest] <= 3:
            break
        column_widths[largest] -= 1

    border_style = Style(foreground="#587062", dim=True)
    lines = [TextLine((Span(_table_border("╭", "┬", "╮", column_widths), border_style),))]
    lines.append(_table_content_line(header, column_widths, ["center"] * column_count, header=True))
    lines.append(TextLine((Span(_table_border("├", "┼", "┤", column_widths), border_style),)))
    visible_rows = rows[:MAX_MARKDOWN_TABLE_ROWS]
    lines.extend(_table_content_line(row, column_widths, alignments) for row in visible_rows)
    if len(rows) > len(visible_rows):
        omitted = [f"… {len(rows) - len(visible_rows)} rows omitted", *([""] * (column_count - 1))]
        lines.append(_table_content_line(omitted, column_widths, ["left"] * column_count))
    lines.append(TextLine((Span(_table_border("╰", "┴", "╯", column_widths), border_style),)))
    return lines


def _table_border(left: str, junction: str, right: str, widths: list[int]) -> str:
    return left + junction.join("─" * (width + 2) for width in widths) + right


def _table_content_line(
    cells: list[str], widths: list[int], alignments: list[str], *, header: bool = False
) -> TextLine:
    border_style = Style(foreground="#587062", dim=True)
    base_style = Style(foreground=FOREGROUND, bold=header)
    spans: list[Span] = [Span("│", border_style)]
    for value, cell_width, alignment in zip(cells, widths, alignments, strict=True):
        content = _clip_inline_spans(value, cell_width, base_style)
        used = sum(_text_width(span.text) for span in content)
        remaining = max(0, cell_width - used)
        left = remaining if alignment == "right" else remaining // 2 if alignment == "center" else 0
        right = remaining - left
        spans.extend((Span(" " * (left + 1), base_style), *content, Span(" " * (right + 1), base_style)))
        spans.append(Span("│", border_style))
    return TextLine(tuple(spans))


def _clip_inline_spans(value: str, width: int, base_style: Style) -> list[Span]:
    value = value.expandtabs(4)
    source = _inline_markdown(value, base_style=base_style)
    needs_ellipsis = _text_width(_plain_inline(value)) > width
    clipped: list[Span] = []
    used = 0
    truncated = False
    for span in source:
        chunk = ""
        for character in span.text:
            character_width = wcwidth(character)
            character_width = character_width if character_width >= 0 else 1
            reserve = 1 if needs_ellipsis else 0
            if used + character_width > width - reserve:
                truncated = True
                break
            chunk += character
            used += character_width
        if chunk:
            clipped.append(Span(chunk, span.style))
        if truncated:
            break
    if truncated:
        clipped.append(Span("…", base_style))
    return clipped


def _plain_inline(value: str) -> str:
    return "".join(span.text for span in _inline_markdown(value))


def _text_width(value: str) -> int:
    return sum(max(0, wcwidth(character)) for character in value)


def _markdown_line(line: str, *, in_code: bool) -> tuple[list[Span] | None, bool]:
    stripped = line.strip()
    if stripped.startswith("```"):
        if in_code:
            return None, False
        language = stripped[3:].strip()
        label = f"  {language}" if language else ""
        return ([Span(label, Style(foreground=MUTED, background="#222725"))] if label else None), True
    if in_code:
        return [Span(f"  {line}", Style(foreground="#d9dfdb", background="#222725"))], True

    heading = re.match(r"^(#{1,3})\s+(.+)$", line)
    if heading:
        level = len(heading.group(1))
        return _inline_markdown(
            heading.group(2), base_style=Style(foreground=FOREGROUND, bold=True, dim=level > 1)
        ), False

    bullet = re.match(r"^(\s*)[-+*]\s+(.+)$", line)
    if bullet:
        prefix = [Span(f"{bullet.group(1)}• ", Style(foreground=GREEN, bold=True))]
        return prefix + _inline_markdown(bullet.group(2)), False

    quote = re.match(r"^\s*>\s?(.*)$", line)
    if quote:
        return [Span("│ ", STYLE_GREEN)] + _inline_markdown(quote.group(1), base_style=STYLE_DETAIL), False

    return _inline_markdown(line), False


def _inline_markdown(value: str, *, base_style: Style = STYLE_NORMAL) -> list[Span]:
    fragments: list[Span] = []
    position = 0
    for match in _INLINE_MARKDOWN.finditer(value):
        if match.start() > position:
            fragments.append(Span(value[position : match.start()], base_style))
        token = match.group(0)
        if token.startswith(("**", "__")):
            fragments.append(Span(token[2:-2], replace(base_style, bold=True)))
        elif token.startswith("`"):
            fragments.append(Span(token[1:-1], Style(foreground="#d9dfdb", background="#303633")))
        elif token.startswith("["):
            label, _, target = token[1:].partition("](")
            fragments.append(Span(label, Style(foreground=LIGHT_GREEN)))
            fragments.append(Span(f" <{target[:-1]}>", STYLE_MUTED))
        else:
            fragments.append(Span(token[1:-1], replace(base_style, italic=True)))
        position = match.end()
    if position < len(value) or not fragments:
        fragments.append(Span(value[position:], base_style))
    return fragments


def _compact_path(path: Path, *, limit: int = 38) -> str:
    value = str(path)
    home = str(Path.home())
    if value == home or value.startswith(home + os.sep):
        value = "~" + value[len(home) :]
    return value if len(value) <= limit else "…" + value[-(limit - 1) :]
