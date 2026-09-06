"""Run a minimal persistent coding agent from the current working directory."""

import argparse
import asyncio
import os
import platform
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from kcs_agent import (
    Agent,
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AnyMessage,
    AssistantMessage,
    CodingExtension,
    CompactionExtension,
    DeepSeekProvider,
    ReasoningEffort,
    SQLiteSessionExtension,
    SQLiteSessionStorage,
    SystemMessage,
    ToolGuidelinesExtension,
    ToolMessage,
    UserMessage,
    new_uuid7,
)

SLASH_COMMANDS = ("/sessions", "/history", "/new", "/use", "/help", "/quit", "/exit")


def build_system_prompt() -> str:
    """Describe the coding task and the safe, non-secret execution environment."""
    now = datetime.now().astimezone()
    return f"""You are a small coding agent working in the current directory.
Inspect relevant files before editing them. Make focused changes, run suitable checks,
and clearly report what changed. Do not run destructive commands unless the user
explicitly asks for them.

Runtime environment:
- Working directory: {Path.cwd().resolve()}
- Local time: {now.isoformat(timespec="seconds")}
- Time zone: {now.tzname() or "unknown"}
- Operating system: {platform.platform()}
- Machine architecture: {platform.machine() or "unknown"}
- Python: {platform.python_implementation()} {platform.python_version()}
- Executable: {sys.executable}
- Shell: {os.getenv("SHELL", "unknown")}

Treat the working directory as the workspace root. All file paths passed to tools must
be relative to it. Environment details may become stale during a long-running process;
use tools to verify mutable state when accuracy matters."""


def parse_args() -> argparse.Namespace:
    """Parse the small set of options needed by the example."""
    parser = argparse.ArgumentParser(description="Run the persistent KCS coding agent")
    parser.add_argument("--session", help="Session ID to restore; defaults to the latest session")
    parser.add_argument("--model", default=os.getenv("KCS_AGENT_MODEL", "deepseek-v4-flash"))
    parser.add_argument(
        "--database",
        type=Path,
        default=Path.home() / ".kcs-agent" / "coding-agent.sqlite3",
        help="SQLite history database",
    )
    parser.add_argument(
        "--compaction-max-tokens",
        type=int,
        default=int(os.getenv("KCS_COMPACTION_MAX_TOKENS", "128000")),
        help="Compact before a model call when context exceeds this token estimate",
    )
    parser.add_argument(
        "--compaction-keep-tokens",
        type=int,
        default=int(os.getenv("KCS_COMPACTION_KEEP_TOKENS", "32000")),
        help="Minimum recent-token budget retained without summarization",
    )
    parser.add_argument(
        "--compaction-reasoning-effort",
        choices=[effort.value for effort in ReasoningEffort],
        default=os.getenv("KCS_COMPACTION_REASONING_EFFORT", ReasoningEffort.LOW.value),
        help="Reasoning effort used by the compaction model",
    )
    args = parser.parse_args()
    if args.compaction_max_tokens < 1 or args.compaction_keep_tokens < 1:
        parser.error("compaction token limits must be positive")
    if args.compaction_keep_tokens >= args.compaction_max_tokens:
        parser.error("--compaction-keep-tokens must be smaller than --compaction-max-tokens")
    return args


def print_help() -> None:
    """Print commands handled locally instead of sending them to the model."""
    print(
        """Commands:
  /sessions          List persisted sessions, newest first
  /history [limit]   Show messages in the active session (default: 20)
  /new               Start a new session
  /use <session-id>  Switch to and restore another session
  /help              Show this help
  /quit              Exit

Input:
  Tab                Complete slash commands, or insert four spaces
  Shift+Tab          Select the previous completion
  Esc then Enter     Insert a newline
  Enter              Send the complete input"""
    )


def create_prompt_session(database: Path) -> PromptSession[str]:
    """Create a Unicode-aware editor with completion, paste, and persistent input history."""
    bindings = KeyBindings()

    @bindings.add("tab")
    def complete_or_indent(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state is not None:
            buffer.complete_next()
            return
        prefix = buffer.document.text_before_cursor
        if prefix.startswith("/") and " " not in prefix:
            matches = [command for command in SLASH_COMMANDS if command.startswith(prefix)]
            common_prefix = os.path.commonprefix(matches)
            if common_prefix != prefix:
                buffer.insert_text(common_prefix[len(prefix) :])
            elif len(matches) > 1:
                buffer.start_completion(select_first=False)
            return
        buffer.insert_text("    ")

    @bindings.add("s-tab")
    def previous_completion(event) -> None:
        event.current_buffer.complete_previous()

    @bindings.add("escape", "enter")
    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    history_path = database.with_suffix(database.suffix + ".history")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(history_path)),
        completer=WordCompleter(list(SLASH_COMMANDS), sentence=True),
        complete_while_typing=False,
        enable_history_search=True,
        key_bindings=bindings,
        multiline=False,
    )


def print_sessions(storage: SQLiteSessionStorage, active_session_id: str) -> None:
    """Render compact session summaries from SQLite."""
    sessions = storage.list_sessions()
    if not sessions:
        print("No persisted sessions.")
        return
    for session in sessions:
        marker = "*" if session.session_id == active_session_id else " "
        updated_at = session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{marker} {session.session_id}  {session.message_count:>4} messages  {updated_at}")


def message_sections(message: AnyMessage) -> list[tuple[str, str]]:
    """Split one persisted message into labeled sections, including reasoning."""
    match message:
        case UserMessage():
            return [("content", message.text)]
        case AssistantMessage():
            sections = []
            if message.reasoning:
                sections.append(("reasoning", message.reasoning))
            if message.content:
                sections.append(("content", message.content))
            if message.tool_calls:
                calls = "\n".join(f"{call.name}({call.arguments})" for call in message.tool_calls)
                sections.append(("tool calls", calls))
            return sections or [("content", "")]
        case ToolMessage(name=name, content=content):
            return [(f"tool result: {name}", content)]
        case SystemMessage(content=content):
            return [("content", content)]


def print_history(storage: SQLiteSessionStorage, session_id: str, limit: int) -> None:
    """Print the newest raw messages while preserving their original order."""
    offset = max(storage.count_messages(session_id) - limit, 0)
    records = storage.list_raw_messages(session_id, limit=limit, offset=offset)
    if not records:
        print("This session has no messages.")
        return
    for record in records:
        created_at = record.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{record.sequence:>4}  {record.message.role.value:<9}  {created_at}")
        for label, content in message_sections(record.message):
            print(f"      {label}:")
            print(textwrap.indent(content or "(empty)", "        "))


def print_event(event: AgentEvent, output_state: dict[str, bool]) -> None:
    """Render streamed model and tool events without an additional UI dependency."""
    match event.type:
        case AgentEventType.COMPACTION_STARTED:
            output_state["compaction_reasoning"] = False
            output_state["compaction_summary"] = False
            print("\n[compaction] started", flush=True)
        case AgentEventType.COMPACTION_REASONING_DELTA:
            if not output_state["compaction_reasoning"]:
                print("[compaction reasoning] ", end="", flush=True)
                output_state["compaction_reasoning"] = True
            print(event.delta, end="", flush=True)
        case AgentEventType.COMPACTION_TEXT_DELTA:
            if not output_state["compaction_summary"]:
                print("\n[compaction summary] ", end="", flush=True)
                output_state["compaction_summary"] = True
            print(event.delta, end="", flush=True)
        case AgentEventType.COMPACTION_COMPLETED:
            status = "applied" if event.applied else "skipped because the summary was not smaller"
            print(f"\n[compaction] {status}", flush=True)
        case AgentEventType.REASONING_DELTA:
            if not output_state["reasoning"]:
                print("\n[reasoning] ", end="", flush=True)
                output_state["reasoning"] = True
            print(event.delta, end="", flush=True)
        case AgentEventType.TEXT_DELTA:
            if not output_state["answer"]:
                print("\n[assistant] ", end="", flush=True)
                output_state["answer"] = True
            print(event.delta, end="", flush=True)
        case AgentEventType.TOOL_STARTED if event.tool_calls:
            call = event.tool_calls[0]
            print(f"\n[tool] {call.name}({call.arguments})", flush=True)
        case AgentEventType.TOOL_COMPLETED if isinstance(event.message, ToolMessage):
            preview = event.message.content
            if len(preview) > 500:
                preview = preview[:497] + "..."
            print(f"[tool result] {preview}", flush=True)
        case AgentEventType.TOOL_FAILED if event.tool_calls:
            print(f"[tool failed] {event.tool_calls[0].name}: {event.error}", flush=True)
        case _:
            return


async def run_request(agent: Agent, session_id: str, prompt: str) -> None:
    """Stream one request and let SessionPersistenceExtension save every message."""
    output_state = {
        "compaction_reasoning": False,
        "compaction_summary": False,
        "reasoning": False,
        "answer": False,
    }
    async for event in agent.stream(prompt, config=AgentConfig(session_id=session_id)):
        print_event(event, output_state)
    print()


async def main() -> None:
    """Start the REPL, restoring the requested or most recently active session."""
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API")
    if not api_key:
        raise SystemExit("DEEPSEEK_API is required")

    persistence = SQLiteSessionExtension(args.database)
    storage = persistence.storage
    recent_sessions = storage.list_sessions(limit=1)
    session_id = args.session or (recent_sessions[0].session_id if recent_sessions else new_uuid7())
    provider = DeepSeekProvider(args.model, api_key)
    prompt_session = create_prompt_session(args.database)
    agent = await Agent.create(
        provider,
        system_prompt=build_system_prompt(),
        extensions=[
            CodingExtension(),
            persistence,
            ToolGuidelinesExtension(),
            CompactionExtension(
                provider,
                max_tokens=args.compaction_max_tokens,
                keep_recent_tokens=args.compaction_keep_tokens,
                reasoning_effort=ReasoningEffort(args.compaction_reasoning_effort),
            ),
        ],
        config=AgentConfig(session_id=session_id),
    )

    print(f"KCS Coding Agent | cwd={Path.cwd()} | session={session_id}")
    print(
        f"Compaction | trigger>{args.compaction_max_tokens} tokens | "
        f"keep>={args.compaction_keep_tokens} recent tokens | reasoning={args.compaction_reasoning_effort}"
    )
    print_help()
    try:
        with patch_stdout():
            while True:
                try:
                    value = await prompt_session.prompt_async("\nyou> ")
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                if not value.strip():
                    continue
                if not value.startswith("/"):
                    try:
                        await run_request(agent, session_id, value)
                    except Exception as error:
                        print(f"Agent error: {error}")
                    continue

                command, _, argument = value.strip().partition(" ")
                match command:
                    case "/sessions":
                        print_sessions(storage, session_id)
                    case "/history":
                        try:
                            limit = int(argument) if argument else 20
                            if limit < 1:
                                raise ValueError
                        except ValueError:
                            print("Usage: /history [positive-limit]")
                        else:
                            print_history(storage, session_id, limit)
                    case "/new":
                        session_id = new_uuid7()
                        print(f"Started session {session_id}")
                    case "/use" if argument.strip():
                        session_id = argument.strip()
                        print(f"Using session {session_id}; history will be restored on the next request.")
                    case "/use":
                        print("Usage: /use <session-id>")
                    case "/help":
                        print_help()
                    case "/quit" | "/exit":
                        break
                    case _:
                        print(f"Unknown command: {command}. Use /help.")
    finally:
        await provider.aclose()
        persistence.close()


if __name__ == "__main__":
    asyncio.run(main())
