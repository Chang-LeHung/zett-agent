# Zett Agent

A small Python model/tool loop. Python 3.12+, MIT licensed.

This is an **unreleased simplification** of the published 0.1.0 API. Develop locally
with `uv sync`. No compatibility wrappers are kept for the old generic messages,
session state containers, or extensions.

## Start here

Read these files in order:

1. `messages.py`: four message classes with direct fields.
2. `model.py`: a model receives messages and streams its response.
3. `tools/base.py`: typed tool definitions and Pydantic-generated input schemas.
4. `tools/coding.py`: working-directory file, search, and shell tools.
5. `agent.py`: call the model, execute its tools, repeat until it answers.

```text
user message -> model -> assistant answer
                  |
                  +-> tool calls -> tool results -> model
```

## Synchronous calls

Use the same runtime from ordinary functions, without an async entry point:

```python
from zett_agent import create_agent_sync

with create_agent_sync(model) as agent:
    print(agent.run("Hello").content)
    with agent.stream("Continue") as events:
        for event in events:
            print(event.type, event.delta)
```

Existing objects expose `.sync()` views: `with provider.sync() as provider_sync`,
`with read_file.sync() as read`, or `with storage.sync() as storage_sync`.
Use `SyncRuntime.call()` for any other async function and share a runtime when
reusing SDK resources. Supplied models and storage remain caller-owned.
Extension lifecycle hooks remain asynchronous; event dispatchers also accept plain `def` callbacks.
For a blocking custom model, use `SyncModelAdapter`.

Run the complete offline example with
`uv run python docs/_examples/synchronous.py`. See the **Synchronous Python**
guide in the generated documentation for cleanup, concurrency, and cancellation.

## Multiple sessions on one Agent

An initialized Agent can run different sessions concurrently on one event loop.
Each request owns its state, tool metadata, queues, and extension context.
Overlapping requests for the same session are rejected until cleanup finishes.

```python
replies = await asyncio.gather(
    agent.run("Explain indexing", config=AgentRunConfig("session-a")),
    agent.run("Review my plan", config=AgentRunConfig("session-b")),
)
state_a = agent.get_state("session-a")
```

Use `get_state(session_id)` for session-specific reads. `agent.state` is only
the most recently started request's convenience view. External events require
explicit config when multiple sessions are active. History requires an
accumulator or persistence extension; extensions=[] does not remember dialogue.

This is conversation isolation, not a security sandbox: filesystem and shell
tools still share the configured working directory and process permissions.
Model clients and custom tool handlers must support concurrent calls; custom
extensions must keep mutable request data under AgentRunContext or session IDs.
Do not mutate shared Agent configuration while requests are running.

## A complete example without an API key

```bash
uv sync
uv run python examples/basic.py
# 5
```

The example's small `DemoModel` requests an addition tool and then returns its result.

A real provider uses the same interface:

```python
import asyncio
import os

from zett_agent import Agent, AgentRunConfig, AgentState, AssistantMessage, DeepSeekProvider, UserMessage, tool


@tool
def add(left: int, right: int) -> int:
    """Add two integers.

    Guidelines:
        - Use for exact integer addition.
    """
    return left + right


async def main() -> None:
    model = DeepSeekProvider("deepseek-v4-flash", os.environ["DEEPSEEK_API"])
    try:
        agent = await Agent.create(model, tools=[add], config=AgentRunConfig(session_id="calculator-session"))
        reply = await agent.run(
            "Use add to calculate 2 + 3.",
            config=AgentRunConfig(session_id="calculator-session"),
        )
        print(reply.content)
    finally:
        await model.aclose()


asyncio.run(main())
```

## Initialization

Direct construction requires explicit initialization:

```python
agent = Agent(model)
await agent.initialize(config=AgentRunConfig(session_id="session-42"))
reply = await agent.run("Hello")
```

Alternatively, `await Agent.create(model, config=...)` returns an initialized
agent. Calling run or consuming stream before initialization raises
`AgentProtocolError`. Initialization binds the default session configuration;
it does not load messages. Requests default to that configuration.

## Messages and history

Use `SystemMessage(content=...)`, `UserMessage(content=...)`,
`AssistantMessage(content=..., tool_calls=...)`, and
`ToolMessage(tool_call_id=..., name=..., content=...)`.

There are no `data`/`metadata` wrappers or message generics.
User messages also support the existing text/image content blocks.

Every request receives a fresh `AgentState`. Restore history through an extension:

```python
class RestoreHistory(AgentExtension):
    async def on_state(self, context):
        context.state.messages.extend(await load_messages(context.config.session_id))


agent = await Agent.create(
    model,
    extensions=[RestoreHistory()],
    config=AgentRunConfig(session_id="conversation-42"),
)
reply = await agent.run(
    "What was my previous question?",
    config=AgentRunConfig(session_id="conversation-42"),
)
```

`AgentRunConfig.session_id` identifies the conversation that owns a run and is
included in every streamed event. `on_state` runs for every request and fills
the new state before the incoming user message is appended.

## Extensions

Subclass `AgentExtension` to observe lifecycle steps or modify `state.messages`.
Its hooks are grouped in `extensions/base.py` by responsibility:

- `AgentSetupHooksMixin` prepares request tools and messages.
- `AgentRunHooksMixin` observes the complete request and terminal outcome.
- `AgentModelHooksMixin` observes primary model calls.
- `AgentToolHooksMixin` observes tool calls.
- `AgentEventHooksMixin` receives internal extension events and external input.

`AgentExtension` combines these groups and provides no-op defaults, so an
extension only overrides the hooks it needs.

`extensions/` keeps each concrete extension in its own module;
`extensions/events.py` defines internal notification events.
Available hooks are `on_tool`, `on_state`, `on_message`, `before_run`, `before_turn`,
`before_model`, `after_model`, `before_tool`, `after_tool`, `after_turn`,
`after_run`, `on_success`, `on_error`, and `on_event`. Hooks run sequentially in
extension priority order, with registration order breaking ties. A hook
publishes visible progress with `await context.emit(event)`; only
`Agent.stream()` yields events.

`before_turn(context)` and `after_turn(context, result)` bracket one model/tool
turn: one model call plus every tool invocation it requested. A request runs one
turn per model/tool cycle, so `after_turn` fires with each tool-calling message
and once more for the final answer, before `after_run`. Later turns come from
steering or queued internal input that continues the same request.

`on_success(context, result)` runs once per successful request, after all
`after_run` hooks and before `RUN_COMPLETED` is emitted. It does not run for
intermediate model steps, failed requests, or cancellation. Callback failures
propagate through `on_error` and prevent the completion event.

Each hook receives an `AgentRunContext` containing this run's `config`, fresh
`state`, and the live `tools` dictionary. A new context and state are created for
each run; only the tools registry remains shared with the Agent. Responses, tool calls, and tool
results remain separate arguments on their respective hooks.

Register additional tools through `context.register_tool(tool)` in `on_tool`.
All tool hooks finish before state hooks, so tool guidance includes the complete
registry. Model schemas and execution use the same registry.

Request setup and turns run in this order:

```text
on_tool -> on_state -> on_message -> append_message -> before_run
        -> [before_turn -> before_model -> model -> tools -> after_turn] * n
        -> after_run
```

`on_state` restores history and injects system instructions. `on_message` may
replace `context.input_message` to apply a mode selected through an external
event before the input is appended. The runtime publishes the transformed message once through
`MessageAppendedEvent`; `before_run` sees the finalized input in state.messages.

The system prompt enters each fresh `state.messages` before `on_state` runs.
Model requests read the state messages directly.

## Middleware

Every `AgentExtension` also inherits `MiddlewareHook`. Override its middleware
methods when an extension must wrap the actual provider request or local tool
handler rather than merely observe a lifecycle boundary:

```python
from contextlib import aclosing
from dataclasses import replace

from zett_agent import AgentExtension, SystemMessage


class AuditExtension(AgentExtension):
    async def on_model_request(self, context, request, call_next):
        request = replace(
            request,
            messages=(*request.messages, SystemMessage(content="Audit enabled.")),
        )
        async with aclosing(call_next(request)) as events:
            async for event in events:
                yield event

    async def on_tool_call(self, context, call, call_next):
        return await call_next()


agent = await Agent.create(model, config=config, extensions=[AuditExtension()])
```

Extensions use their existing priority order for middleware too; the first is
the outermost layer. Model middleware can
replace `ModelRequest` and observe streaming events. Tool middleware can adjust
`call.arguments`, return a cached result without calling `call_next`, or raise an
exception that becomes a failed `ToolMessage`. Extension instances may run
concurrently across sessions and parallel tool calls.

By default, each Agent creates its own `InMemoryMessageAccumulator` and
`ToolGuidelinesExtension`, so `Agent(model, tools=[add])` enables both. Passing
an explicit `extensions` sequence replaces the defaults; `extensions=[]`
disables them. To share history across Agent instances, supply a shared accumulator:

```python
memory = InMemoryMessageAccumulator()
guidance = ToolGuidelinesExtension()

agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="math"),
    tools=[add],
    extensions=[memory, guidance],
)
await agent.run("Add 2 and 3", config=AgentRunConfig(session_id="math"))

# A later Agent instance can continue the same in-memory session.
next_agent = await Agent.create(
    model, tools=[add], extensions=[memory, guidance], config=AgentRunConfig(session_id="math")
)
await next_agent.run("Now add 4", config=AgentRunConfig(session_id="math"))
```

`ToolGuidelinesExtension` injects guidance into every fresh state through
`on_state`, after the other system instructions and before dialogue.
`InMemoryMessageAccumulator` stores one mutable message list per session and
offers `messages(session_id)` and `clear(session_id)` for inspection and cleanup.

## Tools

For database-backed sessions, see [Session storage](docs/session-storage.md).
`SQLiteSessionExtension(path)` owns a SQLite storage and reloads the latest
snapshot and raw-log tail before every request, including repeated calls on the same Agent. New messages are
persisted through `MessageAppendedEvent`; only compaction creates snapshots.
`BaseSessionPersistenceExtension[StorageT]` contains all lifecycle integration.
A backend-specific extension only constructs its `SessionStorage` and passes it
to `super().__init__`; it does not implement Agent hooks. When an application
already owns a storage instance, use `SessionPersistenceExtension(storage)` as
the ready-made adapter.
Each session row stores only its identity, optional parent, display title,
provider-neutral agent name, and timestamps. Title and agent name are storage
metadata, not Agent runtime configuration. Read them with
`SQLiteSessionExtension.get_session(session_id)` and change them with
`update_session(session_id, title=..., agent_name=...)`.
`delete_session(session_id)` explicitly removes the session, its Raw Log, and
its snapshots.

`run()` and `stream()` accept JSON-compatible `metadata` and `tags` as
request-scoped extension context:

```python
reply = await agent.run(
    "Summarize this note",
    metadata={"source": "clipboard", "asset_ids": ["asset-1"]},
    tags={"domain": "python"},
)
```

The values are available as `AgentRunContext.metadata` and `AgentRunContext.tags` at
every lifecycle hook. They are never attached to `UserMessage`,
`AssistantMessage`, or `ToolMessage`, so provider requests cannot receive them.
The persistence extension captures the current context values beside every
appended Raw Log message in dedicated JSON columns. `RawMessageRecord` exposes
the decoded values to a UI or another storage consumer.

Every `AgentExtension` has a `priority` of `100`. Lower values run first across
all lifecycle hooks, internal event subscribers, and external-event handlers.
Sorting is stable, so extensions with equal priorities retain the order in
which they were registered. An extension can declare `priority = 10` on its
class or override `extension.priority` for one instance.

Tool lifecycle events expose `tool_calls: list[ToolCall]`. Sequential execution
currently emits one call in each `TOOL_STARTED`, `TOOL_COMPLETED`, or
`TOOL_FAILED` event; using a list keeps the event protocol ready for a future
concurrent tool batch without another field-shape change.

Use `CompactionExtension` to summarize older context before a model call:

```python
from zett_agent import CompactionExtension

agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="example"),
    extensions=[
        InMemoryMessageAccumulator(),
        ToolGuidelinesExtension(),
        CompactionExtension(model, max_tokens=128_000, keep_recent_tokens=32_000),
    ],
)
```

The default trigger is 128,000 tokens, retaining at least 32,000 recent tokens.
These are compaction budgets, not a model context-window declaration. Configure
them for both the primary model and the summarization model, leaving room for
tool schemas, summary instructions, and generated output.

Both limits use tokens. The default o200k_base tokenizer counts message representations;
this estimates provider context usage. Supply count_tokens for model-specific accounting,
including image token costs. Recent turns are retained until keep_recent_tokens is reached.
The extension preserves system instructions and complete recent user turns,
including tool calls and results. Its output is system messages, one historical
checkpoint, then recent dialogue. Later checkpoints incorporate earlier ones.
An oversized current turn is kept intact. Invalid summaries raise an error without
changing messages; summaries that do not reduce size are ignored. This updates
active context; register `SessionPersistenceExtension` to retain immutable raw
history and create a snapshot only when compaction occurs.

After replacing context successfully, compaction calls `await context.publish`
with an immutable `CompactionEvent`. It contains `compressed_from`,
`compressed_to`, `kept_from`, `kept_to`, and `summary`. Ranges are one-based,
inclusive positions in the non-system context before that compaction, not raw
log IDs or user-turn numbers. A previous checkpoint counts as one message.
At the public stream boundary it also emits `COMPACTION_STARTED`, incremental
`COMPACTION_REASONING_DELTA` / `COMPACTION_TEXT_DELTA`, and
`COMPACTION_COMPLETED`. The completion event exposes `applied` and the internal
`CompactionEvent`, allowing Web and TUI clients to show compaction as a distinct
runtime state before `MODEL_STARTED`.

```python
class Observer(AgentExtension):
    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        match event:
            case CompactionEvent():
                print(event.compressed_from, event.compressed_to)
```

Publishing awaits each registered extension in priority order. Handler errors
stop delivery and propagate; completed compaction is not rolled back. Context
does not retain events or compaction flags. Subscribers own any history they
need. Consumers that stop iterating early must close the stream, for example
with `contextlib.aclosing`.

`@tool` reads its prompt metadata from the function docstring. The first
paragraph becomes the description; `Args`, `Snippet`, and `Guidelines` provide
JSON Schema field descriptions and grouped system-prompt guidance:

```python
from pathlib import Path


@tool
def read_file(path: str) -> str:
    """Read one text file.

    Args:
        path: File path relative to the workspace.

    Snippet:
        read_file(path="README.md")

    Guidelines:
        - Read the current content before editing it.
    """
    return Path(path).read_text()
```

Python annotations still define types and Pydantic validates arguments before
execution. A description and at least one guideline are required. Explicit
decorator metadata remains available as an override.

The built-in local tools operate relative to the process's current working
directory:

- `FileSystemExtension(read_only=True)` registers `read_file`, `glob`, and
  `grep` only.
- `FileSystemExtension(read_only=False)` additionally registers `write_file`
  and `replace_in_file`.
- `CodingExtension()` provides writable filesystem tools plus `run_shell`.

`run_shell` is intentionally excluded from `FileSystemExtension`: arbitrary
commands cannot be classified as read-only at the extension boundary.

```python
from zett_agent import (
    Agent,
    AgentRunConfig,
    glob,
    grep,
    read_file,
    replace_in_file,
    run_shell,
    write_file,
)

agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="session-42"),
    tools=[glob, grep, read_file, write_file, replace_in_file, run_shell],
)
```

File paths must be relative and cannot escape the current working directory.
`glob` discovers paths and `grep` searches UTF-8 text with regular expressions.
Reads support one-based line ranges, writes are atomic, and exact replacement
requires a unique match unless `replace_all=True`. Shell output includes
`exit_code`, `stdout`, `stderr`, timeout state, and truncation state. `run_shell`
executes arbitrary host commands and is not a security sandbox; register it only
for trusted agents.

Tool output uses bounded previews:

- `read_file` returns at most 50 KiB of UTF-8 content and the requested line
  count. `next_line` and `next_column` are one-based resume coordinates; pass
  them as `start_line` and `start_column`. Even a single very long line can be
  read without losing characters. Reads scan incrementally, including files
  larger than 2 MiB; counting `total_lines` still requires scanning the file.
- `grep` caps serialized match records at 50 KiB and each text window at 2,000
  bytes. A long-line window includes the first match, with `text_start_column`
  and `text_truncated` explaining its position. `truncated` means additional
  results were omitted. Files over 2 MiB remain excluded from grep scans.
- `glob` caps returned path text at 50 KiB as well as `max_results`.
- `run_shell` redirects stdout and stderr to separate files, avoiding an
  unbounded in-memory capture. Each preview is at most 50 KiB and 2,000 lines,
  including an explicit omission marker. Approximately one quarter of the byte
  budget shows startup context and three quarters shows final diagnostics.
  These are byte/line limits, not token limits, and JSON encoding adds overhead.

If either shell preview truncates, both original streams are retained under
`.zett-tool-output/shell-*/` in the working directory. Use `read_file` with
`stdout_path` or `stderr_path` to inspect omitted content. Untruncated command
files are removed; retained logs require manual cleanup and have no disk quota.
Stdout/stderr ordering across streams is not reconstructed. On POSIX, timeout
and cancellation kill the process group; on other platforms only the direct
process is killed. Commands should remain foreground, bounded operations.

Continuous ranges suit file reads because they preserve source order. Match
windows suit grep because they keep the relevant location visible. Head/tail
previews suit shell commands because startup details and final errors can both
matter; the full saved output is necessary when the root cause lies in between.

`ToolGuidelinesExtension` groups all snippets before all guidelines and appends
both sections to the system instructions.

## Skills and MCP

`SkillExtension` discovers `SKILL.md` files recursively below its roots. With
no arguments it searches `~/.zett/skills`, `~/.agent/skills`,
`~/.claude/skills`, and `~/.cursor/skills`. Explicit roots replace these
defaults and can be used to opt into project-local skills. Skill names and
resolved files are deduplicated; the first configured root has precedence.
`SkillFileParser` validates UTF-8 content, closed front matter,
required `name` and `description` fields, a normalized lowercase skill name,
and a non-empty Markdown body. Invalid files are skipped without stopping
discovery. The extension injects a small system catalog containing each skill
name, description, and path, then
registers `read_skill`. The complete file enters context only after the model
loads a relevant skill by its advertised name; arbitrary paths are rejected.

```python
from pathlib import Path

from zett_agent import SkillExtension, ToolGuidelinesExtension

extensions = [
    SkillExtension([Path(".agent/skills"), Path.home() / ".agent" / "skills"]),
    ToolGuidelinesExtension(),
]
```

`McpExtension` connects configured Streamable HTTP or stdio servers for one
request, discovers every page of their tool list, and closes all transports on
success, failure, or cancellation. Tool names are namespaced as
`server__tool` by default so different servers cannot silently shadow each
other.

```python
from zett_agent import McpExtension, McpHttpServer, McpStdioServer

mcp = McpExtension(
    [
        McpHttpServer(name="docs", url="http://127.0.0.1:8000/mcp"),
        McpStdioServer(name="local", command="python", args=("mcp_server.py",)),
    ]
)
```

Both extensions are optional. They are not installed in a concrete application
agent unless that application includes them in its explicit extension list.

Tool code is organized under `zett_agent/tools/`: `base.py` defines `AgentTool`,
the `@tool` decorator, schema generation, and prompt guidance; `coding.py`
contains the local coding tools; `output.py` owns bounded preview behavior.
The package `__init__.py` exposes the stable public tool API.

## Streaming

`CodingExtension()` registers `read_file`, `write_file`, `replace_in_file`,
`glob`, `grep`, and `run_shell` for each request, using the current working directory:

```python
agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="coding"),
    extensions=[CodingExtension(), ToolGuidelinesExtension()],
)
```

Include `InMemoryMessageAccumulator()` or a persistence extension when history
is needed. Shell execution is included and uses the host process permissions.

Configure the default reasoning level with `Agent(..., reasoning_effort=...)` or
`Agent.create(..., reasoning_effort=...)`. Both `run()` and `stream()` inherit
that default when their `reasoning_effort` argument is omitted; passing an
explicit value overrides it for that request only.

`agent.stream(message, config=..., reasoning_effort=...)` yields
`AgentEvent` objects in order. Events expose compaction progress, text/reasoning
deltas, model responses, tool calls, tool results, and the final answer. `run()` collects that stream and
returns the final `AssistantMessage`.

Tool errors are returned to the model. Model errors propagate to the application.

## Todo-write extension

`TodoWriteExtension` registers a session-scoped `todo_write` tool for ordered,
observable task progress:

```python
todo_extension = TodoWriteExtension()
agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="coding"),
    extensions=[todo_extension, ToolGuidelinesExtension()],
)
```

The model submits the complete list on every call. The first task starts as
`processing`, later tasks remain `pending`, and completing the current task
advances `processing` to exactly the next position. After the final update,
every task is `completed` and no task is processing. Existing task content and
order cannot change, and invalid updates fail atomically without replacing the
last valid state.

The Tool result contains the complete validated list, its zero-based
`processing_index`, the current `processing` item, and a final `completed`
flag. Applications can call `todo_extension.todos(session_id)` while the
request is active. The extension clears request-local state after success,
failure, or cancellation so a finished request cannot leak stale todos into a
later run.

## Ask-user extension

`AskUserExtension` registers an `ask_user` tool and pauses its tool step while a
streaming UI collects input. Add the extension to the Agent, forward its
`AskUserEvent` to the client, then route the response back through the Agent's
`emit_external_event()` boundary:

```python
ask_user_extension = AskUserExtension()
agent = await Agent.create(model, config=config, extensions=[ask_user_extension])


async def stream_to_ui():
    async for event in agent.stream("Prepare my document"):
        if isinstance(event, AskUserEvent):
            send_to_ui(event.name, event.payload)


# Called independently by the UI response endpoint while stream_to_ui waits.
def accept_from_ui(name: str, payload: dict[str, object]) -> list[str]:
    return agent.emit_external_event(ExternalEvent(name=name, payload=payload), config=config)
```

The outbound payload includes `session_id`, `tool_call_id`, question, ordered options,
`allow_multiple`, and the expected response event name. The inbound envelope
always contains only `name` and `payload`. Routing identity is passed separately
as `AgentRunConfig`; the payload's `tool_call_id` identifies the pending question.
Other response fields are
application-defined and are returned unchanged to the model. Once the response
is accepted, the stream proceeds directly to `TOOL_STARTED`; `TOOL_COMPLETED`
confirms that the answer has been returned to the model.
`emit_external_event()` calls `accept(config, event)` on every extension in
priority order and returns the names of all accepting extensions. An empty list
means no extension accepted the event, including malformed, duplicated, stale,
or cancelled ask-user responses. The event and payload are forwarded unchanged;
extensions decide whether to accept them. If config is omitted, the Agent uses
the active request's config, then the initialized config, or None before
initialization. Custom extensions may accept events even without a config.

## Plan Mode extension

`PlanModeExtension` lets the model propose a session-scoped planning mode, but
requires explicit user confirmation before activating it:

```python
plan_mode = PlanModeExtension()
agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="coding-session"),
    extensions=[
        CodingExtension(),
        ToolGuidelinesExtension(),
        plan_mode,
    ],
)

async for event in agent.stream("Design the authentication refactor"):
    if event.name == ENTER_PLAN_MODE_EVENT_NAME:
        approved = await confirm_in_ui(event.payload["reason"])
        agent.emit_external_event(
            ExternalEvent(
                name=ENTER_PLAN_MODE_RESPONSE_EVENT_NAME,
                payload={
                    "session_id": event.session_id,
                    "tool_call_id": event.tool_calls[0].id,
                    "approved": approved,
                },
            )
        )
```

Outside Plan Mode, the model receives an `enter_plan_mode` Tool. A valid Tool
Call emits the `enter_plan_mode` Custom Event and waits before `TOOL_STARTED`.
The extension enters Plan Mode only after receiving a correlated
`enter_plan_mode_response` External Event with a strict boolean `approved`
field. A response without a pending Tool Call is rejected, so the user cannot
proactively activate the mode. Rejection returns a normal declined Tool result
to the model.

After approval, the stream emits `plan_mode_entered`, replaces all system
instructions with the Plan Mode prompt, and exposes `read_file`, `write_file`,
`replace_in_file`, `glob`, `grep`, `run_shell`, and `exit_plan_mode`. Any other
Tool Call is rejected before execution.

When the plan is ready, the model calls `exit_plan_mode` with the complete
Markdown plan. The extension emits an `exit_plan_mode` Custom Event and waits
for a correlated `exit_plan_mode_response` External Event. Approval emits
`plan_mode_exited` and restores the original system instructions and tools for
the next model step; rejection keeps the session in Plan Mode so the model can
revise the plan. As with entry, a response without a pending Tool Call is
rejected, so neither transition can be triggered out of band.

## Subagent extension

`SubAgentExtension` registers a foreground `task` tool with built-in
`reasoning` and read-only `explore` profiles. Every task receives a fresh UUIDv7
session. Each built-in profile declares its own `SQLiteSessionExtension()` and
persists `parent_session_id` without inspecting the parent Agent's extensions:

```python
storage = SQLiteSessionStorage("sessions.sqlite3")
history = SessionPersistenceExtension(storage)
subagents = SubAgentExtension()

agent = await Agent.create(
    model,
    config=AgentRunConfig(session_id="parent-session"),
    extensions=[
        history,
        subagents,
        ToolGuidelinesExtension(),
    ],
)
```

The generated tool schema restricts `subagent_type` to configured names, while
snippets and selection guidelines are added by `ToolGuidelinesExtension`. The
built-in `reasoning` profile has no tools and uses high reasoning effort.
`explore` receives only `read_file`, `glob`, and `grep`, so it cannot modify the
workspace or run shell commands.

Create custom profiles with `SubAgentDefinition`. A profile may select its own
model, prompt, tools, extensions, reasoning effort, and iteration limit.
`SubAgentDefinition.extensions` is the complete, ordered child lifecycle: add a
`SessionPersistenceExtension` there when the child history must be durable, and
add `ToolGuidelinesExtension` when its tools need prompt guidance. The task tool
does not silently insert either extension. Built-in profiles declare both rules
explicitly. This first version runs in the foreground and returns the child
agent's final report to the parent as a normal tool result; it does not implement
background jobs or task resume.

## Persistent coding-agent example

Run the minimal terminal coding agent from the directory it should work in:

```bash
export DEEPSEEK_API="..."
uv run --project /path/to/zett-agent python /path/to/zett-agent/examples/coding_agent.py
```

It registers `glob`, `grep`, `read_file`, `write_file`, `replace_in_file`, and
`run_shell`. Conversation history is stored in
`~/.zett-agent/coding-agent.sqlite3`; restarting the example restores the most
recent session. Use `/sessions` to list sessions, `/history` to inspect the
active history, `/new` to start another session, and `/use <session-id>` to
switch sessions. The prompt editor supports Unicode and bracketed paste. Tab
completes slash commands or inserts indentation; Esc followed by Enter inserts
a newline, while Enter sends the input.

Compaction is enabled with a 128,000-token trigger and a 32,000-token recent
budget. To observe it quickly with a real provider, use a disposable session and
lower limits:

```bash
uv run python examples/coding_agent.py \
  --session compaction-demo \
  --compaction-max-tokens 500 \
  --compaction-keep-tokens 100 \
  --compaction-reasoning-effort high
```

After a few turns, the terminal streams compaction reasoning and summary events.
The immutable Raw Log remains complete, while the generated checkpoint is saved
in SQLite and restored on later runs.
Cancel the consuming asyncio task to interrupt generation. A running synchronous
tool cannot be forcibly stopped by task cancellation.

Native provider details stay in `providers/`; they are not part of the core loop.
Existing adapters: OpenAI, DeepSeek, Anthropic, Google, and Ollama.

Provider HTTP clients trust the process environment by default. Standard terminal
variables such as `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY` are
applied automatically. For example:

```bash
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
uv run python examples/coding_agent.py
```

An explicitly supplied custom HTTP transport takes precedence and does not use
environment proxy mounts, which keeps mocked and embedded transports isolated.

## Documentation

From the repository root:

```bash
make docs                         # Build HTML from public source docstrings
make docs-serve                   # Preview at http://127.0.0.1:8000
make docs-serve DOCS_PORT=8080     # Choose a different local port
make docs-check                   # Validate API coverage, examples, and links
make docs-examples                # Run the 11 offline example programs
make docs-ui-check                # Check desktop/mobile navigation and search
```

The reference includes every name exported by `zett_agent.__all__`, grouped
into client, runtime, messages, events, models, providers, tools, storage,
extensions, and constants. Edit Google-style source docstrings to update API
descriptions; generated pages under `docs/_generated` are not checked in.
Examples and model requests are not executed during documentation builds.

## Checks

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

Live checks are opt-in and use `DEEPSEEK_API`:

```bash
ZETT_AGENT_LIVE_TESTS=1 uv run pytest tests/test_live_providers.py
```
