# KCS Agent

A small Python model/tool loop. Python 3.12+, MIT licensed.

This is an **unreleased simplification** of the published 0.1.0 API. Develop locally
with `uv sync`. No compatibility wrappers are kept for the old generic messages,
session state containers, or extensions.

## Start here

Read these files in order:

1. `messages.py`: four message classes with direct fields.
2. `model.py`: a model receives messages and streams its response.
3. `tools.py`: a named function with a Pydantic-generated input schema.
4. `agent.py`: call the model, execute its tools, repeat until it answers.

```text
user message -> model -> assistant answer
                  |
                  +-> tool calls -> tool results -> model
```

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

from kcs_agent import Agent, AgentConfig, AgentState, AssistantMessage, DeepSeekProvider, UserMessage, tool


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
        agent = await Agent.create(model, tools=[add], config=AgentConfig(session_id="calculator-session"))
        reply = await agent.run(
            "Use add to calculate 2 + 3.",
            config=AgentConfig(session_id="calculator-session"),
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
await agent.initialize(config=AgentConfig(session_id="session-42"))
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
    async def on_message(self, context):
        context.state.messages.extend(await load_messages(context.config.session_id))


agent = await Agent.create(
    model,
    extensions=[RestoreHistory()],
    config=AgentConfig(session_id="conversation-42"),
)
reply = await agent.run(
    "What was my previous question?",
    config=AgentConfig(session_id="conversation-42"),
)
```

`AgentConfig.session_id` identifies the conversation that owns a run and is
included in every streamed event. `on_message` runs for every request and fills
the new state before the incoming user message is appended.

## Extensions

Subclass `AgentExtension` to observe lifecycle steps or modify `state.messages`.
Its hooks are grouped in `extension_hooks.py` by responsibility:

- `AgentSetupHooksMixin` prepares request tools and messages.
- `AgentRunHooksMixin` observes the complete request and terminal outcome.
- `AgentModelHooksMixin` wraps primary model calls.
- `AgentToolHooksMixin` wraps tool calls.
- `AgentEventHooksMixin` emits streaming events and receives published events.

`AgentExtension` combines these groups and provides no-op defaults, so an
extension only overrides the hooks it needs.

`extensions.py` contains the concrete in-memory history and tool guidance
extensions; `extension_events.py` defines internal notification events.
Available hooks are `on_message`, `before_run`, `before_model`,
`before_model_events`, `after_model`, `before_tool`, `after_tool`, `after_run`,
`on_success`, `on_error`, and `on_event`. Hooks run sequentially in extension
registration order. `before_model_events` is an async event stream for visible
pre-model work such as compaction.

`on_success(context, result)` runs once per successful request, after all
`after_run` hooks and before `RUN_COMPLETED` is emitted. It does not run for
intermediate model steps, failed requests, or cancellation. Callback failures
propagate through `on_error` and prevent the completion event.

Each hook receives an `AgentContext` containing this run's `config`, fresh
`state`, and the live `tools` dictionary. A new context and state are created for
each run; only the tools registry remains shared with the Agent. Responses, tool calls, and tool
results remain separate arguments on their respective hooks.

Register additional tools through `context.tools[tool.name] = tool` in
`on_message`. Place that extension before `ToolGuidelinesExtension()` so guidance
includes the injected tools. Model schemas and execution use the same registry.

The system prompt enters each fresh `state.messages` before `on_message` runs.
Model requests read the state messages directly.

By default, each Agent creates its own `InMemoryMessageAccumulator` and
`ToolGuidelinesExtension`, so `Agent(model, tools=[add])` enables both. Passing
an explicit `extensions` sequence replaces the defaults; `extensions=[]`
disables them. To share history across Agent instances, supply a shared accumulator:

```python
memory = InMemoryMessageAccumulator()
guidance = ToolGuidelinesExtension()

agent = await Agent.create(
    model,
    config=AgentConfig(session_id="math"),
    tools=[add],
    extensions=[memory, guidance],
)
await agent.run("Add 2 and 3", config=AgentConfig(session_id="math"))

# A later Agent instance can continue the same in-memory session.
next_agent = await Agent.create(
    model, tools=[add], extensions=[memory, guidance], config=AgentConfig(session_id="math")
)
await next_agent.run("Now add 4", config=AgentConfig(session_id="math"))
```

`ToolGuidelinesExtension` injects guidance into every fresh state through
`on_message`, after the other system instructions and before dialogue.
`InMemoryMessageAccumulator` stores one mutable message list per session and
offers `messages(session_id)` and `clear(session_id)` for inspection and cleanup.

## Tools

For database-backed sessions, see [Session storage](docs/session-storage.md).
`SQLiteSessionExtension(path)` owns a SQLite storage and reloads the latest
snapshot and raw-log tail before every request, including repeated calls on the same Agent. New messages are
persisted through `MessageAppendedEvent`; only compaction creates snapshots.

Use `CompactionExtension` to summarize older context before a model call:

```python
from kcs_agent import CompactionExtension

agent = await Agent.create(
    model,
    config=AgentConfig(session_id="example"),
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
    async def on_event(self, context: AgentContext, event: ExtensionEvent) -> None:
        match event:
            case CompactionEvent():
                print(event.compressed_from, event.compressed_to)
```

Publishing awaits each registered extension in order. Handler errors stop
delivery and propagate; completed compaction is not rolled back. Context does
not retain events or compaction flags. Subscribers own any history they need.
Consumers that stop iterating early must close the stream, for example with
`contextlib.aclosing`.

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

```python
from kcs_agent import (
    Agent,
    AgentConfig,
    glob,
    grep,
    read_file,
    replace_in_file,
    run_shell,
    write_file,
)

agent = await Agent.create(
    model,
    config=AgentConfig(session_id="session-42"),
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
`.kcs-tool-output/shell-*/` in the working directory. Use `read_file` with
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

## Streaming

`CodingExtension()` registers `read_file`, `write_file`, `replace_in_file`,
`glob`, `grep`, and `run_shell` for each request, using the current working directory:

```python
agent = await Agent.create(
    model,
    config=AgentConfig(session_id="coding"),
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

## Persistent coding-agent example

Run the minimal terminal coding agent from the directory it should work in:

```bash
export DEEPSEEK_API="..."
uv run --project /path/to/kcs-agent python /path/to/kcs-agent/examples/coding_agent.py
```

It registers `glob`, `grep`, `read_file`, `write_file`, `replace_in_file`, and
`run_shell`. Conversation history is stored in
`~/.kcs-agent/coding-agent.sqlite3`; restarting the example restores the most
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

Native provider details stay in `providers.py`; they are not part of the core loop.
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

## Checks

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

Live checks are opt-in and use `DEEPSEEK_API`:

```bash
KCS_AGENT_LIVE_TESTS=1 uv run pytest tests/test_live_providers.py
```
