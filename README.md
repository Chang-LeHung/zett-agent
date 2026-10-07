<p align="center">
  <img src="https://raw.githubusercontent.com/Chang-LeHung/zett-agent/main/docs/_static/logo.svg" width="88" height="88" alt="Zett Agent" />
</p>

<h1 align="center">Zett Agent</h1>

<p align="center">Small core. Yours to shape.</p>

<p align="center">
  <a href="https://chang-lehung.github.io/zett-agent/">Documentation</a> ·
  <a href="https://chang-lehung.github.io/zett-agent/learn/first-agent.html">Quickstart</a> ·
  <a href="https://chang-lehung.github.io/zett-agent/examples/">Examples</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/zett-agent?color=7663e0)](https://pypi.org/project/zett-agent/)
[![Python](https://img.shields.io/badge/python-3.10%2B-7663e0)](https://pypi.org/project/zett-agent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-7663e0)](https://github.com/Chang-LeHung/zett-agent/blob/main/LICENSE)
[![CI](https://github.com/Chang-LeHung/zett-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Chang-LeHung/zett-agent/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-chang--lehung.github.io-7663e0)](https://chang-lehung.github.io/zett-agent/)

Zett Agent is a small Python runtime for tool-calling agents. One provider-neutral
loop calls a model, runs the tools it asks for, and streams every step back to
your application. Start with a model and a few functions. Add skills, session
storage, approvals, or delegation without adopting an application framework.

- **Provider-neutral.** OpenAI, Anthropic, Google, DeepSeek, and Ollama behind one protocol.
- **Tools from functions.** Annotate a Python function; the schema, validation, and dispatch are generated.
- **Capabilities on demand.** Load skill instructions when needed; discover deferred tools on supported Responses endpoints.
- **Streaming first.** Partial text, reasoning, tool calls, and results for terminals and UIs.
- **Durable when you want it.** In-memory history by default, SQLite or your own store on request.
- **Your application stays yours.** Embed the loop in a script, terminal, UI, or service; add lifecycle behavior with extensions.

Read the [documentation](https://chang-lehung.github.io/zett-agent/) or start
with the complete tool-calling example below. Python 3.10+, typed, MIT licensed.

## Install

```bash
pip install zett-agent
```

```bash
uv add zett-agent
```

## Quickstart

Set `OPENAI_MODEL` to a model that supports tool calling and
`OPENAI_API_KEY` to your credential. This example makes a real network request
and may incur charges. Save it as `agent.py` and run `python agent.py`:

```python
import asyncio
import os

from zett_agent.client import create_agent
from zett_agent.providers.openai import OpenAIProvider
from zett_agent.tools.base import tool


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers exactly."""
    return left + right


async def main() -> None:
    model = OpenAIProvider(
        model=os.environ["OPENAI_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    try:
        client = await create_agent(model, tools=[add])
        reply = await client.run("Use add to calculate 20 plus 22.")
        print(reply.content)
    finally:
        await model.aclose()


asyncio.run(main())
```

The model can request `add`; the runtime validates its arguments, runs it,
returns the result, and asks the model to finish. `reply.content` contains the
final answer. Calling `client.run(...)` again continues the same in-memory
conversation. The exact answer and tool choice depend on the model.

No API key yet? [Your first agent](https://chang-lehung.github.io/zett-agent/learn/first-agent.html)
is a complete, deterministic tutorial with no network access, and
[Streaming tools](https://chang-lehung.github.io/zett-agent/examples/streaming-tools.html)
exercises the same tool round trip offline.

## Choose a provider

Swap the adapter without changing local tools, callbacks, or the client API.
Use `response=True` on an OpenAI-compatible adapter only when your endpoint
supports Responses; Chat Completions is the default.

| Provider | Module |
| --- | --- |
| OpenAI (Chat Completions or Responses) | `zett_agent.providers.openai` |
| Anthropic | `zett_agent.providers.anthropic` |
| Google GenAI | `zett_agent.providers.google` |
| DeepSeek | `zett_agent.providers.deepseek` |
| Ollama | `zett_agent.providers.ollama` |

See [Connect a model provider](https://chang-lehung.github.io/zett-agent/learn/providers.html)
for complete setup with each provider, reasoning effort, retries, proxies,
and cache routing. Model features are endpoint-specific; a common adapter
protocol does not guarantee identical capabilities.

## Stream to a terminal or UI

Dispatch callbacks receive events as they arrive; `run()` still returns the final
answer.

```python
from zett_agent.dispatcher import AgentEventDispatcher
from zett_agent.events import AgentEvent


class Console(AgentEventDispatcher):
    async def on_text_delta_event(self, event: AgentEvent) -> None:
        print(event.delta, end="", flush=True)


client = await create_agent(model, event_dispatcher=Console())
await client.run("Summarize this repository.")
```

If you need cancellation or forwarding, consume the stream directly with
`async with aclosing(client.stream(...)) as events`. See
[Streaming](https://chang-lehung.github.io/zett-agent/learn/streaming.html).

## Tools

Decorate a typed function; Zett Agent builds the schema and turns the docstring
into model-facing guidance.

```python
from zett_agent.tools.base import tool


@tool
def add(left: int, right: int) -> int:
    """Add two integers exactly.

    Args:
        left: First operand.
        right: Second operand.

    Guidelines:
        - Use for exact integer addition.
    """
    return left + right
```

Register it with `create_agent(model, tools=[add])`. The model can then request
the call, and the runtime validates arguments, runs it, appends the result, and
asks the model to continue. Filesystem and shell tools live in `FileSystemExtension`
and `CodingExtension`. See
[Define and register tools](https://chang-lehung.github.io/zett-agent/learn/tools.html).

Each tool needs typed arguments, a description, and at least one guideline.
Use `@tool(guidelines="...")` for a short function or a `Guidelines` section
in its docstring. Guidelines help the model choose tools; they do not enforce
permissions.

## Keep and restore conversations

`AgentRunConfig.session_id` identifies a conversation. Add a persistence
extension to survive restarts:

```python
from zett_agent.agent import AgentRunConfig
from zett_agent.extensions.sqlite import SQLiteSessionExtension

config = AgentRunConfig(session_id="project-notes")
storage = SQLiteSessionExtension("sessions.sqlite")
try:
    client = await create_agent(model, config=config, extensions=[storage])
    reply = await client.run("Remember my preference for Python.")
finally:
    await storage.close()
```

Run this inside an async function with a configured model. Reopen the same
database and session ID to continue after restart. Explicit extension lists
replace default history and tool guidance; include `ToolGuidelinesExtension()`
if you register tools. Close the provider separately when finished. See
[Keep and restore conversations](https://chang-lehung.github.io/zett-agent/learn/sessions.html).

## Embed it safely

Your application owns credentials, session access, and tool permissions.
File, shell, and MCP tools are not sandboxed. Prompts, AGENTS.md, and skills
guide behavior but do not authorize actions.

Set application deadlines, close partially consumed streams, and serialize
each conversation. Cancellation stops unfinished work; it does not undo
already completed tool side effects. Follow
[Integrate an application](https://chang-lehung.github.io/zett-agent/learn/application.html)
for concurrency, errors, timeouts, and shutdown.

## Built-in extensions

Pass the ones you need; `extensions=[...]` replaces the optional defaults.

| Extension | Adds |
| --- | --- |
| `InMemoryMessageAccumulator` | Process-local conversation history (default). |
| `SQLiteSessionExtension` | Durable raw history and compaction checkpoints. |
| `JSONLExtension` | One self-contained audit record per request. |
| `ToolGuidelinesExtension` | Tool snippets and usage rules in the prompt. |
| `AgentsMdExtension` | Working-directory `AGENTS.md` instructions. |
| `FileSystemExtension` / `CodingExtension` | Read, search, edit, and shell tools. |
| `CompactionExtension` | Summarize older turns before the context limit. |
| `AskUserExtension` | Let the model ask the UI a question and wait. |
| `PlanModeExtension` | Model-proposed planning with user approval. |
| `SubAgentExtension` | Delegate to configured child profiles. |
| `SkillExtension` | Discover and load local `SKILL.md` instructions. |
| `McpExtension` | Connect MCP servers as tools. |
| `TodoWriteExtension` | Ordered, observable task progress. |
| `GoalExtension` | Continue explicitly armed Goal Mode work with an evaluator. |
| `ToolSearchExtension` | Load optional tools on demand instead of up front. |

Use an extension when behavior belongs *inside* the request lifecycle; use an
`AgentEventDispatcher` for display-only callbacks. You never subclass `Agent` or
write a second loop. See
[Extend the runtime](https://chang-lehung.github.io/zett-agent/extending/index.html).

## Documentation

| | |
| --- | --- |
| [Get started](https://chang-lehung.github.io/zett-agent/learn/index.html) | Install, first agent, providers, streaming, tools, sessions. |
| [Configure your agent](https://chang-lehung.github.io/zett-agent/learn/configuration.html) | Defaults, per-request settings, metadata, and extension composition. |
| [Integrate an application](https://chang-lehung.github.io/zett-agent/learn/application.html) | Timeouts, cancellation, concurrent sessions, and resource ownership. |
| [Test your agent](https://chang-lehung.github.io/zett-agent/learn/testing.html) | Deterministic models and offline provider tests. |
| [Capabilities on demand](https://chang-lehung.github.io/zett-agent/learn/on-demand.html) | Editable skills, deferred tools, and endpoint compatibility. |
| [How it works](https://chang-lehung.github.io/zett-agent/concepts/index.html) | Ownership, lifecycle, context, events. |
| [Extend the runtime](https://chang-lehung.github.io/zett-agent/extending/index.html) | Extensions, hooks, middleware, model and storage adapters. |
| [Examples](https://chang-lehung.github.io/zett-agent/examples/index.html) | Runnable programs for streaming, approvals, compaction, delegation. |
| [API reference](https://chang-lehung.github.io/zett-agent/reference/index.html) | Every public class, function, and field. |

## Requirements

- Python 3.10 or newer (CI covers 3.10 through 3.14).
- No Node.js or external service is required by the library itself.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests examples
make docs-live         # preview the manual with live reload at http://127.0.0.1:8000
make docs-serve        # build once, then serve the static HTML
make check             # lint, tests with coverage, and documentation checks
```

`make docs-live` rebuilds on every edit under `docs/` or `src/`, and reloads the
open browser page. Pass `DOCS_PORT=8080` to change the port and `DOCS_OPEN=`
to skip opening a browser. `make docs` builds once without serving.

The repository layout:

- `src/zett_agent/` — the runtime published as `zett-agent`.
- `docs/` — the Sphinx manual; `make docs-serve` previews it locally.
- `examples/` — runnable programs, most of which need no API key.

ZettCode, the terminal coding agent built on this runtime, lives in its own
repository: <https://github.com/Chang-LeHung/zettcode>.

## License

MIT
