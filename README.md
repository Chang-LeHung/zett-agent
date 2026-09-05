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

from kcs_agent import Agent, DeepSeekProvider, UserMessage, tool


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


async def main() -> None:
    model = DeepSeekProvider("deepseek-v4-flash", os.environ["DEEPSEEK_API"])
    try:
        agent = Agent(model, tools=[add])
        reply = await agent.run(UserMessage(content="Use add to calculate 2 + 3."))
        print(reply.content)
    finally:
        await model.aclose()


asyncio.run(main())
```

## Messages and history

Use `SystemMessage(content=...)`, `UserMessage(content=...)`,
`AssistantMessage(content=..., tool_calls=...)`, and
`ToolMessage(tool_call_id=..., name=..., content=...)`.

There are no `data`/`metadata` wrappers or message generics.
User messages also support the existing text/image content blocks.

Each call supplies the latest user message. Pass earlier messages explicitly:

```python
reply = await agent.run(
    UserMessage(content="What was my previous question?"),
    history=[
        UserMessage(content="What is 2 + 3?"),
        AssistantMessage(content="5"),
    ],
)
```

The agent copies the history and keeps no conversation state between calls.
Applications own persistence and compaction. KCS prepares its snapshot and replay
tail before calling this loop.

## Tools

`@tool` uses the function's first docstring paragraph as its description.
Pydantic generates the argument schema and validates input before execution.
Use `Annotated[T, Field(description="...")]` for parameter descriptions.

Optional usage rules use `@tool(guidelines="...")` and are appended to the system
instructions. There is no docstring DSL, separate parser, or tool extension.

## Streaming

`agent.stream(message, history=..., reasoning_effort=...)` yields
`AgentEvent` objects in order. Events expose text/reasoning deltas, model responses,
tool calls, tool results, and the final answer. `run()` collects that stream and
returns the final `AssistantMessage`.

Tool errors are returned to the model. Model errors propagate to the application.
Cancel the consuming asyncio task to interrupt generation. A running synchronous
tool cannot be forcibly stopped by task cancellation.

Native provider details stay in `providers.py`; they are not part of the core loop.
Existing adapters: OpenAI, DeepSeek, Anthropic, Google, and Ollama.

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
