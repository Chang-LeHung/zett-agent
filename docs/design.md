# Minimal Agent Design

The runtime has one responsibility: repeat model calls and tool execution until
the model returns an answer without tool calls.

## Public API

```python
agent = await Agent.create(
    model,
    tools=[add],
    extensions=[history_extension],
    config=AgentConfig(session_id="calculator-session"),
)
reply = await agent.run(
    "Add 2 and 3",
    config=AgentConfig(session_id="calculator-session"),
)
print(reply.content)
```

## Ownership

- Messages own their content directly.
- Model responses own token usage and finish reasons.
- Tools own argument validation and execution.
- The agent creates a fresh mutable message state for every request.
- Extensions can modify messages at explicit lifecycle hooks.
- Extensions own history restoration, compaction, and optional persistence.

Streaming events make the existing KCS UI work. Provider adapters handle vendor
protocols, including signed thinking blocks needed to replay tool calls.

There is no generic state container, checkpoint system, version lock, or
cancellation token. Use `AgentState` for messages, small lifecycle extensions,
and asyncio task cancellation.
