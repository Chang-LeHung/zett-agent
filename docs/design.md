# Minimal Agent Design

The runtime has one responsibility: repeat model calls and tool execution until
the model returns an answer without tool calls.

## Public API

```python
agent = Agent(model, tools=[add])
reply = await agent.run(UserMessage(content="Add 2 and 3"), history=[])
print(reply.content)
```

## Ownership

- Messages own their content directly.
- Model responses own token usage and finish reasons.
- Tools own argument validation and execution.
- The agent owns only the local messages list and bounded loop.
- The application owns sessions, history, storage, compaction, and measurements.

Streaming events make the existing KCS UI work. Provider adapters handle vendor
protocols, including signed thinking blocks needed to replay tool calls.

There is no extension framework, generic state container, checkpoint system,
version lock, or cancellation token. Use explicit arguments and asyncio task
cancellation. Add new abstractions only when a concrete new requirement needs one.
