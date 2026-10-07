"""Run a real provider adapter against an in-memory HTTP transport."""

import asyncio
import json

import httpx

from zett_agent.client import create_agent
from zett_agent.model import ReasoningEffort, RetryOptions
from zett_agent.providers.openai import OpenAIProvider


def respond(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["messages"][-1]["content"] == "Hello"
    assert payload["stream"] is True
    chunks = [
        {"choices": [{"delta": {"content": "Hello from a mock endpoint."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body + "data: [DONE]\n\n",
    )


async def main() -> None:
    model = OpenAIProvider(
        model="fixture-model",
        api_key="offline-test-key",
        transport=httpx.MockTransport(respond),
        retry=RetryOptions(max_retries=0),
    )
    try:
        client = await create_agent(model, reasoning_effort=ReasoningEffort.OFF)
        reply = await client.run("Hello")
        assert reply.content == "Hello from a mock endpoint."
        print(f"Mock provider reply: {reply.content}")
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
