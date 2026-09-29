"""Keep registered tools behind client-side tool search."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Protocol
from weakref import WeakKeyDictionary

# ``tools.base`` builds the search tool's schema with ``get_type_hints``, which
# evaluates annotations against this module's globals, so the SDK type stays a
# real import even though it is only used for typing.
from openai.types.responses import FunctionToolParam

from ..agent import AgentRunContext
from ..model import ModelEvent, ModelRequest
from ..tools.base import (
    AgentTool,
    render_tool_search_text,
    tool,
)
from .base import AgentExtension, ModelRequestNext
from .events import CompactionEvent, ExtensionEvent

#: Public name of the internal search tool; it is never sent as a function.
TOOL_SEARCH_TOOL_NAME = "tool_search"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens plus individual CJK characters."""
    return _TOKEN_PATTERN.findall(text.lower()) + [char for char in text if "\u4e00" <= char <= "\u9fff"]


class BM25Search:
    """Rank a set of tools with BM25 over their names and descriptions.

    Every model-facing text of a tool is indexed: its name (split on ``_``), its
    description, its parameter descriptions from the input schema, its snippet,
    and its guidelines. Terms that appear in no document score zero, so a query
    that matches nothing returns an empty list. Ties keep tool order, and
    ``limit`` caps how many definitions one search sends back.

    The catalog is fixed at construction, so term counts, document frequencies,
    and the average document length are computed once and reused by every search.

    Args:
        tools: Tools this index may return.
        limit: Maximum number of tools to return.
        min_score: Minimum BM25 score a tool must exceed to be returned.
        k1: Term-frequency saturation.
        b: Length-normalization strength.

    Examples:
        Rank a catalog for one query::

            matches = BM25Search([warehouse_stock, billing_lookup]).search(["warehouse stock"])
    """

    def __init__(
        self,
        tools: Sequence[AgentTool],
        *,
        limit: int = 20,
        min_score: float = 0.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tools = tuple(tools)
        self.limit = limit
        self.min_score = min_score
        self.k1 = k1
        self.b = b
        self._documents = [_tokenize(render_tool_search_text(candidate)) for candidate in self.tools]
        self._counts = [Counter(document) for document in self._documents]
        self._frequencies = Counter(term for document in self._documents for term in set(document))
        self._average_length = (
            sum(len(document) for document in self._documents) / len(self._documents) if self._documents else 0.0
        )

    def search(self, queries: Sequence[str], *, min_score: float | None = None) -> list[AgentTool]:
        """Return the tools scoring above the threshold, best first."""
        terms = [term for query in queries for term in _tokenize(query)]
        if not terms or self.limit <= 0 or not self.tools:
            return []
        threshold = self.min_score if min_score is None else min_score
        scored: list[tuple[float, int, AgentTool]] = []
        for index, (candidate, counts, document) in enumerate(
            zip(self.tools, self._counts, self._documents, strict=True)
        ):
            score = 0.0
            for term in set(terms):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._frequencies[term]
                inverse = math.log(1 + (len(self.tools) - document_frequency + 0.5) / (document_frequency + 0.5))
                length_norm = 1 - self.b + self.b * len(document) / self._average_length
                score += inverse * frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
            if score > threshold:
                scored.append((score, index, candidate))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [candidate for _, _, candidate in scored[: self.limit]]


def _definition(tool: AgentTool) -> FunctionToolParam:
    """Render one tool as the definition the model loads.

    A deferred tool never reaches the provider schema and
    :class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension` skips it, so this result is the
    only place the model can learn its rules: the guidelines travel with the
    description.
    """
    description = tool.description
    if tool.guidelines:
        rules = "\n".join(f"- {guideline}" for guideline in tool.guidelines)
        description = f"{description}\n\nGuidelines:\n{rules}"
    return FunctionToolParam(
        type="function",
        name=tool.name,
        description=description,
        parameters=tool.parameters,
        strict=None,
    )


class ToolSearchStorage(Protocol):
    """Async key/value boundary used to remember one session's sent definitions.

    The extension stores a JSON array of tool names per session key and only
    calls these three methods, so an application can back the contract with SQL,
    Redis, or a file. Values are opaque strings; a missing key is not an error.
    """

    async def get(self, key: str) -> str | None:
        """Return the stored value, or None when the key is unset."""

    async def set(self, key: str, value: str) -> None:
        """Store one value, replacing whatever the key held."""

    async def delete(self, key: str) -> None:
        """Remove one key; deleting a missing key is a no-op."""


class ToolSearchExtension(AgentExtension):
    """Search registered local tools with BM25 through the Responses API.

    Tools marked ``deferred=True`` are the searchable ones: this extension adds
    one request-scoped ``tool_search`` tool and removes every deferred definition
    from provider requests. Matching definitions reach the model through
    ``tool_search_output``; their handlers remain locally registered. Tools
    without ``deferred`` keep flowing to the model as ordinary functions, so
    :class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension` renders their guidance as usual.

    The search tool's signature and docstring supply its schema, description,
    snippet, and guidelines, just like any other ``@tool``-decorated function.
    Definitions the session already received are not offered twice: a second
    search in a later turn would otherwise repeat a name the conversation
    already declares, which strict endpoints reject. The record covers one
    session; a compaction drops it by default, because the checkpoint replaces
    the messages that declared those definitions.

    Args:
        limit: Maximum number of matching tools considered per search.
        min_score: Default score threshold. A search call may override it with
            ``score``; only tools scoring strictly above it are returned.
        k1: BM25 term-frequency saturation.
        b: BM25 length-normalization strength.
        storage: Optional key/value store holding one session's sent
            definitions. None keeps the record in memory for the lifetime of
            this extension, which covers several turns of one process but not a
            restart. Provide one to make the record durable.
        resend_definitions_after_compaction: Re-offer definitions the session
            already received once a compaction replaces the messages that
            declared them. False keeps suppressing them after a checkpoint, so
            the model keeps the tools it loaded earlier but cannot load them
            again by searching.

    .. note::
        Clearing the record after a compaction matters because the checkpoint
        no longer declares the definitions the model loaded earlier. Keeping it
        would hide those tools from the model for the rest of the conversation.

    Examples:
        Keep optional capabilities behind search::

            agent = await Agent.create(
                model,
                tools=[warehouse_stock, billing_lookup],
                extensions=[ToolSearchExtension(storage=redis_store)],
            )

    .. note::
        Client-side tool search needs the Responses API. Chat Completions,
        Anthropic, Google, and Ollama reject ``local_tool_search`` tools, and an
        endpoint that ignores the protocol never issues a search.
    """

    def __init__(
        self,
        *,
        limit: int = 20,
        min_score: float = 0.0,
        k1: float = 1.5,
        b: float = 0.75,
        storage: ToolSearchStorage | None = None,
        resend_definitions_after_compaction: bool = True,
    ) -> None:
        self._limit = limit
        self._min_score = min_score
        self._k1 = k1
        self._b = b
        self._storage = storage
        self._resend_after_compaction = resend_definitions_after_compaction
        self._session_sent: dict[str, set[str]] = {}
        self._run_sent: WeakKeyDictionary[AgentRunContext, set[str]] = WeakKeyDictionary()
        self._lock = asyncio.Lock()

    async def on_tool(self, context: AgentRunContext) -> None:
        """Register the decorated BM25 search tool for this request."""
        context.register_tool(self._build_search_tool(context))

    async def on_model_request(
        self,
        context: AgentRunContext,
        request: ModelRequest,
        call_next: ModelRequestNext,
    ) -> AsyncIterator[ModelEvent]:
        """Hide deferred definitions; visible tools keep their ordinary schema."""
        filtered = replace(request, tools=tuple(tool for tool in request.tools if not tool.deferred))
        async with aclosing(call_next(filtered)) as events:
            async for event in events:
                yield event

    async def on_event(self, context: AgentRunContext, event: ExtensionEvent) -> None:
        """Let the conversation rediscover definitions a compaction removed.

        The checkpoint replaces the ``tool_search_output`` items that declared
        them, so the model must be able to search them up again instead of being
        told they are already loaded.
        """
        if isinstance(event, CompactionEvent) and self._resend_after_compaction:
            async with self._lock:
                await self._clear_sent(context)

    @staticmethod
    def _storage_key(session_id: str) -> str:
        """Return the storage key that holds one session's sent definitions."""
        return f"tool_search:sent:{session_id}"

    async def _load_sent(self, session_id: str) -> set[str]:
        """Return the definitions this session already received."""
        if self._storage is None:
            return set(self._session_sent.get(session_id, ()))
        payload = await self._storage.get(self._storage_key(session_id))
        if not payload:
            return set()
        try:
            names = json.loads(payload)
        except json.JSONDecodeError:
            # A damaged record must not stall the conversation. Forgetting it
            # risks one duplicate definition, which is recoverable; raising
            # would fail every later search in the session.
            return set()
        if not isinstance(names, list):
            return set()
        return {str(name) for name in names}

    async def _store_sent(self, session_id: str, names: set[str]) -> None:
        """Remember the definitions this session received, durably when asked."""
        self._session_sent[session_id] = set(names)
        if self._storage is not None:
            await self._storage.set(self._storage_key(session_id), json.dumps(sorted(names)))

    async def _clear_sent(self, context: AgentRunContext) -> None:
        """Forget every definition recorded for the request's session."""
        self._run_sent.pop(context, None)
        session_id = context.config.session_id
        self._session_sent.pop(session_id, None)
        if self._storage is not None:
            await self._storage.delete(self._storage_key(session_id))

    async def _select_new(self, context: AgentRunContext, matches: Sequence[AgentTool]) -> list[AgentTool]:
        """Keep the matches this conversation has not declared yet.

        Parallel tool calls can run several searches at once, so the read,
        filter, and write happen under one lock; otherwise two calls that
        overlap would both hand out the same definition.
        """
        session_id = context.config.session_id
        async with self._lock:
            sent = set(self._run_sent.get(context, ()))
            sent |= await self._load_sent(session_id)
            discovered = [match for match in matches if match.name not in sent]
            names = sent | {match.name for match in discovered}
            self._run_sent[context] = names
            if discovered:
                await self._store_sent(session_id, names)
            return discovered

    def _build_search_tool(self, context: AgentRunContext) -> AgentTool:
        """Define the actual tool, backed by this session's discovery record."""

        # We must not resend definitions the model already received. A
        # request-local set is not enough: the definitions it hands out live on
        # in the conversation as ``tool_search_output`` items, so a second
        # search in a later turn would re-offer names the history already
        # declares. A strict endpoint answers that with "the tool name is
        # invalid or duplicated" and the conversation cannot send any request
        # (seen with a Responses-compatible DeepSeek gateway). The record is
        # therefore kept per session in ``_session_sent``/``storage``, and
        # ``on_event`` clears it after a compaction folds away those outputs,
        # unless the caller opted out with ``resend_definitions_after_compaction``.

        @tool(name=TOOL_SEARCH_TOOL_NAME, local_tool_search=True)
        async def search(queries: list[str], score: float = self._min_score) -> list[FunctionToolParam]:
            """Search registered tools by BM25 keyword relevance.

            Args:
                queries: Keyword queries describing the capabilities to find; one search may ask for several.
                score: Minimum BM25 score to accept; results at or below it are dropped.

            Snippet:
                tool_search(queries=["warehouse stock inventory"])

            Guidelines:
                - Use to discover deferred tools that are not listed in the tool definitions.
                - Send short keyword queries; BM25 searches tool names, descriptions, Args, snippets, and guidelines.
                - Previously returned definitions are not repeated; use the tools already found in this conversation.

            Returns:
                Newly discovered OpenAI function definitions, ordered by relevance.
            """
            searchable = tuple(candidate for candidate in context.tools.values() if candidate.deferred)
            index = BM25Search(searchable, limit=self._limit, k1=self._k1, b=self._b)
            matches = index.search(queries, min_score=score)
            discovered = await self._select_new(context, matches)
            return [_definition(match) for match in discovered]

        return search
