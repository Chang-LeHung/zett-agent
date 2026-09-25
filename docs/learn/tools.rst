Define and register tools
=============================

Tools are typed operations the model can request. A call carries a name,
arguments, and an ID; a ToolMessage answers that ID. The Agent handles validation,
execution, serialization, and another model step after the result.

Write the function and its model-facing documentation
---------------------------------------------------------

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: add

``@tool`` reads type annotations to generate JSON Schema. ``Args`` adds parameter
descriptions; ``Snippet`` and ``Guidelines`` supply model-facing usage guidance.
Both ``@tool`` and ``@tool(name="public_name")`` are supported.

Register it with the client
-------------------------------

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: main

The model first requests the tool, the runtime appends its result, and the model
uses that result to answer. Run the full :doc:`../examples/streaming-tools`
program to see reasoning, arguments, result, and answer in sequence.

Direct Python invocation has a different syntax
---------------------------------------------------

Decoration returns an :class:`~zett_agent.AgentTool`, not the original function.
To test it directly::

    value = await add({"left": 20, "right": 22})
    assert value == 42

The snippet ``add(left=20, right=22)`` describes the model-facing call, not this
Python interface. Decorated sync functions run in a worker thread; async ones
are awaited on the event loop. Invalid arguments fail schema validation.

Filesystem and shell tools
------------------------------

``FileSystemExtension(read_only=True)`` grants read, glob, and grep.
Writable mode adds editing tools; ``CodingExtension`` also adds shell execution.
They run relative to the host process's working directory, with its permissions.
They are not a security sandbox. Restrict the process/container and validate
your trust boundary before enabling write or shell capabilities.

Long outputs are bounded so they cannot occupy the entire model context.
File reads support windows; search results have limits; shell previews retain
both startup and final output while preserving full output in a reported file.
Inspect each :doc:`tool schema <../_generated/group-tools>` for exact parameters.

Register tools dynamically: :doc:`../extending/first-extension`.

Local tool search
---------------------

Keep optional tools out of the request and load them on demand with
:class:`~zett_agent.ToolSearchExtension`:

.. code-block:: python

   agent = await Agent.create(
       model,
       tools=[warehouse_stock, billing_lookup],
       extensions=[ToolSearchExtension()],
   )

Tools are registered as usual, and the ones to discover are marked
``deferred=True``. The extension adds its internal ``tool_search`` tool and hides
every deferred definition from provider requests, so a first request declares
``{"type": "tool_search", "execution": "client"}`` plus the schemas of the tools
that are *not* deferred. Deferred tools stay registered for local execution,
their definitions reach the model through ``tool_search_output``, and the model
then calls the discovered tool as an ordinary ``function_call``. Each returned
definition carries the tool's name, description, parameter schema, and
guidelines, so a deferred tool's rules arrive together with the definition that
makes it callable.

Tools without ``deferred`` are untouched: they keep their ordinary function
schema, and :class:`~zett_agent.ToolGuidelinesExtension` renders their snippets
and guidelines as before.

The extension defines and registers one real ``@tool``-decorated search function.
That function directly uses :class:`~zett_agent.BM25Search` over the currently
registered tools' names, descriptions, Args, snippets, and guidelines. There is
no replacement handler, sentinel return value, or custom search callback.

The search tool declares its own parameters, so the model passes one or more
keyword ``queries`` plus an optional ``score`` that overrides the default
threshold for that search. The schema comes from the search function's signature,
and its description travels in the ``tool_search`` declaration itself. The search
tool is not deferred, so adding :class:`~zett_agent.ToolGuidelinesExtension` also
renders its snippet and guidelines in the prompt — pair the two when the model
should read that guidance. Set ``min_score`` on the extension or on
:class:`~zett_agent.BM25Search` to raise the bar for every search; results scoring
at or below it are dropped. Omitting ``score`` uses the configured default;
passing ``score=0.0`` explicitly overrides that default.

To answer searches yourself, mark one tool with ``local_tool_search=True``. The
Responses adapter then declares ``tool_search`` with ``execution: "client"``
instead of exposing that tool as a function, and every search request arrives as
a normal local tool call:

.. code-block:: python

   from zett_agent import tool
   from openai.types.responses import FunctionToolParam

   @tool(local_tool_search=True, guidelines="Use to find optional tools.")
   def find_tools(queries: list[str]) -> list[FunctionToolParam]:
       """Return the tool definitions that match the requested queries.

       Args:
           queries: What the model wants to do.
       """
       return [
           FunctionToolParam(
               type="function",
               name="warehouse_stock",
               description="Return stock for one warehouse",
               parameters={"type": "object", "properties": {"warehouse": {"type": "string"}}},
               strict=None,
           )
       ]

The handler receives the provider's search arguments as its keyword arguments and
returns the OpenAI function definitions the model may load; the Agent validates
that list against ``FunctionToolParam`` and answers with ``tool_search_output``.
Keep discoverable tools out of the request yourself, for example by filtering
``request.tools`` in ``on_model_request``, and mark a tool ``deferred=True`` when
the endpoint may keep a declared schema unloaded. Chat Completions, Anthropic,
Google, and Ollama reject ``local_tool_search`` tools.
