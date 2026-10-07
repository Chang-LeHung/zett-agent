Define and register tools
=========================

Tools are the operations you let the model request. The runtime validates the
arguments, runs the function, appends the result, and asks the model to continue
— you only write the function.

Write a typed function
----------------------

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: add

``@tool`` reads the type annotations to build a JSON Schema. The docstring is
model-facing documentation: ``Args`` describes each parameter, ``Snippet`` shows
the call shape, and ``Guidelines`` states when to use the tool. ``@tool`` and
``@tool(name="public_name")`` both work.

A description and at least one nonempty guideline are required. For a small
tool, provide the guideline on the decorator:

.. code-block:: python

   from zett_agent.tools.base import tool

   @tool(guidelines="Use for exact integer addition.")
   def add(left: int, right: int) -> int:
       """Add two integers exactly."""
       return left + right

Guidelines and snippets help the model choose the right operation. They are
not permission checks. Decorator ``guidelines`` or ``snippet`` values override
the corresponding docstring sections.

Design arguments the model can supply
------------------------------------------------------------

Every parameter needs a type annotation. Named positional-or-keyword and
keyword-only parameters work; positional-only arguments, ``*args``, and
``**kwargs`` do not. Defaults make arguments optional, and Pydantic-supported
types such as ``Literal``, lists, and nested models can express constraints.

.. code-block:: python

   from typing import Literal

   @tool(guidelines="Choose a supported format for the answer.")
   def choose_format(style: Literal["brief", "detailed"] = "brief") -> str:
       """Choose an answer format."""
       return style

Validation rejects unknown fields and invalid inputs before the function runs,
but it follows Pydantic's normal coercion rules rather than guaranteeing strict
JSON types. Add explicit constraints or checks when conversions are unsafe.
Names must be unique across both static and extension-registered tools. Keep
them stable because they are the model-facing API, not just implementation names.

Register it
-----------

Pass tools to the client; the model can now request them.

.. code-block:: python

   client = await create_agent(model, tools=[add])

The model requests the call, the runtime executes it, and the model answers using
the result. Run :doc:`../examples/streaming-tools` to watch reasoning, arguments,
the result, and the final answer in order.

Calling a tool from Python
--------------------------

Decoration returns an :class:`~zett_agent.tools.base.AgentTool`, which is invoked
with a mapping of arguments:

.. code-block:: python

   value = await add({"left": 20, "right": 22})

Synchronous functions run in a worker thread; ``async def`` tools are awaited on
the event loop. Invalid arguments fail schema validation before your code runs.

Choose a result type
--------------------

* Strings reach the model as plain text without extra JSON quoting.
* JSON-compatible values, Pydantic models, and dataclasses are serialized to JSON.
* ``TextContent``, ``ImageContent``, or a nonempty list of these becomes typed
  multimodal tool content; the selected model must support images.

Return a useful, bounded result instead of an open file, client object, or
arbitrary exception. Include the facts the model needs for its next step,
and avoid leaking secrets or returning unnecessarily large data. A return
annotation documents the function; the decorator does not enforce an output
schema for your result.

Handle tool failures safely
------------------------------------------------------------

An ordinary exception from a handler or argument validation becomes an
unsuccessful ``ToolMessage`` and a ``TOOL_FAILED`` event. The next model call
sees ``str(exception)`` and may retry with better arguments or explain the
failure. The original exception is available to callbacks and after-tool hooks.
Cancellation and other ``BaseException`` subclasses are not converted into
recoverable tool results.

Direct Python calls such as ``await add({...})`` still raise the exception.
Sanitize integration errors yourself: exception text from a tool can expose
credentials, paths, or service responses to the model. An exception in a
``before_tool`` lifecycle hook fails the whole request; use tool middleware
when you want a recoverable rejection. See :doc:`../extending/middleware`.

Control concurrent execution
----------------------------

Tools default to ``ToolExecutionMode.PARALLEL``. A model response with several
parallel tools can execute them concurrently; their result events arrive as
each handler finishes. For an operation that needs exclusive local execution,
mark it serial:

.. code-block:: python

   from zett_agent.tools.base import ToolExecutionMode

   @tool(
       execution_mode=ToolExecutionMode.SERIAL,
       guidelines="Use to prepare a mutation; obtain approval before applying it.",
   )
   def prepare_change(description: str) -> str:
       """Prepare a proposed change without applying it."""
       return description

The runtime finishes the parallel batch before running serial tools one at a
time. ``SERIAL`` is not a global lock across sessions or workers. To preserve
the model's call order for the whole response, set
``parallel_tool_call=False`` on the agent or request. Shared mutable resources
still need application-level synchronization and authorization.

Filesystem and shell tools
--------------------------

``FileSystemExtension(read_only=True)`` adds read, image viewing, glob, and grep.
Writable mode also adds write, replacement, and deletion tools;
``CodingExtension`` adds shell execution.

These tools run with the permissions of your process, relative to its working
directory. They are not a security sandbox: restrict the process or container,
and add approval rules before enabling write or shell access for untrusted
input. Long outputs are bounded so a single command cannot consume the whole
context; full output is preserved in a reported file when truncated.

Load optional tools on demand
-----------------------------

For a large toolbox, :class:`~zett_agent.extensions.tool_search.ToolSearchExtension`
keeps tools marked ``deferred=True`` out of the initial provider request and
lets the model search for their definitions. This requires a compatible
Responses endpoint; ordinary tool calling works with all built-in providers.

See :doc:`on-demand` for setup, a skill-loading example, and the distinction
between loading instructions and executing tools.

Next: :doc:`sessions` to keep the conversation, or
:doc:`../extending/first-extension` to register tools dynamically.
