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
