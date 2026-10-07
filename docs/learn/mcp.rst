Connect MCP tools
=================

Use :class:`~zett_agent.extensions.mcp.McpExtension` to connect tools supplied
by Model Context Protocol servers. The model sees ordinary tool definitions;
when it requests one, the extension calls the server and returns its result.
Choose this for an existing MCP integration instead of rewriting its tools as
local Python functions.

Connect a Streamable HTTP server
--------------------------------

.. code-block:: python

   import os

   from zett_agent.client import create_agent
   from zett_agent.extensions.mcp import McpExtension, McpHttpServer
   from zett_agent.extensions.memory import InMemoryMessageAccumulator
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   mcp = McpExtension(
       servers=[
           McpHttpServer(
               name="catalog",
               url=os.environ["MCP_URL"],
               headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
           ),
       ],
   )
   client = await create_agent(
       model,
       extensions=[InMemoryMessageAccumulator(), ToolGuidelinesExtension(), mcp],
   )
   reply = await client.run("Find the product matching this description.")

This is an opt-in network example: configure ``model`` using :doc:`providers`,
and set the URL and token for a trusted server. Omit ``headers`` if the server
does not require them. Headers are sent for the transport lifecycle and are
excluded from the server object's ``repr``, but your application must still
keep credentials out of logs and source control.

Launch a local stdio server
------------------------------------------------------------

.. code-block:: python

   import sys
   from pathlib import Path

   from zett_agent.extensions.mcp import McpStdioServer

   mcp = McpExtension(
       servers=[
           McpStdioServer(
               name="local",
               command=sys.executable,
               args=("server.py",),
               cwd=Path("trusted-server").resolve(),
           ),
       ],
   )

Replace the path and arguments with a server you have installed. Commands and
arguments are passed separately, not interpolated into a shell string.
The process still runs with your application's permissions. Launch only
trusted servers and restrict their filesystem, environment, and network
access at the process or container boundary.

Use a configuration file
------------------------

.. code-block:: json
   :caption: mcp.json

   {
     "mcpServers": {
       "catalog": {
         "type": "streamable-http",
         "url": "https://your-server.example/mcp"
       },
       "local": {
         "command": "python",
         "args": ["/absolute/path/to/server.py"]
       }
     }
   }

Load it with ``McpExtension(config_path="mcp.json")``. Both ``servers`` and
``mcpServers`` are accepted root keys, but a file must use only one. HTTP
entries can include ``headers``; stdio entries can include ``env`` and ``cwd``.
Paths and commands in the file are your responsibility; do not assume they
resolve relative to the configuration file. There is no automatic ``${VAR}``
interpolation for secrets; construct server definitions in Python when values
come from environment variables.

``McpExtension()`` looks for ``~/.zett/mcp.json``. Supplying an explicit
``servers`` list avoids that implicit lookup. A custom ``config_path`` and a
server list can be combined; duplicate server names are rejected. A missing
configuration file loads no servers, while malformed configuration raises an
error when the extension is constructed.

Tool names and results
----------------------

A server named ``catalog`` with a tool named ``search`` exposes
``catalog__search`` by default. Namespacing avoids collisions with local tools
and other servers. ``namespace_tools=False`` keeps the original names, but you
must then ensure every registered name is unique.

Server-provided input schemas describe the tools. Remote validation and your
application policy remain important; MCP tools do not pass through the
``@tool`` function validator. If a result has ``structured_content``, it is
returned as that structured value; otherwise content blocks are serialized as
JSON data. Image blocks are not automatically promoted into typed vision
messages. A remote ``is_error`` result becomes an ordinary tool failure the
model can see.

Connections and failure
-----------------------

The extension connects and discovers the complete paginated tool list at the
start of **each request**, then keeps those connections for the model/tool
round trips. Success, failure, and cancellation close them. A stdio process
may therefore be launched again on the next request.

If connection or discovery fails, setup fails and the request raises; it does
not silently omit a broken server. An ordinary failure during a tool call
produces ``TOOL_FAILED`` and permits the model to continue. Bound the request
with an application timeout and choose which remote error text can be shown.
The extension does not grant remote servers additional authorization.

Verify the integration offline
------------------------------

.. literalinclude:: ../_examples/mcp_tools.py
   :language: python
   :pyobject: main

Run ``uv run python docs/_examples/mcp_tools.py`` from a checkout:

.. code-block:: text

   MCP tool result: 42
   MCP connection closed.

The :download:`complete example <../_examples/mcp_tools.py>` supplies an
in-memory MCP client through ``client_factory``. It verifies registration,
dispatch, structured results, and transport cleanup without starting a server.

Next: :doc:`tools` for local tool behavior, or
:doc:`../extending/server-tools` for tools executed by a model provider rather
than an MCP server.
