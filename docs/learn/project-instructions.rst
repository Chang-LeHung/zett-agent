Load project instructions
=========================

Use :class:`~zett_agent.extensions.agents_md.AgentsMdExtension` when a project
has conventions the agent should follow in every request: test commands,
coding standards, or review expectations. Put those conventions in
``AGENTS.md`` rather than repeating them in every user prompt.

Create a project file
---------------------

.. code-block:: markdown
   :caption: workspace/AGENTS.md

   # Project guidance

   - Keep public functions typed.
   - Run the relevant tests before proposing a change.
   - Explain compatibility risks in reviews.
   - Ask before changing persistent data.

Load that directory explicitly:

.. code-block:: python

   from zett_agent.client import create_agent
   from zett_agent.extensions.agents_md import AgentsMdExtension
   from zett_agent.extensions.memory import InMemoryMessageAccumulator
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   instructions = AgentsMdExtension(directory="workspace", include_parents=False)
   client = await create_agent(
       model,
       extensions=[
           InMemoryMessageAccumulator(),
           ToolGuidelinesExtension(),
           instructions,
       ],
   )
   reply = await client.run("Review the proposed change.")

Here ``model`` is a configured adapter from :doc:`providers`. This extension
adds instructions; it does not add filesystem, editing, or shell tools.
Keeping default history and tool guidance requires including them in your
explicit extension list; see :doc:`configuration`.

Choose the discovery scope
--------------------------

With ``include_parents=True`` (the default), discovery checks the configured
directory and each ancestor up to the filesystem root. Files are included
outermost first, innermost last:

.. code-block:: text

   workspace/AGENTS.md               broader project guidance
   workspace/services/AGENTS.md      service guidance
   workspace/services/api/AGENTS.md  more specific guidance

Set ``directory="workspace/services/api"`` to include all three, or
``include_parents=False`` to load only that directory's file. The extension
does not recursively scan descendants or automatically load different files
for each path a tool opens. If you omit ``directory``, it resolves the process's
current working directory at the start of each request. A directory argument
selects guidance only; it does not change where filesystem tools run.

.. warning::

   Parent discovery may include files outside your repository. For an
   application serving untrusted workspaces, use an explicit directory and
   ``include_parents=False`` rather than relying on process-wide ``chdir``.

Inspect and update guidance
---------------------------

.. code-block:: python

   print(instructions.sources())       # Ordered paths that exist.
   print(instructions.instructions())  # Rendered readable guidance.

Avoid logging rendered instructions when they contain sensitive information.
The extension rereads files before each request, so edits apply on the next
turn without recreating the client. Missing, empty, unreadable, or non-UTF-8
files are skipped. Each file is limited to ``max_chars=32_000`` by default;
longer files are included with a truncation marker. Increase the limit only
after considering the model's context budget.

Use ``file_name="PROJECT.md"`` if your application follows another convention.
Paths returned by ``sources()`` do not guarantee readable, nonempty content;
``instructions()`` shows what can actually be included.

Try an offline request
----------------------

The following example creates an isolated project, checks what the model
receives, edits its guidance, and verifies that the next request sees the edit.
All files are temporary.

.. literalinclude:: ../_examples/project_instructions.py
   :language: python
   :pyobject: main

Run from a checkout:

.. code-block:: console

   uv run python docs/_examples/project_instructions.py

Expected output:

.. code-block:: text

   Project guidance: careful reviews
   Project guidance: concise reviews

:download:`Download the complete example <../_examples/project_instructions.py>`.

Instructions, skills, and permissions
------------------------------------------------------------

``AGENTS.md`` is always-loaded guidance for the chosen directory.
:doc:`Skills <on-demand>` are optional task instructions loaded through
``read_skill``. Your application system prompt establishes overall behavior.
None of these is a security boundary: untrusted instructions cannot authorize
a tool or override your application's access policy. Enforce permissions in
the tool handler, middleware, or process isolation.
