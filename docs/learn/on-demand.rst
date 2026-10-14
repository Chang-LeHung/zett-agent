Load tools and skills on demand
========================================

An agent does not need every capability in every conversation. Keep ordinary
tools available from the start, describe skills briefly, and load their full
instructions when a task needs them. For a large toolbox, supported Responses
endpoints can also discover deferred tool definitions through search.

These are two separate choices: a **skill** supplies instructions, while a
**tool** executes a function. Loading instructions does not grant the model
additional permissions.

Create a skill in plain text
----------------------------------------

Store each skill in a directory with a ``SKILL.md`` file:

.. code-block:: text

   skills/
   └── code-review/
       └── SKILL.md

The file needs a name, a short description, and a Markdown body:

.. literalinclude:: ../_examples/skills/code-review/SKILL.md
   :language: markdown
   :caption: skills/code-review/SKILL.md

Use lowercase letters, numbers, and single hyphens in the name. The description
helps the model decide when to read the skill; the body explains how to carry
out the task. Invalid files are skipped, so check ``extension.skills`` if a
skill does not appear.

Choose the directories to load
----------------------------------------

Pass explicit roots so your application controls where instructions come from:

.. code-block:: python

   from pathlib import Path

   from zett_agent.client import create_agent
   from zett_agent.extensions.memory import InMemoryMessageAccumulator
   from zett_agent.extensions.skill import SkillExtension
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   skills = SkillExtension([Path("skills")])
   client = await create_agent(
       model,
       extensions=[
           InMemoryMessageAccumulator(),
           ToolGuidelinesExtension(),
           skills,
       ],
   )
   reply = await client.run("Review this change for correctness.")

The model receives a catalog of names and descriptions and can call
``read_skill(name="code-review")`` to read the complete file. It does not
receive every skill body up front. With a real model, whether it chooses to
read a particular skill depends on your prompt and its capabilities.

Skill discovery happens when you construct the extension. Recreate it after
adding files or changing catalog metadata. Editing an existing skill body
affects future reads, but does not rewrite instructions already in the
conversation. Earlier roots take precedence when names collide.

With no roots, discovery uses the user-level ``~/.zett/skills``,
``~/.agent/skills``, ``~/.claude/skills``, and ``~/.cursor/skills`` directories.
Project-local skills are not included automatically. Passing an explicit
list replaces those defaults; use ``SkillExtension([])`` to discover none.

A skill name is limited to 64 characters and its description to 1,024.
The parser accepts simple YAML-style metadata; it is not a general YAML
configuration loader. Keep names, descriptions, and bodies simple and test
discovery with the same roots your application uses. The loader supplies the
``SKILL.md`` content, not every referenced asset. Add appropriately scoped
file-reading tools if your skill needs accompanying files.

.. note::

   An explicit ``extensions=[...]`` list replaces the optional defaults. The
   example includes history and tool guidance to keep both behaviors.

Try a complete offline example
----------------------------------------

This example uses a scripted model to ask for the review skill, then returns a
final answer. It runs the real tool round trip without a network request.

.. literalinclude:: ../_examples/on_demand.py
   :language: python
   :pyobject: main

From a checkout:

.. code-block:: console

   uv run python docs/_examples/on_demand.py

Expected output:

.. code-block:: text

   Skill loaded: code-review
   Review correctness, edge cases, and tests.

The :download:`complete program <../_examples/on_demand.py>` uses
:download:`this SKILL.md file <../_examples/skills/code-review/SKILL.md>`.
Keep the skill at ``skills/code-review/SKILL.md`` beside the downloaded script.

Defer a large toolbox
----------------------------------------

With a compatible Responses endpoint or the Anthropic Messages API, mark
rarely used tools as deferred and add
:class:`~zett_agent.extensions.tool_search.ToolSearchExtension`:

.. code-block:: python

   from zett_agent.extensions.tool_search import ToolSearchExtension
   from zett_agent.tools.base import tool

   @tool(deferred=True, guidelines="Use to check current stock before promising availability.")
   def warehouse_stock(sku: str) -> str:
       """Look up available warehouse stock by product SKU."""
       return inventory_service.stock(sku)

   client = await create_agent(
       responses_model,
       tools=[warehouse_stock],
       extensions=[
           InMemoryMessageAccumulator(),
           ToolGuidelinesExtension(),
           ToolSearchExtension(),
       ],
   )

Here ``responses_model`` is a compatible adapter configured with
``response=True`` (or an :class:`~zett_agent.providers.anthropic.AnthropicProvider`);
``inventory_service`` is your application's inventory client. The model
searches for relevant capabilities and receives matching definitions before
calling them. Tools without ``deferred=True`` remain visible as ordinary tools.

The ranking always happens locally, with BM25 over each tool's name,
description, parameter descriptions, snippet, and guidelines. Only the wire
protocol differs:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Provider
     - How a deferred tool reaches the model
   * - Responses API
     - The search answers with ``tool_search_output`` definitions; deferred tools are not sent up front.
   * - Anthropic Messages API
     - Deferred tools are declared with ``defer_loading``; the search answers with ``tool_reference`` blocks naming the tools to load. Their guidelines ride in the description.

A deferred tool is skipped by
:class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension`, so its
description is the only place the model reads its guidance. When that extension
is registered, its ``snippets`` and ``guidelines`` switches decide what the
description carries, whatever its ``placement``; without it, a deferred tool
carries its guidelines only.

The difference is one field of the :class:`~zett_agent.model.AgentModel`
protocol, ``native_deferred_tools``. It is ``True`` on ``AnthropicProvider`` and
``False`` on every other built-in adapter. A custom adapter sets it to ``True``
when its protocol can load a deferred tool by reference.

Tell the model what it can search for
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A deferred tool is invisible until it is loaded, so a model with only search
keywords to go on has to guess what exists. ``ToolSearchExtension`` therefore
lists the deferred tools (each name and description, plus its snippet and
guidelines) and puts the list where you choose:

.. code-block:: python

   from zett_agent.extensions.tool_search import ToolCatalogPlacement, ToolSearchExtension

   # The default: appended to the search tool's own description.
   ToolSearchExtension()

   # A "# Searchable tools" section of the system instructions instead.
   ToolSearchExtension(catalog=ToolCatalogPlacement.SYSTEM_PROMPT)

   # Shorter entries, or no list at all.
   ToolSearchExtension(catalog_snippets=False, catalog_guidelines=False)
   ToolSearchExtension(catalog=ToolCatalogPlacement.NONE)

The list is independent of
:class:`~zett_agent.extensions.tool_guidelines.ToolGuidelinesExtension` and has
its own switches, so the two combine in any placement and any order. The list
names every deferred tool registered for the request rather than shrinking as
tools are found: a list that changed on every search would change the request
prefix each time and defeat provider prompt caching. With many deferred tools
the list itself costs tokens, so trim it with the switches or use
``NONE`` and rely on search keywords.

.. important::

   Client-side tool search requires the Responses tool-search protocol or the
   Anthropic Messages API. Chat Completions, Google, and Ollama do not support
   this extension. Skill loading uses an ordinary local tool and does not have
   that restriction.

What this does not do
----------------------------------------

Skills are editable instructions, not an automatic learning system. Neither
extension generates tools, learns from a conversation, edits skill files on
its own, or unloads running code. Tools execute with the permissions of your
process. Treat skill files and search results as input, not authorization to
perform privileged actions.

To create and evolve Python tools during a session, opt in to
:doc:`jit-harness`. It is a separate trusted-code feature, not an implicit
behavior of skill loading or deferred tool search.

Next: :doc:`sessions` to keep conversations, or
:doc:`../extending/built-ins` to choose other capabilities.

Related API: :class:`~zett_agent.extensions.skill.SkillExtension`,
:class:`~zett_agent.extensions.tool_search.ToolSearchExtension`.
