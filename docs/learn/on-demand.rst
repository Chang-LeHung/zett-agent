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

With a compatible Responses endpoint, mark rarely used tools as deferred and
add :class:`~zett_agent.extensions.tool_search.ToolSearchExtension`:

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
``response=True``; ``inventory_service`` is your application's inventory
client. The model searches for relevant capabilities and receives matching
definitions before calling them. Tools without ``deferred=True`` remain
visible as ordinary tools.

.. important::

   Client-side tool search requires the Responses tool-search protocol.
   Chat Completions, Anthropic, Google, and Ollama do not support this extension.
   Skill loading uses an ordinary local tool and does not have that restriction.

What this does not do
----------------------------------------

Skills are editable instructions, not an automatic learning system. Neither
extension generates tools, learns from a conversation, edits skill files on
its own, or unloads running code. Tools execute with the permissions of your
process. Treat skill files and search results as input, not authorization to
perform privileged actions.

Next: :doc:`sessions` to keep conversations, or
:doc:`../extending/built-ins` to choose other capabilities.

Related API: :class:`~zett_agent.extensions.skill.SkillExtension`,
:class:`~zett_agent.extensions.tool_search.ToolSearchExtension`.
