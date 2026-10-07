Compact without losing history
==================================

Summarize older turns while keeping the original stored messages. Recent turns
remain available in full, so a long conversation can continue within your
configured context budget.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/compaction.py

The output includes:

.. code-block:: text

   Raw messages: 6; checkpoint boundary: 3; raw tail: 3

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Add compaction to an application
------------------------------------------------------------

.. code-block:: python

   from zett_agent.client import create_agent
   from zett_agent.extensions.compaction import CompactionExtension
   from zett_agent.extensions.sqlite import SQLiteSessionExtension
   from zett_agent.extensions.tool_guidelines import ToolGuidelinesExtension

   history = SQLiteSessionExtension("sessions.sqlite")
   client = await create_agent(
       model,
       extensions=[
           history,
           ToolGuidelinesExtension(),
           CompactionExtension(
               summarizing_model,
               max_tokens=40_000,
               keep_recent_tokens=8_000,
           ),
       ],
   )

``model`` is your primary adapter; ``summarizing_model`` is an adapter capable
of summarizing context as plain text without tool calls. They can be the same
instance. If you omit the model passed to ``CompactionExtension``, it uses the
request's model. Close the storage and any providers you own at application
shutdown; see :doc:`../learn/application`.

The numbers above are illustrative, **not** a detected model limit. Set them
for the model and expected tool outputs you actually use. Automatic compaction
runs before a model step when its estimated context crosses ``max_tokens``;
the estimate includes system instructions and tool definitions. Reported input
usage from a previous model step can improve that estimate.

``keep_recent_tokens`` is a minimum, not a hard upper bound. The containing
user turn is kept whole with its assistant/tool messages. One oversized current
turn cannot be solved by splitting it; reduce its input/tool results or choose
a larger context window. The built-in text tokenizer is an estimate, especially
for images, and does not discover your endpoint's actual context capacity.

What to configure
-----------------

Choose a summarizing model, a threshold for compaction, and a budget for recent
messages. The example sets deliberately small limits so two requests trigger
a summary:

.. literalinclude:: ../_examples/compaction.py
   :language: python
   :pyobject: main

The original six messages remain in SQLite; the next request can restore the
summary plus the three messages after its boundary. A summary is lossy: verify
that it preserves the facts your workflow needs.

.. important::

   The word-based token counter is only a deterministic test fixture. Use a
   token estimate appropriate to your model, leave room for the response and
   tool results, and set limits below the provider's context window.

Ask for compaction
--------------------

``CompactionExtension`` also runs on demand: ``AgentClient.compact`` restores the
session, asks every extension to compact through ``on_compact``, and returns the
checkpoint it stored. No input message is appended and no primary model call is
made, so a UI can summarize a conversation the moment a reader asks for it::

   stored = await client.compact()
   print(stored.summary if stored is not None else "nothing to compact")

The events of the pass reach the client's ``event_dispatcher`` exactly as a
streamed run does, which is how the ``Compaction started`` lines above are
produced.

An explicit ``compact()`` pass does not wait for the automatic threshold. It
still needs older complete dialogue to summarize, and skips a replacement that
is not smaller. ``None`` can mean there is no eligible window or no useful
reduction; it does not mean the request failed. ``COMPACTION_COMPLETED`` with
``applied=False`` reports a generated summary that was discarded.

Compaction is a model request and can incur charges or fail. A missing,
empty, or tool-calling summary is a protocol error, not a successful checkpoint.
Summaries are lossy: use tests and review their content for workflows where
exact historical wording is important. Keep the raw log for a full transcript.

Complete source
-------------------

:download:`Download compaction.py <../_examples/compaction.py>`.

.. literalinclude:: ../_examples/compaction.py
   :language: python
   :linenos:

Next: :doc:`../learn/sessions` for durable history, or
:doc:`../concepts/context` for choosing context limits.
