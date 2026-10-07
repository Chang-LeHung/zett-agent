What the model sees
===================

"Context" is the list of messages sent with a request. Understanding it explains
most surprising behavior: why a turn was forgotten, why compaction changed the
prompt, and why a UI transcript can be longer than what the model received.

Message roles
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Role
     - Meaning
   * - system
     - Application instructions, rebuilt for every request.
   * - user
     - User text, or ordered text and images.
   * - assistant
     - The model's answer, its reasoning, and any tool calls it requested.
   * - tool
     - A result linked to the assistant call by ``tool_call_id``.
   * - agent
     - An internal continuation; providers see it as user input.

System instructions are rebuilt each time
-----------------------------------------

History extensions restore dialogue, not prompts. Instructions are added again
for every request, so changing your system prompt or tools takes effect on the
next turn without stale instructions piling up.

That is also why ``extensions=[]`` loses the conversation: no extension restores
the messages. See :doc:`../learn/sessions`.

Transcript versus model input
-----------------------------

After compaction the two diverge, and both are useful:

.. code-block:: text

   raw log      1  2  3  4  5  6  7  8      what the user saw
                └──── checkpoint ────┘
   model input  [checkpoint]  6  7  8        what the model receives

* Show the raw log in a UI: ``await storage.list_raw_messages(session_id)``.
* Send the model input: the latest checkpoint plus the raw messages after it.

A checkpoint records the last original message it covers, so restoring starts
after that point and summarized messages are never sent twice.

Keep model context and display history separate
------------------------------------------------------------

Choose what you persist and what you send back to the model independently.
``Message.persist=False`` asks persistence subscribers not to store a message;
it does not hide it from the current request. ``include_in_messages=False``
keeps a message out of future restored context, not necessarily out of the
raw transcript. Use message attributes for application data instead of
putting private routing information in prompt text.

Do not rebuild provider context by concatenating visible chat bubbles. Tool
calls and matching results are part of the conversation even when a UI hides
them. A restored context must preserve their IDs and structure; a persistence
extension handles that projection for you. See :doc:`../learn/sessions`.

When context runs out
---------------------

:class:`~zett_agent.extensions.compaction.CompactionExtension` watches the size
of the request and, when it crosses a threshold, replaces older turns with one
summary.

* Recent turns are kept whole, including their tool calls and results.
* The current turn is never split, so one oversized turn is left alone.
* If the summary would not be smaller than what it replaces, nothing changes.

Configure it with ``max_tokens`` and ``keep_recent_tokens`` and watch the
``COMPACTION_*`` events. See :doc:`../examples/compaction`.
