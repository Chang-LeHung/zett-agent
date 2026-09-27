Messages, context, and persistence
======================================

Message roles
-----------------

.. list-table:: Provider-neutral messages
   :header-rows: 1

   * - Role / class
     - Meaning
   * - system / SystemMessage
     - Application instructions rebuilt for the current request.
   * - user / UserMessage
     - User text or ordered text/image content.
   * - assistant / AssistantMessage
     - Model answer, optional reasoning, complete tool calls, and replay data.
   * - tool / ToolMessage
     - Serialized result linked to an assistant call by tool_call_id.
   * - agent / AgentMessage
     - Internal continuation instruction; mapped to user at provider boundaries.

``Message.attributes`` holds per-message application information. Provider adapters
do not automatically send arbitrary attributes as model content. Request-level
``metadata`` and ``tags`` live on context and are captured by persistence when
records are appended; they do not rewrite every Message object.

Restored context is not a new message
-----------------------------------------

Use ``on_state`` to restore existing messages directly into state. Use
``on_message`` to replace the pending current input. The Agent then calls
``context.append_message`` to publish the new raw record once.

Calling append_message while restoring history would create duplicate raw events.
Appending directly for newly produced messages bypasses persistence notifications.
These are different operations, not interchangeable convenience methods.

Snapshot boundaries
-----------------------

.. code-block:: text

   Raw Log:  1  2  3  4  5  6  7  8
             \___________/  \______/
                snapshot      raw tail
                through=5      6..8

   Model context = checkpoint message + raw messages 6, 7, 8
   Conversation UI = original messages 1 through 8

Snapshots are immutable checkpoints created on compaction, not copies of every
successful request. ``compacted_through_sequence`` is the last original record
already represented by the checkpoint. Restoring starts strictly after it, so
summarized messages are not sent twice.

Use :class:`~zett_agent.extensions.persistence.SessionView` for the restored model context and
``await storage.list_raw_messages(...)`` for full UI history. Request IDs
correlate records but do not automatically make appends idempotent. Storage
restoration is asynchronous: the runtime awaits the snapshot and raw tail before
the first model call of a request.

Compaction retains coherent turns
-------------------------------------

Recent-token limits are extended to whole turns, including tool calls and results.
The current turn cannot be split away. A threshold crossing does not guarantee a
summary: there must be an older completed region, and the summary must be smaller.
If not, the extension preserves the existing messages.

Signed model replay data
----------------------------

Some providers require opaque signed thinking/tool parts on later round trips.
AssistantMessage stores these as provider/model-scoped ``replay_blocks``. They are
not the displayable ``reasoning`` field. Persist them unchanged and do not reuse
signed blocks with a different provider/model identity. See provider reference
pages for their mapping implementations.
