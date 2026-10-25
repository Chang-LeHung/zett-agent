Implement a storage adapter
===============================

The runtime needs three operations from storage: restore what the model should see,
append one immutable message, and save a compaction checkpoint. Implementing those
turns any database, object store, or file format into session storage.

Reach for this page when the shipped SQL backends do not fit. For SQLite, MySQL, or
PostgreSQL, :doc:`../learn/storage` already covers them.

Keep lifecycle integration in the base extension
----------------------------------------------------

Do not duplicate context restoration, Raw Log events, or compaction bookkeeping
inside a new database extension. Implement the SessionStorage contract and pass
it to BaseSessionPersistenceExtension. The base already knows when to load, append,
snapshot, and clean up request bookkeeping.

The three runtime storage operations
----------------------------------------

.. list-table:: SessionStorage boundary
   :header-rows: 1
   :widths: 18 42 40

   * - Method
     - Returns
     - Invariants
   * - load
     - SessionView: optional checkpoint and ordered raw tail
     - A consistent view; unknown sessions are empty.
   * - append
     - Assigned per-session sequence number
     - Original messages are immutable; appends advance sequence atomically.
   * - snapshot
     - ContextSnapshot with a new version
     - Check expected_version and advancing Raw Log boundary; never rewrite raw messages.

CRUD for session titles and paginated UI history is provided by concrete storage
implementations such as :class:`~zett_agent.storage.SQLSessionStorage`. It is separate
from the three operations required by the runtime persistence extension.

Step 1: decide what one stored message holds
------------------------------------------------

A replay needs the original message, its sequence, and the request it belonged to.
Timing, metadata, tags, and token counts are for display and analysis; store them if
your UI shows them. Encode the message with the runtime codec rather than writing your
own serializer, so replay blocks, tool calls, and images survive:

.. code-block:: python

   from zett_agent.storage import decode_messages, encode_messages

   text = encode_messages([message])  # a JSON string
   restored = decode_messages(text)[0]  # the original message object

Step 2: implement load
--------------------------

Return the latest checkpoint and every message *after* its boundary. An unknown
session returns an empty view, not an error: the first append creates it. Read both
parts from one consistent snapshot, or two requests can observe different versions.

.. literalinclude:: ../_examples/custom_storage.py
   :language: python
   :pyobject: JsonFileStorage.load

Step 3: implement append
----------------------------

Allocate the next per-session sequence and write the message in one transaction. The
returned number is what a later snapshot boundary refers to. Messages are immutable:
an edit is a new append, never a rewrite. Session attributes arrive on the first
append; keep the parent ID immutable.

.. literalinclude:: ../_examples/custom_storage.py
   :language: python
   :pyobject: JsonFileStorage.append

Step 4: implement snapshot
------------------------------

A checkpoint replaces the Raw Log prefix through an inclusive sequence. Reject a
stale ``expected_version`` instead of overwriting it, reject a boundary past the end
of the log, and advance the version by one. Never delete the messages it summarizes;
a UI still displays them.

.. literalinclude:: ../_examples/custom_storage.py
   :language: python
   :pyobject: JsonFileStorage.snapshot

Step 5: hand it to the base extension
------------------------------------------

The extension stays a constructor. Every lifecycle decision stays in the base class.

.. literalinclude:: ../_examples/custom_storage.py
   :language: python
   :pyobject: JsonPersistence

The complete program, including the offline model and the reopen-after-write check,
is :doc:`../examples/custom-storage`. One request against it produces three appends
(system, user, assistant) and a second instance reads them back from the file.

Snapshot consistency checklist
----------------------------------

* Read the checkpoint and its tail consistently; do not combine different database versions.
* Treat compacted_through_sequence as an inclusive represented boundary.
* Start the replay tail strictly after that boundary.
* Reject stale expected_version rather than silently overwriting a checkpoint.
* Do not store a checkpoint after every successful response.
* Preserve message roles, attributes, replay blocks, timing, metadata, tags, and usage.
* Do not treat request_id as an automatic idempotency key.

Testing a storage implementation
------------------------------------

Write the tests against the contract, not the database driver:

* An unknown session loads as empty, and the first append creates it.
* Two appends in one request keep their order, and a later load returns them in sequence.
* A tool round trip stores and restores replay blocks and tool results unchanged.
* A snapshot with a stale version is rejected, and the stored one is untouched.
* A snapshot boundary past the end of the Log is rejected.
* A compacted session loads the checkpoint plus the tail after its boundary, not the
  messages it summarized.
* A second storage instance over the same file or database sees the same conversation.
* Deleting a session removes its messages and checkpoints, and leaves other sessions alone.

Use a temporary directory or a dedicated test schema, and release every connection
before the test ends.

Supporting checkpoint type
------------------------------

The snapshot contract uses this class from its defining module:

.. code-block:: python

   from zett_agent.extensions.compaction import CompactedMessage

   checkpoint = CompactedMessage(content="The user prefers Python.")

CompactedMessage is a specialized system message marking a context checkpoint.
Its ``content`` contains the summary; inherited ``attributes`` holds optional
message annotations. It is not exported from the package root. Store this
checkpoint separately from the original Raw Log, together with the inclusive
sequence boundary described above.

Related API: :class:`~zett_agent.extensions.persistence.SessionStorage`,
:class:`~zett_agent.extensions.persistence.BaseSessionPersistenceExtension`,
:class:`~zett_agent.extensions.persistence.SessionView`,
:class:`~zett_agent.storage.SQLSessionStorage`,
:class:`~zett_agent.extensions.sql.SQLSessionExtension`.
