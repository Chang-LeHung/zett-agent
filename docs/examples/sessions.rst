Restore SQLite history
==========================

Reopen a real temporary database, restore history, paginate messages, update the title, and delete the session.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/sessions.py

The output includes:

.. code-block:: text

   Raw roles: system, user, assistant, system, user, assistant

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

What to take into your application
------------------------------------------

Keep one stable ``session_id`` for a conversation and open a persistence
extension over the same database when the process restarts. The example closes
the first connection before creating a second client, so the restored history
comes from SQLite, not an in-memory cache.

.. literalinclude:: ../_examples/sessions.py
   :language: python
   :pyobject: main

The example also paginates messages, renames the conversation, and deletes it.
It only deletes a session in a temporary database. In your application,
session deletion removes history; put it behind an explicit user action.

Complete source
-------------------

:download:`Download sessions.py <../_examples/sessions.py>`.

.. literalinclude:: ../_examples/sessions.py
   :language: python
   :linenos:

Next: :doc:`../learn/sessions` for storage choices and resource ownership, or
:doc:`compaction` for long conversations.
