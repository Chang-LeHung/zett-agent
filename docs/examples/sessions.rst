Restore SQLite history
==========================

Reopen a real temporary database, restore history, paginate messages, update the title, and delete the session.

Run it
----------

From the repository root:

.. code-block:: console

   uv run --directory backend/zett-agent python docs/_examples/sessions.py

The output includes:

.. code-block:: text

   Raw roles: user, assistant, user, assistant

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download sessions.py <../_examples/sessions.py>`.

.. literalinclude:: ../_examples/sessions.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
