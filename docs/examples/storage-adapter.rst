Connect a storage adapter
=============================

Delegate transactions to SQLite while auditing the three persistence operations. New databases must enforce their own sequence and version guarantees.

Run it
----------

From the repository root:

.. code-block:: console

   uv run --directory backend/zett-agent python docs/_examples/storage_adapter.py

The output includes:

.. code-block:: text

   load, append:user, append:assistant

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

Complete source
-------------------

:download:`Download storage_adapter.py <../_examples/storage_adapter.py>`.

.. literalinclude:: ../_examples/storage_adapter.py
   :language: python
   :linenos:

See :doc:`../extending/hooks` for lifecycle ordering and
:doc:`../reference/index` for the complete API reference.
