Install and run from source
===============================

Prerequisites
-----------------

The standalone package requires Python 3.12 or newer. This repository develops
and tests with Python 3.14 and uses ``uv`` for dependency management. Node.js is
not required for the Agent library or its documentation.

From the repository root::

    uv sync
    uv run python -c "import zett_agent; print(zett_agent.__file__)"

The printed path should point into this checkout's ``src/zett_agent`` directory.
That confirms examples will use the source you are reading.

Run without an API key
--------------------------

.. code-block:: bash

   uv run python docs/_examples/first_agent.py

Expected output::

    Turn 1: Hello
    Turn 2: Remember this conversation

This example uses a deterministic model adapter, but exercises the real runtime,
stream collection, typed messages, and in-memory conversation history. It makes
no network requests and creates no user database.

Build this documentation
----------------------------

.. code-block:: bash

   make docs-serve
   # Open http://127.0.0.1:8000

Use ``DOCS_PORT=8080`` to change the port. The command builds before serving;
after edits, stop it with Ctrl+C and rerun it. ``make docs-check`` builds a fresh
copy, checks navigation and links, and runs the offline examples.

``make docs-examples`` runs just the downloadable offline programs.
``make docs-ui-check`` installs a test Chromium browser and verifies real desktop
and mobile navigation and search. This browser is isolated from your own profile.

Resource ownership
----------------------

The caller owns provider clients and explicitly created persistence extensions.
Await ``model.aclose()`` and ``persistence.close()`` when finished: storage I/O
is asynchronous, so both are coroutines. Closing a stream is different: it stops
one request, not the entire provider or database.

Next: :doc:`first-agent`.
