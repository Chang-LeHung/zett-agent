Stream a tool round trip
============================

Turn a Python function into a tool and show its progress in your application.
The model asks for ``add``, receives ``42``, and produces a final answer.

Run it
----------

From the repository root:

.. code-block:: console

   uv run python docs/_examples/streaming_tools.py

The output includes:

.. code-block:: text

   Answer: 42

This program uses deterministic offline models and temporary storage where needed.
It requires no API key and is executed by the documentation test suite.

The three pieces to reuse
-------------------------

**Define the capability.** Type annotations describe the arguments, and the
docstring tells the model when to call it:

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: add

**Display progress.** Event callbacks receive reasoning, the tool call, its
result, and incremental answer text:

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: ConsoleEvents

**Connect them to a client.** Replace ``MathModel()`` with your provider when
you are ready to make real requests:

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :pyobject: main

The scripted model always calls the tool. A real model chooses whether to call
it, and may not return visible reasoning even when it reasons internally.

Complete source
-------------------

:download:`Download streaming_tools.py <../_examples/streaming_tools.py>`.

.. literalinclude:: ../_examples/streaming_tools.py
   :language: python
   :linenos:

Next: :doc:`../learn/streaming` for event handling and cancellation, or
:doc:`../learn/tools` for schemas, errors, and safe execution.
