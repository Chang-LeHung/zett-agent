Write your first extension
==============================

**Goal:** implement ``/note extensions`` as a request-scoped capability. The model
will receive a rewritten instruction, call a note tool, and emit progress events
that a UI can display. **Prerequisites:** :doc:`../learn/tools` and
:doc:`../concepts/lifecycle`. No model API credentials are required.

When an extension is the right abstraction
----------------------------------------------

Use one for dynamic tools, context preparation, custom input commands, approvals,
or persistence hooks. Do not put a text-rendering callback here unless it must
participate in lifecycle decisions; AgentEventDispatcher is simpler for display.

Step 1: register a tool in on_tool
--------------------------------------

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: NoteExtension.on_tool

``context.register_tool`` rejects duplicate names and copies mutable schema
metadata for this request. Registering through context avoids leaking tools into
other sessions. The registered handler can close over context when it needs
request-local information; do not store that handler on a global singleton.

All on_tool hooks finish before any on_state hook, so prompt extensions can see
the complete tool registry even when they were registered earlier in the list.

Step 2: inject instructions and transform only the new input
----------------------------------------------------------------

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: NoteExtension.on_state

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: NoteExtension.on_message

``on_state`` is where an extension restores history or adds system instructions.
``on_message`` sees ``context.input_message`` after restoration but before append.
The example keeps the original input in attributes, replaces the command with an
instruction the model understands, and creates a request-local counter.

Do **not** append the user message here. The runtime appends the final transformed
input exactly once after every on_message hook finishes. This matters for both
in-memory history and append-only persistence.

Step 3: emit a UI progress event
------------------------------------

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: NoteExtension.before_model_events

The hook is an async generator, not an async function returning an event. It runs
before each model call, so a tool round trip produces two progress events. Use a
namespaced custom name such as ``note.model_step`` and a small payload suitable
for your UI transport. Yielding CUSTOM does not change the Agent phase.

Step 4: clean up every terminal path
----------------------------------------

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: NoteExtension.on_success

The full class below additionally handles ``on_error`` and ``RunCancelledEvent``.
An on_success-only cleanup leaks state when a user stops generation. Contexts are
request identities, so two sessions can safely coexist in the same dictionary.
Read :doc:`state` for ownership and idempotent cleanup rules.

Step 5: compose with history and prompt guidance
----------------------------------------------------

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :pyobject: main

NoteExtension has priority 50, ahead of default-priority 100 extensions. Hooks
still execute stage by stage, not extension by extension. Explicit extension
lists replace defaults, which is why memory and tool guidance appear in this list.

Run and inspect the result
------------------------------

.. code-block:: bash

   uv run --directory backend/zett-agent python docs/_examples/custom_extension.py

Expected output::

    Model step: 1
    Model step: 2
    Extensions customize lifecycle hooks.
    Request state cleared

The scripted NoteModel checks the transformed input, preserved attributes, and
injected instructions before requesting the real registered tool. The final
assertion verifies request-local state was removed. To use an LLM, replace only
NoteModel with a provider adapter; the extension and UI callbacks stay the same.

Complete implementation
---------------------------

:download:`Download custom_extension.py <../_examples/custom_extension.py>`.

.. literalinclude:: ../_examples/custom_extension.py
   :language: python
   :linenos:

Next: :doc:`hooks`, :doc:`events`, :doc:`state`.

Related API: :class:`~zett_agent.AgentExtension`, :class:`~zett_agent.AgentContext`,
:meth:`~zett_agent.AgentContext.register_tool`,
:meth:`~zett_agent.AgentContext.register_server_tool`, and
:class:`~zett_agent.AgentEvent`.
