Troubleshooting
===================

No navigation on the page
-----------------------------

Run ``make docs-serve`` from the current checkout and open the printed URL.
The left sidebar is global, including on the homepage; on a narrow screen, open
it with the navigation toggle. The right sidebar is only the current page's
outline. If an older server is running, stop it before rebuilding. Browser search
and the sidebar Search documentation link both open the generated search index.

The second turn forgot the first
------------------------------------

Check ``extensions``. Passing an explicit list removes the default memory
extension. Include InMemoryMessageAccumulator or a persistence extension, and
reuse the same session ID. Different clients with different accumulator instances
do not share process memory simply because their IDs match.

The model never sees my injected message
--------------------------------------------

Use ``on_state`` for restoring context, ``on_message`` for replacing the current
input, and ``before_model`` for per-step changes. Do not write to ``agent.state``
from a shared extension: use the supplied ``context.state``. Other earlier/later
hooks may replace messages; inspect priorities and the :doc:`lifecycle <../concepts/lifecycle>`.

My custom event never reaches the UI
----------------------------------------

``context.publish`` broadcasts ExtensionEvent internally. For UI output, call
``await context.emit(AgentEvent(...))`` from a lifecycle hook. The hook still
returns None; ``Agent.stream`` alone yields the queued event. See
:doc:`../extending/events`.

Approval remains pending
----------------------------

Match the outbound event's session ID and tool-call ID; provide the expected
external event name and payload. An empty accepting-extension list means no
pending receiver matched. Do not block the asyncio loop with ``input()``; use an
async UI or ``asyncio.to_thread`` for terminal input. Cancellation invalidates
pending routes, so a late response will not resume a closed request.

Reasoning or cache statistics are missing
---------------------------------------------

The model must actually return them. Reasoning effort is a request preference,
not a promise of visible reasoning. Usage belongs to MODEL_COMPLETED, not every
text fragment. Missing counters default to zero; cache_hit_rate is None without
input tokens. It is a ratio, so multiply by 100 for a percentage.

A retry duplicated text
---------------------------

The built-in providers do not retry after an event has escaped. Check application
retry code and double-dispatching. Do not append final message.content after
already appending all TEXT_DELTA fragments to the same answer buffer.

Compaction did not run
--------------------------

It requires a crossed threshold and a completed older turn that can be replaced.
The current turn is retained whole; a single oversized turn cannot be split.
If the generated summary is not smaller, COMPACTION_COMPLETED has applied=False.
Use :doc:`../examples/compaction` for a reproducible test.
