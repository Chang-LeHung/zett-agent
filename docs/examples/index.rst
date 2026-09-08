Runnable examples
=====================

Start with :doc:`../learn/first-agent`, then build your own extension in
:doc:`../extending/first-extension`. Both include complete downloadable programs.

These examples use the actual runtime with deterministic adapters. They teach
framework behavior without requiring credentials, not real-model reasoning quality.
For a real provider, follow :doc:`../learn/providers`.

.. toctree::
   :maxdepth: 1

   streaming-tools
   sessions
   compaction
   approval
   observability
   subagent
   cancellation
   goal
   storage-adapter

Run every offline example with ``make docs-examples``. Tests disable network
connections and use a temporary working directory for each program.
