Examples
========

Small, complete programs you can run, read, and copy. Each one uses the real
runtime with a deterministic adapter, so it produces the same output without
credentials — and teaches framework behavior rather than model quality.

Start with :doc:`../learn/first-agent`, then build your own extension in
:doc:`../extending/first-extension`. For a real provider, follow
:doc:`../learn/providers`.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Streaming tools
      :link: streaming-tools
      :link-type: doc

      Watch a tool call, its result, and the final answer in one stream.

   .. grid-item-card:: Sessions
      :link: sessions
      :link-type: doc

      Restore a conversation from SQLite in a new process.

   .. grid-item-card:: On-demand skills
      :link: ../learn/on-demand
      :link-type: doc

      Read one skill's instructions through a real tool round trip.

   .. grid-item-card:: Compaction
      :link: compaction
      :link-type: doc

      Replace older turns with a checkpoint when context grows.

   .. grid-item-card:: Approval
      :link: approval
      :link-type: doc

      Pause a privileged tool until the UI answers.

   .. grid-item-card:: Observability
      :link: observability
      :link-type: doc

      Record model usage, cache counts, and message timing for diagnostics.

   .. grid-item-card:: Subagent
      :link: subagent
      :link-type: doc

      Delegate one task to a child profile with its own tools.

   .. grid-item-card:: Cancellation
      :link: cancellation
      :link-type: doc

      Stop mid-stream and keep the runtime reusable.

   .. grid-item-card:: Goal mode
      :link: goal
      :link-type: doc

      Keep working until an evaluator agrees the goal is met.

   .. grid-item-card:: Storage adapter
      :link: storage-adapter
      :link-type: doc

      Back sessions with your own storage implementation.

Run every offline example with ``make docs-examples``. Tests disable network
connections and use a temporary working directory for each program.

.. toctree::
   :hidden:

   streaming-tools
   sessions
   compaction
   approval
   observability
   subagent
   cancellation
   goal
   storage-adapter
