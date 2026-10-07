How it works
============

You can build a lot with the guides alone. These pages explain the few
contracts that are worth understanding before you push the runtime further —
what one request owns, when context changes, and which channel carries which
message to your UI.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Architecture and ownership
      :link: architecture
      :link-type: doc

      What each object owns, and where your application's responsibility starts.

   .. grid-item-card:: Request lifecycle
      :link: lifecycle
      :link-type: doc

      The exact order of hooks, phases, and cleanup inside one request.

   .. grid-item-card:: Context and memory
      :link: context
      :link-type: doc

      Which messages the model sees, and how compaction changes them.

   .. grid-item-card:: Events and channels
      :link: events
      :link-type: doc

      UI events, internal notifications, and external input are different things.

If you only read one page before writing a stateful extension, read
:doc:`lifecycle`; if you are wiring a UI, read :doc:`events`.

.. toctree::
   :hidden:

   architecture
   lifecycle
   context
   events
