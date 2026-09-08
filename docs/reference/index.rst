API reference
=================

Reference pages describe signatures, field meanings, lifecycle contracts, and
source implementations. For a complete program, start with the
:doc:`examples <../examples/index>`; for extension design, read
:doc:`the extension tutorial <../extending/first-extension>`.

.. toctree::
   :maxdepth: 1

   ../_generated/group-client
   ../_generated/group-agent
   ../_generated/group-messages
   ../_generated/group-events
   ../_generated/group-model
   ../_generated/group-providers
   ../_generated/group-tools
   ../_generated/group-storage
   ../_generated/group-extensions
   ../_generated/group-constants
   ../docstrings

The reference includes all names in ``zett_agent.__all__`` plus the protected
subclass helpers that extension authors need. It intentionally omits unrelated
inherited methods from Python enums and third-party validation frameworks.

Common entry points
-----------------------

* Create a client: :func:`~zett_agent.create_agent`.
* Run or stream: :class:`~zett_agent.AgentClient` and :class:`~zett_agent.Agent`.
* Write an extension: :class:`~zett_agent.AgentExtension` and :class:`~zett_agent.AgentContext`.
* Wait for external input: :class:`~zett_agent.ExternalEventExtension`.
* Add persistence: :class:`~zett_agent.BaseSessionPersistenceExtension`.
* Implement a model: :class:`~zett_agent.AgentModel`.
