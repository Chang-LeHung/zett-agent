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

The reference documents every public object the runtime defines, grouped by
responsibility, plus the protected subclass helpers that extension authors
need. It intentionally omits unrelated inherited methods from Python enums and
third-party validation frameworks.

Nothing is re-exported by a package ``__init__``. Import each name from the
module that defines it, for example ``from zett_agent.agent import Agent``,
``from zett_agent.tools.base import tool``,
``from zett_agent.providers.openai import OpenAIProvider``, or
``from zett_agent.extensions.coding import CodingExtension``.

Common entry points
-----------------------

* Create a client: :func:`~zett_agent.client.create_agent`.
* Run or stream: :class:`~zett_agent.client.AgentClient` and :class:`~zett_agent.agent.Agent`.
* Write an extension: :class:`~zett_agent.extensions.base.AgentExtension` and :class:`~zett_agent.agent.AgentRunContext`.
* Wait for external input: :class:`~zett_agent.extensions.external.ExternalEventExtension`.
* Add persistence: :class:`~zett_agent.extensions.persistence.BaseSessionPersistenceExtension`.
* Implement a model: :class:`~zett_agent.model.AgentModel`.
