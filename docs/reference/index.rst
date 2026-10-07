API reference
=============

Everything the package exposes, grouped by what you are trying to do. Each page
documents the signature, the meaning of every field, and the behavior you can
rely on. Source links are included when you want to see the implementation.

If you are new here, start with :doc:`../learn/index` or
:doc:`../examples/index`; come back when you need an exact name or default.

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

Common entry points
-------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Create a client
      :link: ../_generated/create_agent
      :link-type: doc

      :func:`~zett_agent.client.create_agent` builds an
      :class:`~zett_agent.client.AgentClient` over the runtime.

   .. grid-item-card:: Run or stream
      :link: ../_generated/AgentClient
      :link-type: doc

      :class:`~zett_agent.client.AgentClient` for applications,
      :class:`~zett_agent.agent.Agent` for the loop itself.

   .. grid-item-card:: Write an extension
      :link: ../_generated/AgentExtension
      :link-type: doc

      :class:`~zett_agent.extensions.base.AgentExtension` and
      :class:`~zett_agent.agent.AgentRunContext`.

   .. grid-item-card:: Implement a model
      :link: ../_generated/AgentModel
      :link-type: doc

      :class:`~zett_agent.model.AgentModel` is the protocol every adapter follows.

.. note::

   Packages re-export nothing. Import each name from the module that defines it::

      from zett_agent.client import create_agent
      from zett_agent.tools.base import tool
      from zett_agent.providers.openai import OpenAIProvider
      from zett_agent.extensions.coding import CodingExtension

The reference documents every public object the runtime defines, plus the
protected subclass helpers extension authors need. It intentionally omits
unrelated inherited methods from Python enums and third-party validation
frameworks.
