"""The Zett Agent runtime.

The root package holds only the version, so importing it never imports a subsystem, a provider
SDK, or a storage engine. Import a name from the module that defines it, for example
``from zett_agent.agent import Agent``. The ``tools``, ``providers``, ``extensions``,
``extensions.server_tools``, and ``sandbox`` packages also re-export their main names lazily, for
example ``from zett_agent.providers import OpenAIProvider`` and
``from zett_agent.extensions import CodingExtension``.
"""

__version__ = "0.2.4"
