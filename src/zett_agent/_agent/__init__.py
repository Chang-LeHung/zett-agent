"""Private building blocks of :class:`zett_agent.agent.Agent`.

``Agent`` is the one public class. Its machinery is split by responsibility into mixins, one module
each, so that a change to how tools run does not sit in the same file as how a request is released.
The mixins share state through ``self`` and are only meant to be composed by ``Agent``; nothing here
is part of the public contract, and this package re-exports nothing.
"""
