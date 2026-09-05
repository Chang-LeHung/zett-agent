class AgentError(Exception):
    """Base exception for runtime and provider errors."""


class AgentProtocolError(AgentError):
    """A model stream did not produce exactly one final response."""


class AgentIterationLimitError(AgentError):
    """The model/tool loop exceeded its configured limit."""
