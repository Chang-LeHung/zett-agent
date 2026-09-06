"""Generic session storage exposed as a ready-to-use Agent extension."""

from .persistence import BaseSessionPersistenceExtension, SessionStorage


class SessionPersistenceExtension(BaseSessionPersistenceExtension[SessionStorage]):
    """Adapt any application-owned SessionStorage to Agent lifecycle hooks.

    Subclass BaseSessionPersistenceExtension only when an extension must
    construct or own its storage resources.
    """
