"""Extensions that declare tools executed by the model server.

Import the extension matching the active protocol, for example
``from zett_agent.extensions.server_tools import OpenAIServerToolExtension``. A name loads its
defining module the first time it is used, so importing this package imports none of them.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from ..._lazy import lazy_exports as _lazy_exports

_MODULES = {
    "anthropic": ("AnthropicServerToolExtension",),
    "base": ("ServerToolExtension",),
    "deepseek": ("DeepSeekMessagesServerToolExtension", "DeepSeekServerToolExtension"),
    "google": ("GoogleServerToolExtension",),
    "openai": ("OpenAIServerToolExtension",),
    "openrouter": ("OpenRouterServerToolExtension",),
    "xai": ("XAIServerToolExtension",),
}
_EXPORTS = {name: f"{__name__}.{module}" for module, names in _MODULES.items() for name in names}

__all__ = tuple(_EXPORTS)
__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS)

if _TYPE_CHECKING:
    from .anthropic import AnthropicServerToolExtension as AnthropicServerToolExtension
    from .base import ServerToolExtension as ServerToolExtension
    from .deepseek import DeepSeekMessagesServerToolExtension as DeepSeekMessagesServerToolExtension
    from .deepseek import DeepSeekServerToolExtension as DeepSeekServerToolExtension
    from .google import GoogleServerToolExtension as GoogleServerToolExtension
    from .openai import OpenAIServerToolExtension as OpenAIServerToolExtension
    from .openrouter import OpenRouterServerToolExtension as OpenRouterServerToolExtension
    from .xai import XAIServerToolExtension as XAIServerToolExtension
