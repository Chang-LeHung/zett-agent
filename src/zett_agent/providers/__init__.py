"""Provider adapters.

Each adapter imports straight from the package, for example
``from zett_agent.providers import OpenAIProvider``. A name loads its defining module the first time
it is used, and each adapter imports its provider SDK inside the calls that need it, so importing
this package, or an adapter, never pays for an SDK. The defining modules, such as
``zett_agent.providers.openai``, stay importable for everything the package does not list.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .._lazy import lazy_exports as _lazy_exports

_MODULES = {
    "anthropic": ("AnthropicProvider",),
    "anthropic_compatible": ("AnthropicCompatibleProvider",),
    "base": (
        "ProviderAuthError",
        "ProviderError",
        "ProviderHTTPError",
        "ProviderRefusalError",
        "ProviderResponseError",
        "ProviderTimeoutError",
    ),
    "cerebras": ("CerebrasProvider",),
    "deepseek": ("DeepSeekAnthropicProvider", "DeepSeekProvider"),
    "glm": ("GLMAnthropicProvider", "GLMProvider"),
    "google": ("GoogleProvider",),
    "groq": ("GroqProvider",),
    "hosts": ("DeepInfraProvider", "FireworksProvider", "SiliconFlowProvider", "TogetherProvider"),
    "kimi": ("KimiAnthropicProvider", "KimiProvider"),
    "minimax": ("MiniMaxAnthropicProvider", "MiniMaxProvider"),
    "mistral": ("MistralProvider",),
    "ollama": ("OllamaProvider",),
    "openai": ("OpenAIProvider",),
    "openai_compatible": ("OpenAICompatibleProvider",),
    "openrouter": ("OpenRouterProvider",),
    "xai": ("XAIProvider",),
}
_EXPORTS = {name: f"{__name__}.{module}" for module, names in _MODULES.items() for name in names}

__all__ = tuple(_EXPORTS)
__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS)

if _TYPE_CHECKING:
    from .anthropic import AnthropicProvider as AnthropicProvider
    from .anthropic_compatible import AnthropicCompatibleProvider as AnthropicCompatibleProvider
    from .base import ProviderAuthError as ProviderAuthError
    from .base import ProviderError as ProviderError
    from .base import ProviderHTTPError as ProviderHTTPError
    from .base import ProviderRefusalError as ProviderRefusalError
    from .base import ProviderResponseError as ProviderResponseError
    from .base import ProviderTimeoutError as ProviderTimeoutError
    from .cerebras import CerebrasProvider as CerebrasProvider
    from .deepseek import DeepSeekAnthropicProvider as DeepSeekAnthropicProvider
    from .deepseek import DeepSeekProvider as DeepSeekProvider
    from .glm import GLMAnthropicProvider as GLMAnthropicProvider
    from .glm import GLMProvider as GLMProvider
    from .google import GoogleProvider as GoogleProvider
    from .groq import GroqProvider as GroqProvider
    from .hosts import DeepInfraProvider as DeepInfraProvider
    from .hosts import FireworksProvider as FireworksProvider
    from .hosts import SiliconFlowProvider as SiliconFlowProvider
    from .hosts import TogetherProvider as TogetherProvider
    from .kimi import KimiAnthropicProvider as KimiAnthropicProvider
    from .kimi import KimiProvider as KimiProvider
    from .minimax import MiniMaxAnthropicProvider as MiniMaxAnthropicProvider
    from .minimax import MiniMaxProvider as MiniMaxProvider
    from .mistral import MistralProvider as MistralProvider
    from .ollama import OllamaProvider as OllamaProvider
    from .openai import OpenAIProvider as OpenAIProvider
    from .openai_compatible import OpenAICompatibleProvider as OpenAICompatibleProvider
    from .openrouter import OpenRouterProvider as OpenRouterProvider
    from .xai import XAIProvider as XAIProvider
