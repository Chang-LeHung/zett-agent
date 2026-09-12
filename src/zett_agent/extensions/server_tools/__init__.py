"""Provider-specific extensions for tools executed by model servers."""

from .anthropic import AnthropicServerToolExtension
from .base import ServerToolExtension
from .deepseek import DeepSeekServerToolExtension
from .google import GoogleServerToolExtension
from .openai import OpenAIServerToolExtension
from .openrouter import OpenRouterServerToolExtension

__all__ = [
    "AnthropicServerToolExtension",
    "DeepSeekServerToolExtension",
    "GoogleServerToolExtension",
    "OpenAIServerToolExtension",
    "OpenRouterServerToolExtension",
    "ServerToolExtension",
]
