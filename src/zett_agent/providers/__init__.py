"""Provider adapters.

Every adapter is imported from its own module, for example
``from zett_agent.providers.openai import OpenAIProvider``. Each adapter imports
its provider SDK inside the calls that need it, so importing an adapter module
never pays for that SDK.
"""
