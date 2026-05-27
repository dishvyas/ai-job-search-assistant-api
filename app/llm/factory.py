# Factory function that maps the LLM_PROVIDER setting to a concrete implementation.
# This is the single place that knows about concrete provider classes — all other
# code depends on the LLMProvider interface and never imports Gemini or Mock directly.
from app.core.config import settings
from app.llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    """
    Return the configured LLM provider based on settings.llm_provider.

    Supported values:
        - "mock"   → MockLLMProvider (default, no API key required)
        - "gemini" → GeminiLLMProvider (requires GEMINI_API_KEY)
        - "openai" → OpenAILLMProvider (requires OPENAI_API_KEY)
    """
    provider = settings.llm_provider.lower()

    if provider == "mock":
        # Lazy import — avoids loading mock.py at module import time, which keeps
        # startup clean and makes the import graph easier to follow.
        from app.llm.mock import MockLLMProvider

        return MockLLMProvider()

    if provider == "gemini":
        # Lazy import — google-genai is an optional dependency; importing it here
        # means tests that don't set LLM_PROVIDER=gemini never need it installed.
        from app.llm.gemini import GeminiLLMProvider

        return GeminiLLMProvider(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
        )

    if provider == "openai":
        # Lazy import — openai is an optional dependency; importing it here
        # means tests that don't set LLM_PROVIDER=openai never need it installed.
        from app.llm.openai import OpenAILLMProvider

        return OpenAILLMProvider(
            api_key=settings.openai_api_key or "",
            model=settings.openai_model,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. Supported values: 'mock', 'gemini', 'openai'."
    )
