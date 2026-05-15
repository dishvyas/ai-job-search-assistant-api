from app.core.config import settings
from app.llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    """
    Return the configured LLM provider based on settings.llm_provider.

    Supported values:
      - "mock"   → MockLLMProvider (default, no API key required)
      - "gemini" → GeminiLLMProvider (requires GEMINI_API_KEY)
    """
    provider = settings.llm_provider.lower()

    if provider == "mock":
        from app.llm.mock import MockLLMProvider

        return MockLLMProvider()

    if provider == "gemini":
        from app.llm.gemini import GeminiLLMProvider

        return GeminiLLMProvider(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
        )

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported values: 'mock', 'gemini'.")
