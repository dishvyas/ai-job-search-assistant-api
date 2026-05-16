from app.core.config import settings
from app.llm.exceptions import LLMOutputParsingError, LLMProviderError
from app.llm.factory import get_llm_provider
from app.llm.mock import MockLLMProvider
from app.llm.parsing import parse_tailoring_response
from app.schemas.llm_output import TailoringLLMOutput


def _get_llm_output(prompt: str) -> tuple[TailoringLLMOutput, str, bool]:
    """
    Call the configured provider, parse the JSON response, and return
    (parsed_output, provider_used, used_fallback).

    provider_used values:
      - "mock"          — mock provider succeeded
      - "gemini"        — Gemini provider succeeded
      - "fallback-mock" — configured provider failed; fell back to mock

    Fallback is triggered by LLMProviderError or LLMOutputParsingError.
    Non-LLM programming errors are not caught and surface as failures.
    """
    configured_provider = settings.llm_provider.lower()

    try:
        provider = get_llm_provider()
        raw = provider.generate_text(prompt)
        return parse_tailoring_response(raw), configured_provider, False
    except LLMProviderError:
        pass  # provider unavailable — fall through to mock
    except LLMOutputParsingError:
        pass  # provider returned malformed output — fall through to mock

    fallback = MockLLMProvider()
    raw = fallback.generate_text(prompt)
    return parse_tailoring_response(raw), "fallback-mock", True
