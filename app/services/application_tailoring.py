# Single-step tailoring service — one LLM call that produces the full output.
# Separated from the route handler so the fallback logic can be unit-tested
# without an HTTP client. The leading underscore signals this is not public API;
# background_tailoring.py is the only caller.
from app.core.config import settings
from app.llm.exceptions import LLMOutputParsingError, LLMProviderError
from app.llm.factory import get_llm_provider
from app.llm.fallback_reason import sanitize_fallback_reason
from app.llm.mock import MockLLMProvider
from app.llm.parsing import parse_tailoring_response
from app.schemas.llm_output import TailoringLLMOutput


def _get_llm_output(prompt: str) -> tuple[TailoringLLMOutput, str, bool, str | None]:
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
        return parse_tailoring_response(raw), configured_provider, False, None
    except LLMProviderError as exc:
        fallback_reason = sanitize_fallback_reason(exc)
    except LLMOutputParsingError as exc:
        fallback_reason = sanitize_fallback_reason(exc)
    # Two separate except clauses (not a tuple) so each failure mode is
    # individually legible; both have the same recovery action.

    fallback = MockLLMProvider()
    raw = fallback.generate_text(prompt)
    # If mock also fails here it is a code bug, not a runtime condition — let it raise.
    return parse_tailoring_response(raw), "fallback-mock", True, fallback_reason
