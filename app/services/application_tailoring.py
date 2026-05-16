from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.exceptions import LLMOutputParsingError, LLMProviderError
from app.llm.factory import get_llm_provider
from app.llm.mock import MockLLMProvider
from app.llm.parsing import parse_tailoring_response
from app.prompts.tailoring import build_tailoring_prompt
from app.repositories.application_runs import create_application_tailoring_run
from app.schemas.application import ApplicationTailorRequest, ApplicationTailorResponse
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
    Non-LLM programming errors are not caught and surface as 500s.
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


def tailor_application(
    request: ApplicationTailorRequest,
    db: Session,
) -> ApplicationTailorResponse:
    """
    Tailor a job application using the configured LLM provider.

    Generates structured output, persists the run to the database,
    and returns the API response. Falls back to mock on provider failure.
    """
    prompt = build_tailoring_prompt(request)
    llm_output, provider_used, used_fallback = _get_llm_output(prompt)

    fallback_note = " [Fallback mode used]" if used_fallback else ""

    response = ApplicationTailorResponse(
        tailored_summary=llm_output.tailored_summary + fallback_note,
        tailored_bullets=llm_output.tailored_bullets,
        cover_letter_draft=llm_output.cover_letter_draft,
        application_question_answers=llm_output.application_question_answers,
        recruiter_message_draft=llm_output.recruiter_message_draft,
        fit_gap_analysis=llm_output.fit_gap_analysis + fallback_note,
        interview_talking_points=llm_output.interview_talking_points,
    )

    create_application_tailoring_run(
        db=db,
        request=request,
        response=response,
        provider_used=provider_used,
        fallback_used=used_fallback,
    )

    return response
