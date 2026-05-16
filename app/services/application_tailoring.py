from app.llm.exceptions import LLMOutputParsingError, LLMProviderError
from app.llm.factory import get_llm_provider
from app.llm.mock import MockLLMProvider
from app.llm.parsing import parse_tailoring_response
from app.prompts.tailoring import build_tailoring_prompt
from app.schemas.application import ApplicationTailorRequest, ApplicationTailorResponse
from app.schemas.llm_output import TailoringLLMOutput


def _get_llm_output(prompt: str) -> tuple[TailoringLLMOutput, bool]:
    """
    Call the configured provider, parse the JSON response, and return
    (parsed_output, used_fallback).

    Fallback to MockLLMProvider when:
    - the provider raises LLMProviderError (e.g. 503, 429)
    - the provider output cannot be parsed (LLMOutputParsingError)

    Non-LLM programming errors are not caught and will surface as 500s.
    """
    # --- attempt configured provider ---
    try:
        provider = get_llm_provider()
        raw = provider.generate_text(prompt)
        return parse_tailoring_response(raw), False
    except LLMProviderError:
        pass  # provider unavailable — fall through to mock
    except LLMOutputParsingError:
        pass  # provider returned malformed output — fall through to mock

    # --- fallback: mock provider ---
    fallback = MockLLMProvider()
    raw = fallback.generate_text(prompt)
    # Mock always returns valid JSON; if this somehow fails, let it raise.
    return parse_tailoring_response(raw), True


def tailor_application(request: ApplicationTailorRequest) -> ApplicationTailorResponse:
    """
    Tailor a job application using the configured LLM provider.

    The LLM is asked to return structured JSON. That JSON is parsed and
    validated before being mapped into the API response.
    On provider or parsing failure, falls back to the mock provider.
    """
    prompt = build_tailoring_prompt(request)
    llm_output, used_fallback = _get_llm_output(prompt)

    fallback_note = " [Fallback mode used]" if used_fallback else ""

    return ApplicationTailorResponse(
        tailored_summary=llm_output.tailored_summary + fallback_note,
        tailored_bullets=llm_output.tailored_bullets,
        cover_letter_draft=llm_output.cover_letter_draft,
        application_question_answers=llm_output.application_question_answers,
        recruiter_message_draft=llm_output.recruiter_message_draft,
        fit_gap_analysis=llm_output.fit_gap_analysis + fallback_note,
        interview_talking_points=llm_output.interview_talking_points,
    )
