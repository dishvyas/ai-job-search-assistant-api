from pydantic import BaseModel


class TailoringLLMOutput(BaseModel):
    """
    Internal schema for the structured JSON the LLM is asked to produce.
    This is not exposed in the API response — it is an intermediate parsing target.
    """

    tailored_summary: str
    tailored_bullets: list[str]
    cover_letter_draft: str
    application_question_answers: list[str]
    recruiter_message_draft: str
    fit_gap_analysis: str
    interview_talking_points: list[str]
