from datetime import datetime

from pydantic import BaseModel, field_validator


class ApplicationTailorRequest(BaseModel):
    master_resume: str
    job_description: str
    company_info: str | None = None
    user_preferences: str | None = None

    @field_validator("master_resume", "job_description")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class ApplicationTailorResponse(BaseModel):
    tailored_summary: str
    tailored_bullets: list[str]
    cover_letter_draft: str
    application_question_answers: list[str]
    recruiter_message_draft: str
    fit_gap_analysis: str
    interview_talking_points: list[str]


class ApplicationTailoringRunResponse(BaseModel):
    """Read schema for a persisted tailoring run. Does not expose raw resume/JD inputs."""

    model_config = {"from_attributes": True}

    id: int
    tailored_summary: str
    tailored_bullets: list[str]
    cover_letter_draft: str
    application_question_answers: list[str]
    recruiter_message_draft: str
    fit_gap_analysis: str
    interview_talking_points: list[str]
    provider_used: str
    fallback_used: bool
    created_at: datetime
