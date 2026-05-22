# API-layer schemas — separate from ORM models so that the DB structure and the
# HTTP contract can evolve independently. Pydantic validates all input at the
# boundary; invalid requests are rejected with 422 before reaching any service code.
from datetime import datetime

from pydantic import BaseModel, field_validator


class ApplicationTailorRequest(BaseModel):
    master_resume: str
    job_description: str
    # Optional fields — not all callers have company context or style preferences.
    company_info: str | None = None
    user_preferences: str | None = None

    # Explicit whitespace-only check because Pydantic accepts "   " as a valid str.
    @field_validator("master_resume", "job_description")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class ApplicationTailorResponse(BaseModel):
    """Full AI output — used internally by the background task."""

    tailored_summary: str
    tailored_bullets: list[str]
    cover_letter_draft: str
    application_question_answers: list[str]
    recruiter_message_draft: str
    fit_gap_analysis: str
    interview_talking_points: list[str]


class ApplicationTailoringJobResponse(BaseModel):
    """
    Immediate response returned by POST /tailor.

    The run is created in pending state and processed asynchronously.
    Poll GET /runs/{run_id} to retrieve the generated output once completed.
    """

    # Intentionally minimal — callers only need the ID to poll; they don't
    # need the full run record until generation is complete.
    run_id: int
    status: str


class ApplicationTailoringRunResponse(BaseModel):
    """
    Read schema for GET /runs/{run_id}.

    Output fields are None while the run is pending or processing.
    Check `status` before reading output fields:
      - pending / processing  → output fields are None
      - completed             → all output fields are populated
      - failed                → error_message describes the failure
    """

    # from_attributes=True lets Pydantic read values from ORM model attributes
    # instead of requiring a dict — enables model_validate(orm_instance) directly.
    model_config = {"from_attributes": True}

    id: int
    status: str
    error_message: str | None = None

    # AI output — populated only when status == "completed"
    tailored_summary: str | None = None
    tailored_bullets: list[str] | None = None
    cover_letter_draft: str | None = None
    application_question_answers: list[str] | None = None
    recruiter_message_draft: str | None = None
    fit_gap_analysis: str | None = None
    interview_talking_points: list[str] | None = None

    provider_used: str | None = None
    fallback_used: bool
    created_at: datetime

    # Workflow instrumentation — populated after background task runs
    started_at: datetime | None = None
    completed_at: datetime | None = None
    latency_ms: int | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    generation_attempts: int
