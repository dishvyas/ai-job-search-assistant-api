# Repository layer — all DB read/write operations for ApplicationTailoringRun live here.
# Keeping DB logic out of routes and services keeps each layer testable in isolation:
# routes can be tested without a real DB, and services can be tested without HTTP.
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.application import ApplicationTailoringRun
from app.models.run_status import RunStatus
from app.schemas.application import ApplicationTailorRequest
from app.schemas.llm_output import TailoringLLMOutput


def create_pending_run(
    db: Session,
    request: ApplicationTailorRequest,
) -> ApplicationTailoringRun:
    """Create a new run row in pending state and return it."""
    run = ApplicationTailoringRun(
        master_resume=request.master_resume,
        job_description=request.job_description,
        company_info=request.company_info,
        user_preferences=request.user_preferences,
        status=RunStatus.PENDING.value,
    )
    db.add(run)
    db.commit()
    # refresh so run.id is populated from the DB-generated primary key before returning.
    db.refresh(run)
    return run


def update_run_status(
    db: Session,
    run: ApplicationTailoringRun,
    status: str,
    error_message: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    latency_ms: int | None = None,
    generation_attempts: int | None = None,
) -> None:
    """
    Transition a run to a new status.

    Optional timing/attempt fields are set when provided — used both for
    the processing→failed path and can be extended for other transitions.
    """
    run.status = status
    # Only update optional fields when explicitly provided — avoids overwriting
    # previously-set values with None when the caller omits them.
    if error_message is not None:
        run.error_message = error_message
    if started_at is not None:
        run.started_at = started_at
    if completed_at is not None:
        run.completed_at = completed_at
    if latency_ms is not None:
        run.latency_ms = latency_ms
    if generation_attempts is not None:
        run.generation_attempts = generation_attempts
    db.add(run)
    db.commit()


def save_completed_run(
    db: Session,
    run: ApplicationTailoringRun,
    llm_output: TailoringLLMOutput,
    provider_used: str,
    fallback_used: bool,
    started_at: datetime,
    completed_at: datetime,
    latency_ms: int,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
    estimated_cost_usd: float,
    generation_attempts: int,
) -> None:
    """Persist generated output fields, workflow metadata, and mark the run completed."""
    # Append a visible signal to human-readable fields so callers can tell at a glance
    # that the output came from the fallback provider, not the configured one.
    fallback_note = " [Fallback mode used]" if fallback_used else ""

    run.tailored_summary = llm_output.tailored_summary + fallback_note
    run.tailored_bullets = llm_output.tailored_bullets
    run.cover_letter_draft = llm_output.cover_letter_draft
    run.application_question_answers = llm_output.application_question_answers
    run.recruiter_message_draft = llm_output.recruiter_message_draft
    run.fit_gap_analysis = llm_output.fit_gap_analysis + fallback_note
    run.interview_talking_points = llm_output.interview_talking_points
    run.provider_used = provider_used
    run.fallback_used = fallback_used
    run.status = RunStatus.COMPLETED.value

    # Workflow instrumentation
    run.started_at = started_at
    run.completed_at = completed_at
    run.latency_ms = latency_ms
    run.estimated_input_tokens = estimated_input_tokens
    run.estimated_output_tokens = estimated_output_tokens
    run.estimated_cost_usd = estimated_cost_usd
    run.generation_attempts = generation_attempts

    db.add(run)
    db.commit()


def get_application_tailoring_run(db: Session, run_id: int) -> ApplicationTailoringRun | None:
    """Return a single run by ID, or None if not found."""
    # db.get uses the identity map — returns the cached instance if already loaded
    # in this session, avoiding a redundant SELECT.
    return db.get(ApplicationTailoringRun, run_id)
