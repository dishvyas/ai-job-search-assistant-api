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
    db.refresh(run)
    return run


def update_run_status(
    db: Session,
    run: ApplicationTailoringRun,
    status: str,
    error_message: str | None = None,
) -> None:
    """Transition a run to a new status, optionally recording an error message."""
    run.status = status
    if error_message is not None:
        run.error_message = error_message
    db.add(run)
    db.commit()


def save_completed_run(
    db: Session,
    run: ApplicationTailoringRun,
    llm_output: TailoringLLMOutput,
    provider_used: str,
    fallback_used: bool,
) -> None:
    """Persist generated output fields and mark the run as completed."""
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

    db.add(run)
    db.commit()


def get_application_tailoring_run(db: Session, run_id: int) -> ApplicationTailoringRun | None:
    """Return a single run by ID, or None if not found."""
    return db.get(ApplicationTailoringRun, run_id)
