from sqlalchemy.orm import Session

from app.models.application import ApplicationTailoringRun
from app.schemas.application import ApplicationTailorRequest, ApplicationTailorResponse


def create_application_tailoring_run(
    db: Session,
    request: ApplicationTailorRequest,
    response: ApplicationTailorResponse,
    provider_used: str,
    fallback_used: bool,
) -> ApplicationTailoringRun:
    """Persist a tailoring request + response and return the saved record."""
    run = ApplicationTailoringRun(
        master_resume=request.master_resume,
        job_description=request.job_description,
        company_info=request.company_info,
        user_preferences=request.user_preferences,
        tailored_summary=response.tailored_summary,
        tailored_bullets=response.tailored_bullets,
        cover_letter_draft=response.cover_letter_draft,
        application_question_answers=response.application_question_answers,
        recruiter_message_draft=response.recruiter_message_draft,
        fit_gap_analysis=response.fit_gap_analysis,
        interview_talking_points=response.interview_talking_points,
        provider_used=provider_used,
        fallback_used=fallback_used,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_application_tailoring_run(db: Session, run_id: int) -> ApplicationTailoringRun | None:
    """Return a single run by ID, or None if not found."""
    return db.get(ApplicationTailoringRun, run_id)
