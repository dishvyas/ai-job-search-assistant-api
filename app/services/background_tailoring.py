from sqlalchemy.orm import Session

from app.models.run_status import RunStatus
from app.prompts.tailoring import build_tailoring_prompt
from app.repositories.application_runs import (
    get_application_tailoring_run,
    save_completed_run,
    update_run_status,
)
from app.schemas.application import ApplicationTailorRequest
from app.services.application_tailoring import _get_llm_output


def process_tailoring_job(run_id: int, db: Session) -> None:
    """
    Background task: generate tailoring output for a pending run.

    Flow
    ----
    1. Load the run row.
    2. Transition to processing.
    3. Reconstruct the original request and build the LLM prompt.
    4. Call _get_llm_output (includes fallback logic).
    5. Persist output and transition to completed.

    On any unexpected exception the run is marked failed and the error
    message is stored. This prevents the job from silently disappearing
    and gives the caller something useful to read on GET.
    """
    run = get_application_tailoring_run(db, run_id)
    if run is None:
        # Should never happen — run was just created by the route.
        return

    update_run_status(db, run, RunStatus.PROCESSING.value)

    try:
        # Reconstruct the request object so we can reuse build_tailoring_prompt.
        request = ApplicationTailorRequest(
            master_resume=run.master_resume,
            job_description=run.job_description,
            company_info=run.company_info,
            user_preferences=run.user_preferences,
        )
        prompt = build_tailoring_prompt(request)
        llm_output, provider_used, used_fallback = _get_llm_output(prompt)
        save_completed_run(
            db=db,
            run=run,
            llm_output=llm_output,
            provider_used=provider_used,
            fallback_used=used_fallback,
        )
    except Exception as exc:  # noqa: BLE001
        update_run_status(db, run, RunStatus.FAILED.value, error_message=str(exc))
