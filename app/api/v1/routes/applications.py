# HTTP layer — deliberately thin. Routes only validate input, call the repository
# or service layer, and return the response. No business logic lives here.
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.application_runs import (
    create_pending_run,
    get_application_tailoring_run,
)
from app.schemas.application import (
    ApplicationTailoringJobResponse,
    ApplicationTailoringRunResponse,
    ApplicationTailorRequest,
)
from app.services.background_tailoring import process_tailoring_job

router = APIRouter()


@router.post("/tailor", response_model=ApplicationTailoringJobResponse)
def tailor(
    request: ApplicationTailorRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ApplicationTailoringJobResponse:
    """
    Accept a tailoring request, persist it immediately, and enqueue
    background generation. Returns run_id and status=pending right away
    — do not wait for AI output.
    """
    run = create_pending_run(db, request)
    # Pass db explicitly so the background task uses the same session that is
    # alive for this request's lifetime — avoids a second get_db() call in the task.
    background_tasks.add_task(process_tailoring_job, run.id, db)
    # Return the pending receipt immediately; the caller polls GET /runs/{run_id}.
    return ApplicationTailoringJobResponse(run_id=run.id, status=run.status)


@router.get("/runs/{run_id}", response_model=ApplicationTailoringRunResponse)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> ApplicationTailoringRunResponse:
    """
    Return the current state of a tailoring run.

    - pending / processing: metadata + status only; output fields are null.
    - completed: full AI output is present.
    - failed: status + error_message; output fields are null.
    """
    run = get_application_tailoring_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    # model_validate with from_attributes=True maps the ORM instance to the
    # response schema without needing an explicit .dict() or field-by-field copy.
    return ApplicationTailoringRunResponse.model_validate(run)
