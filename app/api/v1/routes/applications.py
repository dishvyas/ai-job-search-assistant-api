from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.application_runs import get_application_tailoring_run
from app.schemas.application import (
    ApplicationTailoringRunResponse,
    ApplicationTailorRequest,
    ApplicationTailorResponse,
)
from app.services.application_tailoring import tailor_application

router = APIRouter()


@router.post("/tailor", response_model=ApplicationTailorResponse)
def tailor(
    request: ApplicationTailorRequest,
    db: Session = Depends(get_db),
) -> ApplicationTailorResponse:
    return tailor_application(request, db)


@router.get("/runs/{run_id}", response_model=ApplicationTailoringRunResponse)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> ApplicationTailoringRunResponse:
    run = get_application_tailoring_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return ApplicationTailoringRunResponse.model_validate(run)
