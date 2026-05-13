from fastapi import APIRouter

from app.schemas.application import ApplicationTailorRequest, ApplicationTailorResponse
from app.services.application_tailoring import tailor_application

router = APIRouter()


@router.post("/tailor", response_model=ApplicationTailorResponse)
def tailor(request: ApplicationTailorRequest) -> ApplicationTailorResponse:
    return tailor_application(request)
