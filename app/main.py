from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.v1.routes.applications import router as applications_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend API for an AI-powered job application workflow platform",
)

app.include_router(health_router)
app.include_router(applications_router, prefix="/api/v1/applications")
