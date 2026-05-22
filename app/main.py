# Application entry point — wires together all routers and creates the FastAPI instance.
# Kept deliberately thin: no business logic lives here, just registration.
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.v1.routes.applications import router as applications_router
from app.api.v1.routes.jobs import router as jobs_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend API for an AI-powered job application workflow platform",
)

# Health router has no prefix — /health stays at the root for load-balancer checks.
app.include_router(health_router)
# /api/v1 prefix makes future API versioning a one-line change here, not scattered across routes.
app.include_router(applications_router, prefix="/api/v1/applications")
# RAG job matching routes — ingest, semantic search, and before/after comparison.
app.include_router(jobs_router, prefix="/api/v1/jobs")
