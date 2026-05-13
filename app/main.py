from fastapi import FastAPI

from app.api.routes.health import router as health_router

app = FastAPI(
    title="AI Job Search Assistant API",
    version="0.1.0",
    description="Backend API for an AI-powered job application workflow platform",
)

app.include_router(health_router)
