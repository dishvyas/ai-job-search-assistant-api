# Health check endpoint — exists for load balancers, uptime monitors, and deployment
# smoke tests. No auth, no DB access: if this endpoint is reachable, the process is up.
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    # Returns a static dict — intentionally no DB or settings access so this
    # endpoint stays healthy even when downstream dependencies are degraded.
    return {"status": "ok", "service": "ai-job-search-assistant-api"}
