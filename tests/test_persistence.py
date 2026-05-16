from fastapi.testclient import TestClient

from app.llm.exceptions import LLMProviderUnavailableError
from app.main import app
from app.models.application import ApplicationTailoringRun

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}


# ---------------------------------------------------------------------------
# POST /tailor — persistence
# ---------------------------------------------------------------------------


def test_tailor_returns_200(db_session):
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_tailor_stores_one_db_row(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    rows = db_session.query(ApplicationTailoringRun).all()
    assert len(rows) == 1


def test_tailor_stores_correct_provider_used(db_session):
    """Default mock mode should store provider_used='mock'."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.provider_used == "mock"
    assert run.fallback_used is False


def test_tailor_stores_correct_list_fields(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert isinstance(run.tailored_bullets, list)
    assert len(run.tailored_bullets) > 0
    assert isinstance(run.interview_talking_points, list)
    assert len(run.interview_talking_points) > 0


def test_tailor_stores_optional_fields_when_provided(db_session):
    payload = {**VALID_PAYLOAD, "company_info": "A fintech startup."}
    client.post("/api/v1/applications/tailor", json=payload)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.company_info == "A fintech startup."


def test_tailor_stores_none_for_absent_optional_fields(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.company_info is None
    assert run.user_preferences is None


def test_tailor_fallback_metadata_is_stored(db_session, monkeypatch):
    """When provider fails, fallback_used=True and provider_used='fallback-mock' are stored."""

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.fallback_used is True
    assert run.provider_used == "fallback-mock"


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------


def test_get_run_returns_stored_output(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert post_response.status_code == 200

    run = db_session.query(ApplicationTailoringRun).first()
    get_response = client.get(f"/api/v1/applications/runs/{run.id}")

    assert get_response.status_code == 200
    body = get_response.json()
    assert body["id"] == run.id
    assert body["tailored_summary"] == run.tailored_summary
    assert body["provider_used"] == "mock"
    assert body["fallback_used"] is False


def test_get_run_returns_correct_list_fields(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    response = client.get(f"/api/v1/applications/runs/{run.id}")
    body = response.json()

    assert isinstance(body["tailored_bullets"], list)
    assert len(body["tailored_bullets"]) > 0
    assert isinstance(body["interview_talking_points"], list)


def test_get_run_returns_404_for_missing_id(db_session):
    response = client.get("/api/v1/applications/runs/99999")
    assert response.status_code == 404


def test_get_run_does_not_expose_raw_resume(db_session):
    """The GET response must not include master_resume or job_description."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    response = client.get(f"/api/v1/applications/runs/{run.id}")
    body = response.json()

    assert "master_resume" not in body
    assert "job_description" not in body


def test_get_run_returns_created_at(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    response = client.get(f"/api/v1/applications/runs/{run.id}")
    body = response.json()

    assert "created_at" in body
    assert body["created_at"] is not None
