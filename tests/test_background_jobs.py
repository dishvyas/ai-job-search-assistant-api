"""
Tests for Milestone 6 — Background Job Processing.

Key testing assumption
----------------------
FastAPI's BackgroundTasks run synchronously inside Starlette's TestClient
before the HTTP response is returned to the caller. This means that after
client.post("/api/v1/applications/tailor", ...) returns, the background
task has already run and the DB row is in its final state.

The POST response body still reflects the *initial* state (pending) because
the response object is constructed before the background task executes.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.models.application import ApplicationTailoringRun
from app.models.run_status import RunStatus

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}


# ---------------------------------------------------------------------------
# POST response — job receipt
# ---------------------------------------------------------------------------


def test_post_tailor_returns_pending_status():
    """POST must immediately return run_id and status=pending."""
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == RunStatus.PENDING.value
    assert isinstance(body["run_id"], int)


def test_post_tailor_does_not_return_output_fields():
    """The immediate response is a job receipt, not the full AI output."""
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    assert "tailored_summary" not in body
    assert "cover_letter_draft" not in body
    assert "tailored_bullets" not in body


# ---------------------------------------------------------------------------
# Background task — happy path
# ---------------------------------------------------------------------------


def test_background_task_completes_run(db_session):
    """After POST returns, the background task must have completed the run."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.status == RunStatus.COMPLETED.value


def test_completed_run_has_no_error_message(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.error_message is None


def test_completed_job_persists_generated_fields(db_session):
    """Output columns must be populated after successful background processing."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    assert run.tailored_summary is not None
    assert isinstance(run.tailored_bullets, list) and len(run.tailored_bullets) > 0
    assert run.cover_letter_draft is not None
    assert isinstance(run.application_question_answers, list)
    assert run.recruiter_message_draft is not None
    assert run.fit_gap_analysis is not None
    assert isinstance(run.interview_talking_points, list) and len(run.interview_talking_points) > 0


def test_completed_job_stores_provider_used(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.provider_used == "mock"
    assert run.fallback_used is False


# ---------------------------------------------------------------------------
# GET endpoint — status-aware responses
# ---------------------------------------------------------------------------


def test_get_run_returns_completed_status(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert body["status"] == RunStatus.COMPLETED.value


def test_get_run_returns_completed_output(db_session):
    """GET for a completed run must include all AI output fields."""
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["tailored_summary"] is not None
    assert isinstance(body["tailored_bullets"], list)
    assert body["cover_letter_draft"] is not None
    assert isinstance(body["application_question_answers"], list)
    assert body["recruiter_message_draft"] is not None
    assert body["fit_gap_analysis"] is not None
    assert isinstance(body["interview_talking_points"], list)


def test_single_step_completed_run_has_null_agent_decision_fields(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["route_decision"] is None
    assert body["review_notes"] is None
    assert body["revision_needed"] is None
    assert body["retrieved_context_count"] is None
    assert body["artifact_context_count"] is None


# ---------------------------------------------------------------------------
# Failed jobs
# ---------------------------------------------------------------------------


def test_failed_job_persists_failed_status(db_session, monkeypatch):
    """When background processing raises an unexpected exception, status=failed."""

    def _always_fail(prompt: str):
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.status == RunStatus.FAILED.value


def test_failed_job_persists_error_message(db_session, monkeypatch):
    """The exception message must be stored in error_message for debugging."""

    def _always_fail(prompt: str):
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.error_message == "Simulated processing failure"


def test_failed_job_output_fields_remain_null(db_session, monkeypatch):
    """A failed run must not have any AI output fields populated."""

    def _always_fail(prompt: str):
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.tailored_summary is None
    assert run.tailored_bullets is None
    assert run.cover_letter_draft is None


def test_get_failed_run_returns_failed_status(db_session, monkeypatch):
    """GET for a failed run exposes status and error_message."""

    def _always_fail(prompt: str):
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert body["status"] == RunStatus.FAILED.value
    assert body["error_message"] == "Simulated processing failure"
    assert body["tailored_summary"] is None


# ---------------------------------------------------------------------------
# Mock mode and structured parsing still work end-to-end
# ---------------------------------------------------------------------------


def test_mock_mode_produces_completed_run(db_session):
    """End-to-end: mock provider → background task → completed run with output."""
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert body["status"] == RunStatus.COMPLETED.value
    assert body["provider_used"] == "mock"
    assert body["fallback_used"] is False
    assert body["tailored_summary"] is not None


def test_structured_parsing_produces_list_fields(db_session):
    """Mock LLM returns valid JSON; parsed list fields must survive the full pipeline."""
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert isinstance(body["tailored_bullets"], list) and len(body["tailored_bullets"]) > 0
    assert (
        isinstance(body["interview_talking_points"], list)
        and len(body["interview_talking_points"]) > 0
    )
