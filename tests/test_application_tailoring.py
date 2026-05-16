import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience in distributed systems.",
    "job_description": "Looking for a backend engineer to build scalable APIs using FastAPI.",
}


def test_tailor_returns_200():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_tailor_response_contains_job_fields():
    """POST now returns a job receipt: run_id and status, not the full AI output."""
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    assert "run_id" in body
    assert "status" in body
    assert isinstance(body["run_id"], int)
    assert body["status"] == "pending"


def test_completed_run_contains_all_output_fields(db_session):
    """
    After the background task runs, GET /runs/{id} returns all AI output fields.

    BackgroundTasks execute synchronously inside TestClient before the
    response is returned to the caller, so by the time client.post() returns
    the run is already in completed state.
    """
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    get_response = client.get(f"/api/v1/applications/runs/{run_id}")
    body = get_response.json()

    expected_output_fields = [
        "tailored_summary",
        "tailored_bullets",
        "cover_letter_draft",
        "application_question_answers",
        "recruiter_message_draft",
        "fit_gap_analysis",
        "interview_talking_points",
    ]
    for field in expected_output_fields:
        assert field in body, f"Missing field: {field}"
        assert body[field] is not None, f"Field is None: {field}"


def test_completed_run_tailored_bullets_is_non_empty_list(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert isinstance(body["tailored_bullets"], list)
    assert len(body["tailored_bullets"]) > 0


def test_completed_run_interview_talking_points_is_non_empty_list(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert isinstance(body["interview_talking_points"], list)
    assert len(body["interview_talking_points"]) > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"master_resume": "", "job_description": "Valid JD"},
        {"master_resume": "   ", "job_description": "Valid JD"},
    ],
)
def test_empty_master_resume_returns_422(payload):
    response = client.post("/api/v1/applications/tailor", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"master_resume": "Valid resume", "job_description": ""},
        {"master_resume": "Valid resume", "job_description": "   "},
    ],
)
def test_empty_job_description_returns_422(payload):
    response = client.post("/api/v1/applications/tailor", json=payload)
    assert response.status_code == 422
