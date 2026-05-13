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


def test_tailor_response_contains_all_fields():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    expected_fields = [
        "tailored_summary",
        "tailored_bullets",
        "cover_letter_draft",
        "application_question_answers",
        "recruiter_message_draft",
        "fit_gap_analysis",
        "interview_talking_points",
    ]
    for field in expected_fields:
        assert field in body, f"Missing field: {field}"


def test_tailored_bullets_is_non_empty_list():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    bullets = response.json()["tailored_bullets"]
    assert isinstance(bullets, list)
    assert len(bullets) > 0


def test_interview_talking_points_is_non_empty_list():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    points = response.json()["interview_talking_points"]
    assert isinstance(points, list)
    assert len(points) > 0


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
