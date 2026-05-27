"""
Tests for Milestone 7 — Workflow Metadata Tracking.

Covers:
- token estimation utilities (unit tests, no DB)
- cost estimation utilities (unit tests, no DB)
- timing fields populated on completed runs
- token/cost fields populated on completed runs
- generation_attempts logic (normal vs fallback vs failed)
- failed runs still capture partial timing metadata
- GET endpoint exposes all new metadata fields
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.llm.cost_estimation import estimate_generation_cost
from app.llm.exceptions import LLMProviderUnavailableError
from app.llm.token_estimation import estimate_input_tokens, estimate_output_tokens
from app.main import app
from app.models.application import ApplicationTailoringRun

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}


# ---------------------------------------------------------------------------
# Token estimation — unit tests (no DB, no HTTP)
# ---------------------------------------------------------------------------


def test_estimate_input_tokens_returns_positive_int():
    tokens = estimate_input_tokens("This is a short prompt with eight words here")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_estimate_input_tokens_word_count_approximation():
    prompt = "one two three four five"
    assert estimate_input_tokens(prompt) == 5


def test_estimate_output_tokens_returns_positive_int():
    tokens = estimate_output_tokens('{"key": "value with several words in it"}')
    assert isinstance(tokens, int)
    assert tokens > 0


def test_estimate_output_tokens_empty_string_returns_zero():
    assert estimate_output_tokens("") == 0


def test_longer_prompt_produces_more_input_tokens():
    short = "hello world"
    long = "hello world " * 50
    assert estimate_input_tokens(long) > estimate_input_tokens(short)


# ---------------------------------------------------------------------------
# Cost estimation — unit tests (no DB, no HTTP)
# ---------------------------------------------------------------------------


def test_mock_provider_cost_is_zero():
    cost = estimate_generation_cost(500, 200, "mock")
    assert cost == 0.0


def test_fallback_mock_provider_cost_is_zero():
    cost = estimate_generation_cost(500, 200, "fallback-mock")
    assert cost == 0.0


def test_gemini_provider_cost_is_positive():
    cost = estimate_generation_cost(1000, 500, "gemini")
    assert cost > 0.0


def test_gemini_cost_scales_with_token_count():
    small_cost = estimate_generation_cost(100, 50, "gemini")
    large_cost = estimate_generation_cost(10_000, 5_000, "gemini")
    assert large_cost > small_cost


def test_unknown_provider_cost_defaults_to_zero():
    cost = estimate_generation_cost(500, 200, "unknown-provider")
    assert cost == 0.0


def test_openai_provider_cost_is_positive():
    cost = estimate_generation_cost(1000, 500, "openai")
    assert cost > 0.0


def test_cost_returns_float():
    cost = estimate_generation_cost(100, 50, "gemini")
    assert isinstance(cost, float)


# ---------------------------------------------------------------------------
# Workflow timing — DB / integration tests
# ---------------------------------------------------------------------------


def test_completed_run_has_started_at(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.started_at is not None
    assert isinstance(run.started_at, datetime)


def test_completed_run_has_completed_at(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.completed_at is not None
    assert isinstance(run.completed_at, datetime)


def test_started_at_before_completed_at(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.started_at <= run.completed_at


def test_completed_run_has_latency_ms(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.latency_ms is not None
    assert isinstance(run.latency_ms, int)
    assert run.latency_ms >= 0


def test_latency_ms_consistent_with_timestamps(db_session):
    """latency_ms should equal the millisecond delta of started_at→completed_at."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    delta_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)
    assert run.latency_ms == delta_ms


# ---------------------------------------------------------------------------
# Token and cost fields — DB / integration tests
# ---------------------------------------------------------------------------


def test_completed_run_has_estimated_input_tokens(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.estimated_input_tokens is not None
    assert run.estimated_input_tokens > 0


def test_completed_run_has_estimated_output_tokens(db_session):
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.estimated_output_tokens is not None
    assert run.estimated_output_tokens > 0


def test_mock_provider_estimated_cost_is_zero(db_session):
    """Mock provider has no real API, so estimated cost must be 0."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# generation_attempts — normal and fallback paths
# ---------------------------------------------------------------------------


def test_successful_run_has_one_generation_attempt(db_session):
    """Mock provider succeeds on first try — exactly one attempt."""
    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.generation_attempts == 1


def test_fallback_run_has_two_generation_attempts(db_session, monkeypatch):
    """Primary provider fails → fallback mock runs: two attempts total."""

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.generation_attempts == 2


def test_failed_run_generation_attempts_is_one(db_session, monkeypatch):
    """When _get_llm_output raises (non-LLM error), one attempt was made."""

    def _always_fail(prompt: str):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.generation_attempts == 1


# ---------------------------------------------------------------------------
# Failed run — timing metadata still captured
# ---------------------------------------------------------------------------


def test_failed_run_has_started_at(db_session, monkeypatch):
    def _always_fail(prompt: str):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.started_at is not None


def test_failed_run_has_completed_at(db_session, monkeypatch):
    def _always_fail(prompt: str):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.completed_at is not None


def test_failed_run_has_latency_ms(db_session, monkeypatch):
    def _always_fail(prompt: str):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()
    assert run.latency_ms is not None
    assert run.latency_ms >= 0


# ---------------------------------------------------------------------------
# GET endpoint exposes metadata fields
# ---------------------------------------------------------------------------


def test_get_completed_run_exposes_timing_fields(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["started_at"] is not None
    assert body["completed_at"] is not None
    assert body["latency_ms"] is not None
    assert body["latency_ms"] >= 0


def test_get_completed_run_exposes_token_fields(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["estimated_input_tokens"] is not None
    assert body["estimated_input_tokens"] > 0
    assert body["estimated_output_tokens"] is not None
    assert body["estimated_output_tokens"] > 0


def test_get_completed_run_exposes_cost_and_attempts(db_session):
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert "estimated_cost_usd" in body
    assert body["estimated_cost_usd"] == 0.0  # mock provider is free
    assert body["generation_attempts"] == 1


def test_get_failed_run_exposes_timing_fields(db_session, monkeypatch):
    """Even failed runs expose timing metadata via GET."""

    def _always_fail(prompt: str):
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr("app.services.background_tailoring._get_llm_output", _always_fail)

    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["started_at"] is not None
    assert body["completed_at"] is not None
    assert body["latency_ms"] is not None
    assert body["generation_attempts"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "started_at",
        "completed_at",
        "latency_ms",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "estimated_cost_usd",
        "generation_attempts",
    ],
)
def test_all_metadata_fields_present_in_get_response(db_session, field):
    """Every metadata field must be present in the GET response schema."""
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert field in body, f"Metadata field missing from GET response: {field}"
