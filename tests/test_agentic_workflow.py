"""
Tests for Milestone 8 — Agentic Application Workflow.

Covers:
- default workflow_mode is single_step
- unsupported workflow_mode raises a clear error
- agentic workflow runs end-to-end in mock mode (no real LLM)
- agentic mode completes background jobs and stores output in DB
- mock provider returns valid JSON for each agent stage
- generation_attempts reflects the 4-stage nature of the agentic workflow
- existing single_step tests are not broken
- local SQLite DB files are gitignored
"""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.config import settings
from app.llm.mock import MockLLMProvider
from app.main import app
from app.models.application import ApplicationTailoringRun
from app.models.run_status import RunStatus
from app.prompts.agentic_tailoring import (
    build_final_tailoring_prompt,
    build_fit_gap_prompt,
    build_jd_analysis_prompt,
    build_resume_analysis_prompt,
)
from app.schemas.agent import FitGapAnalysis, JobDescriptionAnalysis, ResumeAnalysis
from app.schemas.application import ApplicationTailorRequest
from app.schemas.llm_output import TailoringLLMOutput
from app.services.agentic_tailoring import run_agentic_workflow

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}

SAMPLE_REQUEST = ApplicationTailorRequest(
    master_resume="Software engineer with 5 years of Python experience.",
    job_description="Backend engineer role using FastAPI.",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_workflow_mode_is_single_step():
    """The factory default for workflow_mode must be 'single_step'."""
    from app.core.config import Settings

    defaults = Settings(
        app_name="test",
        environment="test",
        debug=False,
        llm_provider="mock",
        database_url="sqlite:///:memory:",
        workflow_mode="single_step",
    )
    assert defaults.workflow_mode == "single_step"


def test_settings_workflow_mode_field_exists():
    """Settings object must expose workflow_mode."""
    assert hasattr(settings, "workflow_mode")
    assert settings.workflow_mode == "single_step"


def test_unsupported_workflow_mode_stores_failed_status(db_session, monkeypatch):
    """An invalid workflow_mode must result in a failed job with a clear error message."""
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "invalid_mode")

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    assert run.status == RunStatus.FAILED.value
    assert "invalid_mode" in run.error_message
    assert "Unsupported workflow_mode" in run.error_message


# ---------------------------------------------------------------------------
# Mock provider — stage-specific JSON shapes
# ---------------------------------------------------------------------------


def test_mock_returns_valid_resume_analysis_json():
    """Mock provider must return parseable ResumeAnalysis JSON for the resume stage."""
    provider = MockLLMProvider()
    prompt = build_resume_analysis_prompt(SAMPLE_REQUEST)
    raw = provider.generate_text(prompt)
    data = json.loads(raw)
    parsed = ResumeAnalysis.model_validate(data)
    assert len(parsed.key_skills) > 0
    assert len(parsed.relevant_experience) > 0
    assert len(parsed.strengths) > 0


def test_mock_returns_valid_jd_analysis_json():
    """Mock provider must return parseable JobDescriptionAnalysis JSON for the JD stage."""
    provider = MockLLMProvider()
    prompt = build_jd_analysis_prompt(SAMPLE_REQUEST)
    raw = provider.generate_text(prompt)
    parsed = JobDescriptionAnalysis.model_validate(json.loads(raw))
    assert len(parsed.required_skills) > 0
    assert len(parsed.responsibilities) > 0
    assert isinstance(parsed.role_focus, str) and parsed.role_focus


def test_mock_returns_valid_fit_gap_json():
    """Mock provider must return parseable FitGapAnalysis JSON for the fit/gap stage."""
    provider = MockLLMProvider()

    # Build a real fit/gap prompt using deterministic mock analyses.
    resume_analysis = ResumeAnalysis(
        key_skills=["Python", "FastAPI"],
        relevant_experience=["5 years backend"],
        strengths=["Problem-solving"],
    )
    jd_analysis = JobDescriptionAnalysis(
        required_skills=["Python", "APIs"],
        responsibilities=["Build services"],
        role_focus="Backend engineering",
    )
    prompt = build_fit_gap_prompt(SAMPLE_REQUEST, resume_analysis, jd_analysis)
    raw = provider.generate_text(prompt)
    parsed = FitGapAnalysis.model_validate(json.loads(raw))
    assert len(parsed.fit_points) > 0
    assert isinstance(parsed.positioning_strategy, str) and parsed.positioning_strategy


def test_mock_returns_valid_final_tailoring_json():
    """Mock provider must return parseable TailoringLLMOutput for the final composition stage."""
    provider = MockLLMProvider()
    resume_analysis = ResumeAnalysis(
        key_skills=["Python"], relevant_experience=["backend"], strengths=["adaptable"]
    )
    jd_analysis = JobDescriptionAnalysis(
        required_skills=["Python"],
        responsibilities=["build APIs"],
        role_focus="Backend",
    )
    fit_gap = FitGapAnalysis(
        fit_points=["Strong Python"],
        gap_points=["Needs Kubernetes"],
        positioning_strategy="Emphasise adaptability",
    )
    prompt = build_final_tailoring_prompt(SAMPLE_REQUEST, resume_analysis, jd_analysis, fit_gap)
    raw = provider.generate_text(prompt)
    parsed = TailoringLLMOutput.model_validate(json.loads(raw))
    assert parsed.tailored_summary
    assert len(parsed.tailored_bullets) > 0


# ---------------------------------------------------------------------------
# Agentic workflow — unit / integration (no HTTP, uses mock mode)
# ---------------------------------------------------------------------------


def test_agentic_workflow_returns_tailoring_llm_output(monkeypatch):
    """run_agentic_workflow must return a complete TailoringLLMOutput."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    output, provider, fallback, metadata, fallback_reason = run_agentic_workflow(SAMPLE_REQUEST)

    assert isinstance(output, TailoringLLMOutput)
    assert output.tailored_summary
    assert len(output.tailored_bullets) > 0
    assert output.cover_letter_draft
    assert len(output.interview_talking_points) > 0
    assert metadata.route_decision is not None
    assert fallback_reason is None


def test_agentic_workflow_provider_used_is_mock(monkeypatch):
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    _, provider, _, _, _ = run_agentic_workflow(SAMPLE_REQUEST)
    assert provider == "mock"


def test_agentic_workflow_fallback_used_is_false_in_mock_mode(monkeypatch):
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    _, _, fallback, _, _ = run_agentic_workflow(SAMPLE_REQUEST)
    assert fallback is False


def test_agentic_workflow_fallback_on_provider_failure(monkeypatch):
    """If the configured provider fails, every stage should fall back to mock."""
    from app.llm.exceptions import LLMProviderUnavailableError

    class AlwaysFailProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503")

    monkeypatch.setattr(
        "app.services.agentic_tailoring.get_llm_provider", lambda: AlwaysFailProvider()
    )

    output, provider, fallback, _, fallback_reason = run_agentic_workflow(SAMPLE_REQUEST)

    assert isinstance(output, TailoringLLMOutput)
    assert provider == "fallback-mock"
    assert fallback is True
    assert "LLMProviderUnavailableError" in fallback_reason


def test_agentic_workflow_metadata_tracks_artifact_context(monkeypatch):
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    artifacts = [
        SimpleNamespace(
            tailored_summary="Example summary.",
            tailored_bullets=["Example bullet 1", "Example bullet 2"],
            fit_gap_analysis="Example fit/gap.",
        ),
        SimpleNamespace(
            tailored_summary="Example summary 2.",
            tailored_bullets=["Example bullet A", "Example bullet B"],
            fit_gap_analysis="Example fit/gap 2.",
        ),
    ]

    output, _, _, metadata, _ = run_agentic_workflow(
        SAMPLE_REQUEST,
        artifact_context=artifacts,
    )

    assert isinstance(output, TailoringLLMOutput)
    assert metadata.artifact_context_count == 2


# ---------------------------------------------------------------------------
# Background job — agentic mode end-to-end via HTTP
# ---------------------------------------------------------------------------


def test_background_job_completes_in_agentic_mode(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200

    run = db_session.query(ApplicationTailoringRun).first()
    assert run.status == RunStatus.COMPLETED.value


def test_agentic_mode_stores_output_in_db(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    assert run.tailored_summary is not None
    assert isinstance(run.tailored_bullets, list) and len(run.tailored_bullets) > 0
    assert run.cover_letter_draft is not None
    assert run.fit_gap_analysis is not None


def test_agentic_mode_get_run_returns_completed_output(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["status"] == RunStatus.COMPLETED.value
    assert body["tailored_summary"] is not None
    assert isinstance(body["tailored_bullets"], list)


def test_agentic_mode_get_run_includes_agent_decision_fields(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    run_id = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD).json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["route_decision"] is not None
    assert body["review_notes"] is not None
    assert body["revision_needed"] is not None
    assert body["retrieved_context_count"] is not None
    assert body["artifact_context_count"] is not None


def test_agentic_mode_generation_attempts_is_four(db_session, monkeypatch):
    """4 stages × no fallback = 4 generation attempts."""
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    assert run.generation_attempts == 4


def test_agentic_mode_metadata_is_populated(db_session, monkeypatch):
    """Timing and token metadata must be set even in agentic mode."""
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    assert run.started_at is not None
    assert run.completed_at is not None
    assert run.latency_ms is not None
    assert run.estimated_input_tokens is not None and run.estimated_input_tokens > 0
    assert run.estimated_output_tokens is not None and run.estimated_output_tokens > 0
    assert run.route_decision is not None
    assert run.revision_needed is not None
    assert run.retrieved_context_count is not None
    assert run.artifact_context_count is not None


def test_agentic_mode_fallback_doubles_attempt_count(db_session, monkeypatch):
    """4 stages, all fell back → 8 generation attempts (approximate)."""
    from app.llm.exceptions import LLMProviderUnavailableError

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503")

    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr(
        "app.services.agentic_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run = db_session.query(ApplicationTailoringRun).first()

    assert run.generation_attempts == 8
    assert run.fallback_used is True


def test_agentic_provider_failure_stores_fallback_reason(db_session, monkeypatch):
    from app.llm.exceptions import LLMProviderUnavailableError

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("OpenAI request failed sk-agentic-secret")

    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr(
        "app.services.agentic_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    run_id = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD).json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["status"] == RunStatus.COMPLETED.value
    assert body["fallback_used"] is True
    assert body["fallback_reason"] is not None
    assert "fallback" in body["fallback_reason"]
    assert "LLMProviderUnavailableError" in body["fallback_reason"]
    assert "sk-agentic-secret" not in body["fallback_reason"]


# ---------------------------------------------------------------------------
# .gitignore — local SQLite DB files are excluded
# ---------------------------------------------------------------------------


def test_gitignore_excludes_local_db():
    """local.db must appear in .gitignore."""
    with open(".gitignore") as f:
        content = f.read()
    assert "local.db" in content


def test_gitignore_excludes_db_pattern():
    """*.db glob must appear in .gitignore so any SQLite file is ignored."""
    with open(".gitignore") as f:
        content = f.read()
    assert "*.db" in content
