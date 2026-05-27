from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.models.application import ApplicationTailoringRun
from app.models.run_status import RunStatus
from app.prompts.agentic_tailoring import build_final_tailoring_prompt, build_fit_gap_prompt
from app.prompts.tailoring import build_tailoring_prompt
from app.rag.artifacts import (
    build_artifact_text,
    retrieve_similar_artifacts,
    store_artifact_embedding_for_run,
)
from app.rag.embed import EMBEDDING_DIM
from app.schemas.agent import FitGapAnalysis, JobDescriptionAnalysis, ResumeAnalysis
from app.schemas.application import ApplicationTailorRequest

client = TestClient(app)


def _make_completed_run(db_session, **overrides) -> ApplicationTailoringRun:
    run = ApplicationTailoringRun(
        master_resume="Original resume text should not be indexed.",
        job_description="Original JD text should not be indexed.",
        status=RunStatus.COMPLETED.value,
        tailored_summary="Tailored summary for a backend platform role.",
        tailored_bullets=[
            "Improved backend reliability by reducing incident volume.",
            "Built FastAPI services supporting async workflows.",
        ],
        cover_letter_draft="Cover letter draft.",
        application_question_answers=["Answer 1"],
        recruiter_message_draft="Concise recruiter note for a strong backend fit.",
        fit_gap_analysis="FIT: Python and APIs. GAP: some infra depth to learn.",
        interview_talking_points=["Talk about reliability.", "Talk about delivery pace."],
        provider_used="mock",
        fallback_used=False,
        generation_attempts=1,
        **overrides,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    return run


def _sample_request() -> ApplicationTailorRequest:
    return ApplicationTailorRequest(
        master_resume="Backend engineer with Python and FastAPI experience.",
        job_description="Platform engineer role using Python services.",
        company_info="B2B workflow company.",
        user_preferences="Emphasize reliability and delivery.",
    )


def test_build_artifact_text_includes_generated_output_fields(db_session):
    run = _make_completed_run(db_session)
    artifact_text = build_artifact_text(run)

    assert "Tailored summary for a backend platform role." in artifact_text
    assert "Improved backend reliability" in artifact_text
    assert "FIT: Python and APIs." in artifact_text
    assert "Talk about reliability." in artifact_text
    assert "Concise recruiter note" in artifact_text


def test_build_artifact_text_excludes_master_resume_and_job_description(db_session):
    run = _make_completed_run(db_session)
    artifact_text = build_artifact_text(run)

    assert run.master_resume not in artifact_text
    assert run.job_description not in artifact_text


def test_artifact_retrieval_disabled_by_default():
    defaults = Settings(
        app_name="test",
        environment="test",
        debug=False,
        llm_provider="mock",
        database_url="sqlite:///:memory:",
    )
    assert defaults.artifact_retrieval_enabled is False


def test_store_artifact_embedding_for_completed_run_when_enabled(db_session, monkeypatch):
    run = _make_completed_run(db_session)
    monkeypatch.setattr("app.rag.artifacts.settings.rag_enabled", True)
    monkeypatch.setattr("app.rag.artifacts.settings.artifact_retrieval_enabled", True)
    mock_vector = [0.2] * EMBEDDING_DIM
    monkeypatch.setattr("app.rag.artifacts.generate_embedding", lambda text: mock_vector)

    store_artifact_embedding_for_run(db_session, run)
    db_session.refresh(run)

    assert list(run.artifact_embedding) == mock_vector


def test_store_artifact_embedding_failure_is_best_effort(db_session, monkeypatch):
    run = _make_completed_run(db_session)
    monkeypatch.setattr("app.rag.artifacts.settings.rag_enabled", True)
    monkeypatch.setattr("app.rag.artifacts.settings.artifact_retrieval_enabled", True)
    monkeypatch.setattr(
        "app.rag.artifacts.generate_embedding",
        lambda text: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )

    store_artifact_embedding_for_run(db_session, run)
    db_session.refresh(run)

    assert run.status == RunStatus.COMPLETED.value
    assert run.artifact_embedding is None


def test_retrieve_similar_artifacts_returns_empty_when_disabled(db_session, monkeypatch):
    _make_completed_run(db_session, artifact_embedding=[0.1] * EMBEDDING_DIM)
    monkeypatch.setattr("app.rag.artifacts.settings.rag_enabled", True)
    monkeypatch.setattr("app.rag.artifacts.settings.artifact_retrieval_enabled", False)

    results = retrieve_similar_artifacts(db_session, "backend engineer")
    assert results == []


def test_single_step_prompt_includes_artifact_context_and_guardrails(db_session):
    artifact = _make_completed_run(db_session)
    prompt = build_tailoring_prompt(_sample_request(), artifact_context=[artifact])

    assert "Retrieved Past Tailored Artifacts" in prompt
    assert "Use them for tone, structure, and positioning inspiration only." in prompt
    assert "Do not copy claims." in prompt
    assert "Do not invent experience not present in the candidate resume." in prompt
    assert "source of truth" in prompt
    assert artifact.master_resume not in prompt
    assert artifact.job_description not in prompt


def test_agentic_prompts_include_artifact_context_and_guardrails(db_session):
    artifact = _make_completed_run(db_session)
    request = _sample_request()
    resume = ResumeAnalysis(
        key_skills=["Python", "FastAPI"],
        relevant_experience=["Backend APIs"],
        strengths=["Reliability"],
    )
    jd = JobDescriptionAnalysis(
        required_skills=["Python", "APIs"],
        responsibilities=["Build services"],
        role_focus="Platform engineering",
    )
    fit_gap = FitGapAnalysis(
        fit_points=["Python backend depth"],
        gap_points=["Some infra ramp-up"],
        positioning_strategy="Emphasize reliability",
    )

    fit_gap_prompt = build_fit_gap_prompt(request, resume, jd, artifact_context=[artifact])
    final_prompt = build_final_tailoring_prompt(
        request,
        resume,
        jd,
        fit_gap,
        artifact_context=[artifact],
    )

    assert "Retrieved Past Tailored Artifacts" in fit_gap_prompt
    assert "Do not copy claims or invent experience" in fit_gap_prompt
    assert "Retrieved Past Tailored Artifacts" in final_prompt
    assert "Do not copy claims." in final_prompt
    assert artifact.master_resume not in final_prompt
    assert artifact.job_description not in final_prompt


def test_single_step_workflow_still_works_when_artifact_retrieval_disabled(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "single_step")
    monkeypatch.setattr("app.services.background_tailoring.settings.rag_enabled", False)
    monkeypatch.setattr(
        "app.services.background_tailoring.settings.artifact_retrieval_enabled", False
    )

    response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Backend engineer with Python experience.",
            "job_description": "FastAPI backend role.",
        },
    )
    run_id = response.json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["status"] == RunStatus.COMPLETED.value
    assert body["provider_used"] == "mock"


def test_agentic_workflow_still_works_when_artifact_retrieval_disabled(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.background_tailoring.settings.rag_enabled", False)
    monkeypatch.setattr(
        "app.services.background_tailoring.settings.artifact_retrieval_enabled", False
    )
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Backend engineer with Python experience.",
            "job_description": "FastAPI backend role.",
        },
    )
    run_id = response.json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["status"] == RunStatus.COMPLETED.value
    assert body["provider_used"] == "mock"


def test_background_tailoring_artifact_embedding_failure_does_not_fail_run(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "single_step")
    monkeypatch.setattr("app.services.background_tailoring.settings.rag_enabled", False)
    monkeypatch.setattr(
        "app.services.background_tailoring.store_artifact_embedding_for_run",
        lambda db, run: (_ for _ in ()).throw(RuntimeError("artifact index failed")),
    )

    response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Backend engineer with Python experience.",
            "job_description": "FastAPI backend role.",
        },
    )
    run_id = response.json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert body["status"] == RunStatus.COMPLETED.value
