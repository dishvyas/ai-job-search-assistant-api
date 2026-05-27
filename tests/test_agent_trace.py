import json

from fastapi.testclient import TestClient

from app.main import app
from app.models.agent_trace import AgentTraceStep
from app.repositories.application_runs import create_pending_run
from app.schemas.application import ApplicationTailorRequest
from app.schemas.llm_output import TailoringLLMOutput
from app.services.agentic_tailoring import run_agentic_workflow

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Senior backend engineer with 6 years building Python APIs at scale.",
    "job_description": "Backend platform engineer role focused on FastAPI services.",
}


def test_agentic_workflow_creates_trace_steps(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = response.json()["run_id"]

    steps = (
        db_session.query(AgentTraceStep)
        .filter(AgentTraceStep.run_id == run_id)
        .order_by(AgentTraceStep.id.asc())
        .all()
    )

    assert [step.step_name for step in steps] == [
        "retrieve_context",
        "analyze_resume",
        "analyze_jd",
        "analyze_fit_gap",
        "decide_route",
        "compose_final",
        "review_output",
    ]
    assert all(step.status == "completed" for step in steps)


def test_trace_endpoint_returns_steps_in_order(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    run_id = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD).json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}/trace").json()

    assert body["run_id"] == run_id
    assert [step["step_name"] for step in body["steps"]] == [
        "retrieve_context",
        "analyze_resume",
        "analyze_jd",
        "analyze_fit_gap",
        "decide_route",
        "compose_final",
        "review_output",
    ]
    assert [step["id"] for step in body["steps"]] == sorted(step["id"] for step in body["steps"])


def test_trace_endpoint_returns_empty_list_for_single_step_runs(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "single_step")

    run_id = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD).json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}/trace").json()

    assert body == {"run_id": run_id, "steps": []}


def test_trace_endpoint_returns_404_for_missing_run():
    response = client.get("/api/v1/applications/runs/999999/trace")
    assert response.status_code == 404
    assert response.json()["detail"] == "Run 999999 not found"


def test_trace_steps_do_not_include_raw_resume_text(db_session, monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    run_id = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD).json()["run_id"]
    body = client.get(f"/api/v1/applications/runs/{run_id}/trace").json()
    serialized = json.dumps(body)

    assert VALID_PAYLOAD["master_resume"] not in serialized
    assert VALID_PAYLOAD["job_description"] not in serialized


def test_revise_output_trace_appears_only_when_revision_runs(db_session, monkeypatch):
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    def patched_compose_final(state):
        return {
            "final_output": TailoringLLMOutput(
                tailored_summary="Draft summary.",
                tailored_bullets=[],
                cover_letter_draft="Draft cover letter.",
                application_question_answers=["Draft answer."],
                recruiter_message_draft="Draft recruiter message.",
                fit_gap_analysis="Draft fit/gap analysis.",
                interview_talking_points=["Draft talking point."],
            ),
            "provider_used": "mock",
            "fallback_used": False,
        }

    request = ApplicationTailorRequest(**VALID_PAYLOAD)
    run = create_pending_run(db_session, request)

    monkeypatch.setattr("app.services.agentic_tailoring._compose_final", patched_compose_final)
    output, _, _ = run_agentic_workflow(request, db=db_session, run_id=run.id)

    assert "[MOCK-REVISED]" in output.tailored_summary

    steps = (
        db_session.query(AgentTraceStep)
        .filter(AgentTraceStep.run_id == run.id)
        .order_by(AgentTraceStep.id.asc())
        .all()
    )
    assert steps[-1].step_name == "revise_output"
    assert steps[-2].step_name == "review_output"


def test_trace_write_failure_does_not_fail_agentic_workflow(db_session, monkeypatch):
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    request = ApplicationTailorRequest(**VALID_PAYLOAD)
    run = create_pending_run(db_session, request)

    def fail_trace_write(**kwargs):
        raise RuntimeError("trace insert failed")

    monkeypatch.setattr("app.services.agentic_tailoring.create_agent_trace_step", fail_trace_write)

    output, provider_used, fallback_used = run_agentic_workflow(
        request,
        db=db_session,
        run_id=run.id,
    )

    assert output.tailored_summary
    assert provider_used == "mock"
    assert fallback_used is False
    assert db_session.query(AgentTraceStep).filter(AgentTraceStep.run_id == run.id).count() == 0
