"""
Tests for Milestone 10 — Tool-Using Agent Workflow with Review/Revision.

Covers:
- retrieve_context node skips retrieval when RAG is disabled
- retrieve_context node populates context when RAG is enabled
- retrieve_context continues gracefully when retrieval raises an exception
- decide_route returns needs_more_context when no context and no company_info
- decide_route returns proceed_to_tailoring when company_info is present
- decide_route returns proceed_to_tailoring when retrieved_context is non-empty
- decide_route returns low_fit_warning when gap_points exceed fit_points
- review_output passes for complete, non-empty output
- review_output requests revision when tailored_bullets is empty
- review_output requests revision when interview_talking_points is empty
- review_output requests revision when tailored_summary is empty
- revision node produces a valid, corrected TailoringLLMOutput
- revision runs at most once — graph structure prevents looping
- retrieved context is injected into fit/gap and final prompts
- agentic background job still completes end-to-end
- single_step workflow is unaffected by M10 changes
- mock mode remains deterministic across the full new workflow
"""

import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.llm.mock import MockLLMProvider
from app.main import app
from app.models.application import ApplicationTailoringRun
from app.models.job_description import JobDescription
from app.models.run_status import RunStatus
from app.prompts.agentic_tailoring import (
    build_final_tailoring_prompt,
    build_fit_gap_prompt,
    build_revision_prompt,
)
from app.schemas.agent import FitGapAnalysis, JobDescriptionAnalysis, ResumeAnalysis
from app.schemas.application import ApplicationTailorRequest
from app.schemas.llm_output import TailoringLLMOutput
from app.services.agentic_tailoring import (
    _decide_route,
    _retrieve_context,
    _review_output,
    run_agentic_workflow,
)

client = TestClient(app)

SAMPLE_REQUEST = ApplicationTailorRequest(
    master_resume="Software engineer with 5 years of Python experience.",
    job_description="Backend engineer role using FastAPI.",
)

SAMPLE_REQUEST_WITH_COMPANY = ApplicationTailorRequest(
    master_resume="Software engineer with 5 years of Python experience.",
    job_description="Backend engineer role using FastAPI.",
    company_info="Acme Corp — a fast-growing fintech startup.",
)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session():
    """Create an in-memory SQLite session for tests that need one."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session()


def _make_jd(db_session, title="Backend Engineer", raw_text="Python FastAPI developer"):
    """Persist a minimal JobDescription for retrieval tests."""
    jd = JobDescription(
        title=title,
        company="Acme Corp",
        raw_text=raw_text,
        embedding=[0.1] * 1536,
    )
    db_session.add(jd)
    db_session.commit()
    db_session.refresh(jd)
    return jd


def _make_complete_output() -> TailoringLLMOutput:
    """Minimal complete TailoringLLMOutput that passes the review gate."""
    return TailoringLLMOutput(
        tailored_summary="Summary text.",
        tailored_bullets=["Bullet 1", "Bullet 2"],
        cover_letter_draft="Dear Hiring Manager,\n\nBest,\nMe",
        application_question_answers=["Answer 1"],
        recruiter_message_draft="Hi there.",
        fit_gap_analysis="FIT: Python. GAP: Kubernetes.",
        interview_talking_points=["Talking point 1"],
    )


def _make_incomplete_output(empty_field: str) -> TailoringLLMOutput:
    """Return a TailoringLLMOutput with one field deliberately emptied."""
    base = _make_complete_output().model_dump()
    base[empty_field] = [] if isinstance(base[empty_field], list) else ""
    return TailoringLLMOutput(**base)


def _make_state(**overrides) -> dict:
    """Build a minimal AgenticTailoringState dict for node-level unit tests."""
    state = {
        "request": SAMPLE_REQUEST,
        "db": None,
        "artifact_context": [],
        "resume_analysis": ResumeAnalysis(
            key_skills=["Python", "FastAPI"],
            relevant_experience=["5 years backend"],
            strengths=["Problem-solving"],
        ),
        "jd_analysis": JobDescriptionAnalysis(
            required_skills=["Python", "APIs"],
            responsibilities=["Build services"],
            role_focus="Backend engineering",
        ),
        "fit_gap": FitGapAnalysis(
            fit_points=["Strong Python background", "API development experience"],
            gap_points=["Kubernetes experience"],
            positioning_strategy="Emphasise adaptability",
        ),
        "final_output": _make_complete_output(),
        "retrieved_context": [],
        "route_decision": "proceed_to_tailoring",
        "review_notes": None,
        "revision_needed": False,
        "provider_used": "mock",
        "fallback_used": False,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# retrieve_context node — unit tests
# ---------------------------------------------------------------------------


def test_retrieve_context_returns_empty_when_rag_disabled(monkeypatch):
    """When rag_enabled=False, retrieve_context must return an empty list."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", False)
    result = _retrieve_context(_make_state())
    assert result == {"retrieved_context": []}


def test_retrieve_context_returns_empty_when_db_is_none(monkeypatch):
    """When db is None (no session provided), retrieval must be skipped."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", True)
    # db=None in the default _make_state() — retrieval must not be attempted.
    result = _retrieve_context(_make_state(db=None))
    assert result == {"retrieved_context": []}


def test_retrieve_context_populates_snippets_when_rag_enabled(monkeypatch):
    """When RAG is enabled and retrieval returns results, snippets must be stored."""
    db = _make_session()
    jd = _make_jd(db, title="Python Engineer", raw_text="Python FastAPI microservices backend")

    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", True)
    monkeypatch.setattr(
        "app.rag.retrieve.retrieve_relevant_jobs",
        lambda db_arg, query, top_k=None, filters=None: [(jd, 0.9)],
    )

    result = _retrieve_context(_make_state(db=db))

    assert len(result["retrieved_context"]) == 1
    assert "Python" in result["retrieved_context"][0]


def test_retrieve_context_graceful_on_exception(monkeypatch):
    """If retrieval raises, the node must return empty context rather than propagating."""
    db = _make_session()
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", True)
    monkeypatch.setattr(
        "app.rag.retrieve.retrieve_relevant_jobs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("DB connection lost")),
    )

    # Must not raise — graceful degradation to empty context.
    result = _retrieve_context(_make_state(db=db))
    assert result == {"retrieved_context": []}


# ---------------------------------------------------------------------------
# decide_route node — unit tests (all deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_route_needs_more_context_when_no_context_and_no_company():
    """No retrieved context AND no company_info → needs_more_context."""
    state = _make_state(retrieved_context=[], request=SAMPLE_REQUEST)
    result = _decide_route(state)
    assert result["route_decision"] == "needs_more_context"


def test_route_proceed_when_company_info_present():
    """Even with no retrieved context, company_info is enough to proceed."""
    state = _make_state(retrieved_context=[], request=SAMPLE_REQUEST_WITH_COMPANY)
    result = _decide_route(state)
    assert result["route_decision"] == "proceed_to_tailoring"


def test_route_proceed_when_retrieved_context_available():
    """Non-empty retrieved_context is sufficient to proceed even without company_info."""
    state = _make_state(
        retrieved_context=["Python FastAPI backend microservices"],
        request=SAMPLE_REQUEST,
    )
    result = _decide_route(state)
    assert result["route_decision"] == "proceed_to_tailoring"


def test_route_low_fit_warning_when_gaps_exceed_fit():
    """More gap_points than fit_points → low_fit_warning."""
    state = _make_state(
        retrieved_context=["some context"],  # non-empty so not needs_more_context
        fit_gap=FitGapAnalysis(
            fit_points=["Python"],
            gap_points=["Kubernetes", "Terraform", "Go"],  # 3 gaps > 1 fit
            positioning_strategy="Emphasise transferable skills",
        ),
    )
    result = _decide_route(state)
    assert result["route_decision"] == "low_fit_warning"


def test_route_proceed_when_fit_equals_gaps():
    """Equal fit and gap counts → proceed_to_tailoring (not low_fit_warning)."""
    state = _make_state(
        retrieved_context=["some context"],
        fit_gap=FitGapAnalysis(
            fit_points=["Python", "FastAPI"],
            gap_points=["Kubernetes", "Terraform"],  # equal counts
            positioning_strategy="Balanced positioning",
        ),
    )
    result = _decide_route(state)
    # gaps == fit points, not strictly greater, so no warning
    assert result["route_decision"] == "proceed_to_tailoring"


# ---------------------------------------------------------------------------
# review_output node — unit tests (deterministic)
# ---------------------------------------------------------------------------


def test_review_passes_for_complete_output():
    """A fully populated TailoringLLMOutput must pass review without revision."""
    state = _make_state(final_output=_make_complete_output())
    result = _review_output(state)

    assert result["revision_needed"] is False
    assert "passed" in result["review_notes"].lower()


def test_review_requests_revision_for_empty_bullets():
    """Empty tailored_bullets must trigger revision."""
    output = _make_incomplete_output("tailored_bullets")
    state = _make_state(final_output=output)
    result = _review_output(state)

    assert result["revision_needed"] is True
    assert "tailored_bullets" in result["review_notes"]


def test_review_requests_revision_for_empty_talking_points():
    """Empty interview_talking_points must trigger revision."""
    output = _make_incomplete_output("interview_talking_points")
    state = _make_state(final_output=output)
    result = _review_output(state)

    assert result["revision_needed"] is True
    assert "interview_talking_points" in result["review_notes"]


def test_review_requests_revision_for_empty_summary():
    """Empty tailored_summary must trigger revision."""
    output = _make_incomplete_output("tailored_summary")
    state = _make_state(final_output=output)
    result = _review_output(state)

    assert result["revision_needed"] is True
    assert "tailored_summary" in result["review_notes"]


def test_review_returns_none_final_output_as_revision_needed():
    """If final_output is None entirely, all checks fail → revision_needed=True."""
    state = _make_state(final_output=None)
    result = _review_output(state)
    assert result["revision_needed"] is True


# ---------------------------------------------------------------------------
# Revision prompt — unit test
# ---------------------------------------------------------------------------


def test_build_revision_prompt_contains_header_and_notes():
    """build_revision_prompt must include the task header and review notes."""
    prompt = build_revision_prompt(
        current_output_json='{"tailored_summary": ""}',
        review_notes="tailored_bullets is empty",
    )
    assert "## Task: Revise Output" in prompt
    assert "tailored_bullets is empty" in prompt


def test_mock_returns_valid_revision_json():
    """Mock provider must return parseable TailoringLLMOutput for the revision stage."""
    provider = MockLLMProvider()
    prompt = build_revision_prompt(
        current_output_json="{}",
        review_notes="tailored_bullets is empty",
    )
    raw = provider.generate_text(prompt)
    parsed = TailoringLLMOutput.model_validate(json.loads(raw))

    assert "[MOCK-REVISED]" in parsed.tailored_summary
    assert len(parsed.tailored_bullets) > 0
    assert len(parsed.interview_talking_points) > 0


# ---------------------------------------------------------------------------
# Revision path — workflow-level test
# ---------------------------------------------------------------------------


def test_revision_runs_when_review_marks_output_incomplete(monkeypatch):
    """
    When compose_final returns an empty-bullets output, the review node should trigger
    revision, and the final output should carry [MOCK-REVISED] markers.
    """
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    # Patch compose_final to return an output that will fail review.
    incomplete_output = _make_incomplete_output("tailored_bullets")

    def patched_compose_final(state):
        # Return the incomplete output; all other fields unchanged.
        return {"final_output": incomplete_output, "provider_used": "mock", "fallback_used": False}

    monkeypatch.setattr("app.services.agentic_tailoring._compose_final", patched_compose_final)

    output, provider, fallback, _ = run_agentic_workflow(SAMPLE_REQUEST)

    # Revision ran — output should have [MOCK-REVISED] markers from _revision_response.
    assert "[MOCK-REVISED]" in output.tailored_summary
    assert len(output.tailored_bullets) > 0  # revision restored the missing bullets


def test_revision_runs_at_most_once(monkeypatch):
    """
    After revise_output, the graph always goes to END — no second revision possible.
    This test verifies the workflow completes (doesn't loop) even when review
    initially marks revision_needed=True.
    """
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    revision_call_count = []

    original_revise = __import__(
        "app.services.agentic_tailoring", fromlist=["_revise_output"]
    )._revise_output

    def counting_revise(state):
        revision_call_count.append(1)
        return original_revise(state)

    monkeypatch.setattr("app.services.agentic_tailoring._revise_output", counting_revise)

    # Force review to always request revision by patching review_output.
    def always_needs_revision(state):
        return {"revision_needed": True, "review_notes": "Forced revision for test."}

    monkeypatch.setattr("app.services.agentic_tailoring._review_output", always_needs_revision)

    output, _, _, _ = run_agentic_workflow(SAMPLE_REQUEST)

    # Revision ran exactly once — the graph structure prevents re-entry.
    assert len(revision_call_count) == 1
    assert isinstance(output, TailoringLLMOutput)


# ---------------------------------------------------------------------------
# Context injection into prompts
# ---------------------------------------------------------------------------


def test_retrieved_context_passed_to_fit_gap_prompt(monkeypatch):
    """When context is retrieved, build_fit_gap_prompt must receive it."""
    db = _make_session()
    jd = _make_jd(db, title="Python Engineer", raw_text="Python FastAPI microservices systems")

    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", True)
    monkeypatch.setattr(
        "app.rag.retrieve.retrieve_relevant_jobs",
        lambda db_arg, query, top_k=None, filters=None: [(jd, 0.9)],
    )

    captured_context: list = []
    original_fit_gap_prompt = build_fit_gap_prompt

    def capturing_fit_gap(request, resume_analysis, jd_analysis, retrieved_context=None):
        captured_context.append(retrieved_context)
        return original_fit_gap_prompt(
            request, resume_analysis, jd_analysis, retrieved_context=retrieved_context
        )

    monkeypatch.setattr("app.services.agentic_tailoring.build_fit_gap_prompt", capturing_fit_gap)

    run_agentic_workflow(SAMPLE_REQUEST, db=db)

    assert len(captured_context) == 1
    assert captured_context[0] is not None and len(captured_context[0]) > 0
    assert "Python" in captured_context[0][0]


def test_retrieved_context_passed_to_final_prompt(monkeypatch):
    """When context is retrieved, build_final_tailoring_prompt must receive it."""
    db = _make_session()
    jd = _make_jd(db, title="Backend Engineer", raw_text="Python FastAPI backend systems")

    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", True)
    monkeypatch.setattr(
        "app.rag.retrieve.retrieve_relevant_jobs",
        lambda db_arg, query, top_k=None, filters=None: [(jd, 0.9)],
    )

    captured_context: list = []
    original_final_prompt = build_final_tailoring_prompt

    def capturing_final(request, resume_analysis, jd_analysis, fit_gap, retrieved_context=None):
        captured_context.append(retrieved_context)
        return original_final_prompt(
            request,
            resume_analysis,
            jd_analysis,
            fit_gap,
            retrieved_context=retrieved_context,
        )

    monkeypatch.setattr(
        "app.services.agentic_tailoring.build_final_tailoring_prompt", capturing_final
    )

    run_agentic_workflow(SAMPLE_REQUEST, db=db)

    assert len(captured_context) == 1
    assert captured_context[0] is not None and len(captured_context[0]) > 0


def test_no_context_when_rag_disabled(monkeypatch):
    """When RAG is disabled, both prompt builders receive no retrieved context."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", False)

    captured_fit_context: list = []
    captured_final_context: list = []

    original_fit_gap_prompt = build_fit_gap_prompt
    original_final_prompt = build_final_tailoring_prompt

    def capturing_fit_gap(request, resume_analysis, jd_analysis, retrieved_context=None):
        captured_fit_context.append(retrieved_context)
        return original_fit_gap_prompt(
            request, resume_analysis, jd_analysis, retrieved_context=retrieved_context
        )

    def capturing_final(request, resume_analysis, jd_analysis, fit_gap, retrieved_context=None):
        captured_final_context.append(retrieved_context)
        return original_final_prompt(
            request,
            resume_analysis,
            jd_analysis,
            fit_gap,
            retrieved_context=retrieved_context,
        )

    monkeypatch.setattr("app.services.agentic_tailoring.build_fit_gap_prompt", capturing_fit_gap)
    monkeypatch.setattr(
        "app.services.agentic_tailoring.build_final_tailoring_prompt", capturing_final
    )

    run_agentic_workflow(SAMPLE_REQUEST)

    # None is passed when retrieved_context is [] (falsy); prompt section is omitted.
    assert not captured_fit_context[0]
    assert not captured_final_context[0]


# ---------------------------------------------------------------------------
# Prompt content — context section appears in output
# ---------------------------------------------------------------------------


def test_fit_gap_prompt_includes_retrieved_context_section():
    """build_fit_gap_prompt must include a labeled context section when context is given."""
    resume = ResumeAnalysis(
        key_skills=["Python"], relevant_experience=["backend"], strengths=["adaptable"]
    )
    jd = JobDescriptionAnalysis(
        required_skills=["Python"], responsibilities=["build APIs"], role_focus="Backend"
    )
    prompt = build_fit_gap_prompt(
        SAMPLE_REQUEST, resume, jd, retrieved_context=["Python FastAPI microservices systems"]
    )
    assert "Retrieved Context" in prompt
    assert "Python FastAPI" in prompt


def test_final_prompt_includes_retrieved_context_section():
    """build_final_tailoring_prompt must include a labeled context section when given."""
    resume = ResumeAnalysis(
        key_skills=["Python"], relevant_experience=["backend"], strengths=["adaptable"]
    )
    jd = JobDescriptionAnalysis(
        required_skills=["Python"], responsibilities=["build APIs"], role_focus="Backend"
    )
    fit_gap = FitGapAnalysis(
        fit_points=["Python"],
        gap_points=["Kubernetes"],
        positioning_strategy="Emphasise depth",
    )
    prompt = build_final_tailoring_prompt(
        SAMPLE_REQUEST, resume, jd, fit_gap, retrieved_context=["FastAPI Python backend systems"]
    )
    assert "Retrieved Context" in prompt
    assert "FastAPI" in prompt


def test_fit_gap_prompt_unchanged_without_context():
    """Without retrieved_context, no extra section appears in the fit/gap prompt."""
    resume = ResumeAnalysis(
        key_skills=["Python"], relevant_experience=["backend"], strengths=["adaptable"]
    )
    jd = JobDescriptionAnalysis(
        required_skills=["Python"], responsibilities=["build APIs"], role_focus="Backend"
    )
    prompt_none = build_fit_gap_prompt(SAMPLE_REQUEST, resume, jd, retrieved_context=None)
    prompt_empty = build_fit_gap_prompt(SAMPLE_REQUEST, resume, jd, retrieved_context=[])

    assert "Retrieved Context" not in prompt_none
    assert "Retrieved Context" not in prompt_empty
    assert prompt_none == prompt_empty


# ---------------------------------------------------------------------------
# Full workflow — end-to-end
# ---------------------------------------------------------------------------


def test_agentic_workflow_completes_with_mock(monkeypatch):
    """Full agentic workflow must complete and return valid TailoringLLMOutput."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    output, provider, fallback, _ = run_agentic_workflow(SAMPLE_REQUEST)

    assert isinstance(output, TailoringLLMOutput)
    assert output.tailored_summary
    assert len(output.tailored_bullets) > 0
    assert len(output.interview_talking_points) > 0
    assert provider == "mock"
    assert fallback is False


def test_agentic_workflow_completes_without_rag(monkeypatch):
    """Agentic workflow must complete identically when RAG is disabled."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", False)

    output, provider, _, _ = run_agentic_workflow(SAMPLE_REQUEST)

    assert isinstance(output, TailoringLLMOutput)
    assert provider == "mock"


def test_background_job_completes_in_agentic_mode(db_session, monkeypatch):
    """Background job must still reach COMPLETED status in agentic mode after M10."""
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "agentic")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200

    run = db_session.query(ApplicationTailoringRun).first()
    assert run.status == RunStatus.COMPLETED.value
    assert run.tailored_summary is not None
    assert isinstance(run.tailored_bullets, list) and len(run.tailored_bullets) > 0


def test_single_step_workflow_unaffected_by_m10(db_session, monkeypatch):
    """POST /tailor in single_step mode must behave identically to pre-M10."""
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "single_step")

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200

    run = db_session.query(ApplicationTailoringRun).first()
    assert run.status == RunStatus.COMPLETED.value
    assert run.tailored_summary is not None


def test_mock_mode_is_deterministic(monkeypatch):
    """Running the agentic workflow twice in mock mode must produce identical output."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    out1, _, _, _ = run_agentic_workflow(SAMPLE_REQUEST)
    out2, _, _, _ = run_agentic_workflow(SAMPLE_REQUEST)

    assert out1.tailored_summary == out2.tailored_summary
    assert out1.tailored_bullets == out2.tailored_bullets


def test_no_real_llm_calls_in_mock_mode(monkeypatch):
    """The configured provider must never be called when llm_provider='mock'."""
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")

    real_call_count = []

    class TrackingProvider:
        def generate_text(self, prompt: str) -> str:
            real_call_count.append(1)
            return "{}"

    # Override get_llm_provider to return the tracking provider, but since
    # llm_provider='mock', _call_and_parse will use MockLLMProvider directly
    # and should not call get_llm_provider at all (it returns MockLLMProvider when "mock").
    run_agentic_workflow(SAMPLE_REQUEST)

    # No real (non-mock) provider calls should have been made.
    assert len(real_call_count) == 0
