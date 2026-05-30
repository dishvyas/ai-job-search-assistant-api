import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.run_status import RunStatus
from app.rag.exceptions import RAGEmbeddingError
from app.rag.ingest import ingest_job_description

client = TestClient(app)

MOCK_VECTOR = [0.1] * 1536


def test_generate_embedding_wraps_provider_errors(monkeypatch):
    class FakeEmbeddingsClient:
        def create(self, model, input):  # noqa: A002
            raise RuntimeError("429 insufficient_quota sk-test-secret")

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.embeddings = FakeEmbeddingsClient()

    fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    from app.rag.embed import generate_embedding

    try:
        generate_embedding("FastAPI backend engineer")
    except RAGEmbeddingError as exc:
        message = str(exc)
        assert "OPENAI_API_KEY" in message
        assert "EMBEDDING_MODEL" in message
        assert "sk-test-secret" not in message
    else:
        raise AssertionError("Expected RAGEmbeddingError to be raised")


def test_jobs_ingest_returns_503_on_embedding_failure(monkeypatch):
    monkeypatch.setattr("app.api.v1.routes.jobs.settings.rag_enabled", True)
    monkeypatch.setattr(
        "app.api.v1.routes.jobs.ingest_job_description",
        lambda **kwargs: (_ for _ in ()).throw(
            RAGEmbeddingError("429 insufficient_quota sk-secret should not leak")
        ),
    )

    response = client.post(
        "/api/v1/jobs/ingest",
        json={"title": "Engineer", "raw_text": "Python backend role"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["detail"].startswith("Embedding provider failed.")
    assert "OPENAI_API_KEY" in body["detail"]
    assert "sk-secret" not in body["detail"]
    assert "Traceback" not in body["detail"]


def test_ingest_job_description_still_succeeds_with_mocked_embedding(db_session, monkeypatch):
    monkeypatch.setattr("app.rag.ingest.generate_embedding", lambda text: MOCK_VECTOR)

    jd = ingest_job_description(
        db=db_session,
        title="Backend Engineer",
        raw_text="Python FastAPI APIs",
        company="Acme",
    )

    assert jd.id is not None
    assert jd.title == "Backend Engineer"
    assert len(jd.embedding) == len(MOCK_VECTOR)
    assert float(jd.embedding[0]) == pytest.approx(MOCK_VECTOR[0])


def test_background_tailoring_gracefully_continues_when_retrieval_fails(monkeypatch):
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "single_step")
    monkeypatch.setattr("app.services.background_tailoring.settings.rag_enabled", True)
    monkeypatch.setattr(
        "app.services.background_tailoring.settings.artifact_retrieval_enabled", False
    )
    monkeypatch.setattr(
        "app.rag.retrieve.retrieve_relevant_jobs",
        lambda db, query, top_k=None, filters=None: (_ for _ in ()).throw(
            RAGEmbeddingError("embedding quota exceeded")
        ),
    )

    response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Backend engineer with Python experience.",
            "job_description": "FastAPI backend role.",
        },
    )
    run_id = response.json()["run_id"]
    run_body = client.get(f"/api/v1/applications/runs/{run_id}").json()

    assert run_body["status"] == RunStatus.COMPLETED.value
    assert run_body["provider_used"] == "mock"
