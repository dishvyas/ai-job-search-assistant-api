import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(autouse=True)
def force_mock_llm_provider(monkeypatch):
    """
    Override LLM provider to mock for every test, regardless of what is set
    in .env. This ensures tests are never coupled to a real API key or the
    local developer's environment.

    Also force rag_enabled=False so background-job tests are not affected by a
    local .env that sets RAG_ENABLED=true. Individual tests that exercise RAG
    behaviour re-patch this to True explicitly.
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("WORKFLOW_MODE", "single_step")
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("ARTIFACT_RETRIEVAL_ENABLED", "false")
    monkeypatch.setattr("app.core.config.settings.llm_provider", "mock")
    monkeypatch.setattr("app.core.config.settings.workflow_mode", "single_step")
    monkeypatch.setattr("app.core.config.settings.rag_enabled", False)
    monkeypatch.setattr("app.core.config.settings.artifact_retrieval_enabled", False)
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "mock")
    # Disable RAG globally — individual RAG tests re-enable it via monkeypatch.
    # Without this, any local .env with RAG_ENABLED=true causes background-job
    # tests to attempt a real OpenAI embedding call and fail.
    monkeypatch.setattr("app.services.background_tailoring.settings.workflow_mode", "single_step")
    monkeypatch.setattr("app.services.background_tailoring.settings.rag_enabled", False)
    monkeypatch.setattr(
        "app.services.background_tailoring.settings.artifact_retrieval_enabled", False
    )
    monkeypatch.setattr("app.services.agentic_tailoring.settings.workflow_mode", "single_step")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.agentic_tailoring.settings.rag_enabled", False)
    monkeypatch.setattr("app.api.v1.routes.jobs.settings.rag_enabled", False)


@pytest.fixture
def db_session():
    """
    Provide a fresh in-memory SQLite session for each test.

    StaticPool ensures all SQLAlchemy connections share the same underlying
    connection, which is required for in-memory SQLite to work correctly
    across a single test (the schema and data would otherwise be invisible
    across different connections to the same :memory: DB).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def override_get_db(db_session):
    """
    Override FastAPI's get_db dependency for every test to use the
    in-memory SQLite session. This means all tests that hit the API
    automatically persist to (and read from) the test DB.
    """

    def _test_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _test_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
