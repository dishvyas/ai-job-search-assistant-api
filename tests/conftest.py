import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app


@pytest.fixture(autouse=True)
def force_mock_llm_provider(monkeypatch):
    """
    Override LLM provider to mock for every test, regardless of what is set
    in .env. This ensures tests are never coupled to a real API key or the
    local developer's environment.
    """
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "mock")


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
