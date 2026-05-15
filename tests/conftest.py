import pytest


@pytest.fixture(autouse=True)
def force_mock_llm_provider(monkeypatch):
    """
    Override LLM provider to mock for every test, regardless of what is set
    in .env. This ensures tests are never coupled to a real API key or the
    local developer's environment.
    """
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "mock")
