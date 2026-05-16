from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.llm.exceptions import LLMProviderError, LLMProviderUnavailableError
from app.llm.gemini import GeminiLLMProvider
from app.llm.mock import MockLLMProvider
from app.main import app
from app.schemas.llm_output import TailoringLLMOutput
from app.services.application_tailoring import _get_llm_output

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_unavailable_error_is_subclass_of_provider_error():
    assert issubclass(LLMProviderUnavailableError, LLMProviderError)


# ---------------------------------------------------------------------------
# Gemini provider — error wrapping
# ---------------------------------------------------------------------------


def test_gemini_raises_value_error_when_api_key_missing():
    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiLLMProvider(api_key="")


def test_gemini_wraps_runtime_failure_as_unavailable_error():
    """SDK exceptions during generate_content must become LLMProviderUnavailableError."""
    provider = GeminiLLMProvider.__new__(GeminiLLMProvider)
    provider.model = "gemini-2.5-flash"

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE")
    provider._client = mock_client

    with pytest.raises(LLMProviderUnavailableError, match="Gemini request failed"):
        provider.generate_text("test prompt")


def test_gemini_does_not_wrap_import_error():
    """ImportError at init time should pass through unchanged, not become LLMProviderError."""
    with patch.dict("sys.modules", {"google": None, "google.genai": None}):
        with pytest.raises((ImportError, TypeError)):
            GeminiLLMProvider(api_key="fake-key")


# ---------------------------------------------------------------------------
# _get_llm_output helper
# ---------------------------------------------------------------------------


def test_get_llm_output_returns_parsed_output_on_success(monkeypatch):
    """When provider succeeds and returns valid JSON, used_fallback must be False."""
    mock_provider = MockLLMProvider()
    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: mock_provider
    )

    output, provider_used, used_fallback = _get_llm_output("some prompt")

    assert used_fallback is False
    assert provider_used == "mock"
    assert isinstance(output, TailoringLLMOutput)


def test_get_llm_output_falls_back_on_provider_error(monkeypatch):
    """When provider raises LLMProviderError, used_fallback must be True."""

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503 UNAVAILABLE")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    output, provider_used, used_fallback = _get_llm_output("some prompt")

    assert used_fallback is True
    assert provider_used == "fallback-mock"
    assert isinstance(output, TailoringLLMOutput)


def test_get_llm_output_does_not_catch_non_llm_errors(monkeypatch):
    """Programming errors (e.g. TypeError) must not be swallowed by the fallback."""

    class BrokenProvider:
        def generate_text(self, prompt: str) -> str:
            raise TypeError("This is a bug, not a provider error")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: BrokenProvider()
    )

    with pytest.raises(TypeError):
        _get_llm_output("some prompt")


# ---------------------------------------------------------------------------
# Endpoint integration — fallback path
# ---------------------------------------------------------------------------


def test_endpoint_returns_200_when_provider_fails(monkeypatch):
    """
    POST must return 200 (job accepted) even when the configured provider is
    unavailable. The background task handles fallback; the route always accepts.
    """

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503 UNAVAILABLE")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_endpoint_fallback_response_notes_fallback_mode(db_session, monkeypatch):
    """
    When the configured provider fails, the background task falls back to mock.
    The completed run's tailored_summary and fit_gap_analysis must note fallback.
    """

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503 UNAVAILABLE")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert "Fallback mode used" in body["fit_gap_analysis"]
    assert "Fallback mode used" in body["tailored_summary"]


def test_endpoint_normal_mode_does_not_mention_fallback(db_session):
    """In normal mock mode, completed run output must not mention fallback."""
    post_response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    run_id = post_response.json()["run_id"]

    body = client.get(f"/api/v1/applications/runs/{run_id}").json()
    assert "Fallback mode used" not in body["tailored_summary"]
    assert "Fallback mode used" not in body["fit_gap_analysis"]
