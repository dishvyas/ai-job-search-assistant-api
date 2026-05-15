from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.llm.exceptions import LLMProviderError, LLMProviderUnavailableError
from app.llm.gemini import GeminiLLMProvider
from app.llm.mock import MockLLMProvider
from app.main import app
from app.services.application_tailoring import _generate_with_fallback

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
# Fallback helper
# ---------------------------------------------------------------------------


def test_fallback_returns_used_fallback_false_on_success(monkeypatch):
    """When provider succeeds, used_fallback must be False."""
    mock_provider = MockLLMProvider()
    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: mock_provider
    )

    output, used_fallback = _generate_with_fallback("some prompt")

    assert used_fallback is False
    assert isinstance(output, str)


def test_fallback_returns_used_fallback_true_on_llm_error(monkeypatch):
    """When provider raises LLMProviderError, used_fallback must be True."""

    def failing_provider():
        raise LLMProviderUnavailableError("503 from provider")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider",
        lambda: (_ for _ in ()).throw(LLMProviderUnavailableError("503")),
    )

    # Simpler: patch get_llm_provider to return a provider whose generate_text raises
    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503 UNAVAILABLE")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    output, used_fallback = _generate_with_fallback("some prompt")

    assert used_fallback is True
    assert isinstance(output, str)
    assert len(output) > 0


def test_fallback_does_not_catch_non_llm_errors(monkeypatch):
    """Programming errors (e.g. TypeError) must not be swallowed by the fallback."""

    class BrokenProvider:
        def generate_text(self, prompt: str) -> str:
            raise TypeError("This is a bug, not a provider error")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: BrokenProvider()
    )

    with pytest.raises(TypeError):
        _generate_with_fallback("some prompt")


# ---------------------------------------------------------------------------
# Endpoint integration — fallback path
# ---------------------------------------------------------------------------


def test_endpoint_returns_200_when_provider_fails(monkeypatch):
    """Endpoint must return 200 even when the configured provider is unavailable."""

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503 UNAVAILABLE")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_endpoint_fallback_response_notes_fallback_mode(monkeypatch):
    """fit_gap_analysis should say 'Fallback mode used' when fallback occurred."""

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503 UNAVAILABLE")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider", lambda: FailingProvider()
    )

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    assert "Fallback mode used" in body["fit_gap_analysis"]
    assert "Fallback mode used" in body["tailored_summary"]


def test_endpoint_normal_mode_does_not_mention_fallback():
    """In normal mock mode, response must not mention fallback."""
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    assert "Fallback mode used" not in body["tailored_summary"]
    assert "Fallback mode used" not in body["fit_gap_analysis"]
