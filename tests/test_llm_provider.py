import pytest
from fastapi.testclient import TestClient

from app.llm.factory import get_llm_provider
from app.llm.mock import MockLLMProvider
from app.main import app
from app.prompts.tailoring import build_tailoring_prompt
from app.schemas.application import ApplicationTailorRequest

client = TestClient(app)

SAMPLE_REQUEST = ApplicationTailorRequest(
    master_resume="Software engineer with 5 years of Python experience.",
    job_description="Looking for a backend engineer to build scalable APIs.",
)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_factory_returns_mock_provider_by_default():
    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_factory_returns_mock_provider_when_configured(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "mock")
    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_factory_raises_for_unsupported_provider(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "openai")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_provider()


# ---------------------------------------------------------------------------
# Mock provider tests
# ---------------------------------------------------------------------------


def test_mock_provider_generate_text_returns_string():
    provider = MockLLMProvider()
    result = provider.generate_text("Hello, world!")
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_provider_includes_prompt_preview():
    provider = MockLLMProvider()
    result = provider.generate_text("Test prompt content here")
    # Mock embeds a prompt preview inside the JSON tailored_summary field
    assert "Test prompt content here" in result


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


def test_prompt_includes_resume():
    prompt = build_tailoring_prompt(SAMPLE_REQUEST)
    assert SAMPLE_REQUEST.master_resume in prompt


def test_prompt_includes_job_description():
    prompt = build_tailoring_prompt(SAMPLE_REQUEST)
    assert SAMPLE_REQUEST.job_description in prompt


def test_prompt_includes_company_info_when_provided():
    request = ApplicationTailorRequest(
        master_resume="Resume text",
        job_description="JD text",
        company_info="A fast-growing fintech startup.",
    )
    prompt = build_tailoring_prompt(request)
    assert "A fast-growing fintech startup." in prompt


def test_prompt_excludes_company_info_when_absent():
    prompt = build_tailoring_prompt(SAMPLE_REQUEST)
    assert "## Company Info" not in prompt


def test_prompt_includes_user_preferences_when_provided():
    request = ApplicationTailorRequest(
        master_resume="Resume text",
        job_description="JD text",
        user_preferences="Emphasize system design experience.",
    )
    prompt = build_tailoring_prompt(request)
    assert "Emphasize system design experience." in prompt


def test_prompt_excludes_user_preferences_when_absent():
    prompt = build_tailoring_prompt(SAMPLE_REQUEST)
    assert "## User Preferences" not in prompt


# ---------------------------------------------------------------------------
# Endpoint integration tests (mock mode)
# ---------------------------------------------------------------------------


def test_tailor_endpoint_returns_200_in_mock_mode():
    response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Software engineer with 5 years of Python experience.",
            "job_description": "Backend engineer role using FastAPI.",
        },
    )
    assert response.status_code == 200


def test_tailor_response_contains_llm_provider_output():
    """The tailored_summary should contain evidence that the LLM provider was called."""
    response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Software engineer with 5 years of Python experience.",
            "job_description": "Backend engineer role using FastAPI.",
        },
    )
    summary = response.json()["tailored_summary"]
    # Mock provider now returns structured JSON; the summary field carries [MOCK] prefix
    assert "[MOCK]" in summary
