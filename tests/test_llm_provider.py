import pytest
from fastapi.testclient import TestClient

from app.llm.exceptions import LLMProviderUnavailableError
from app.llm.factory import get_llm_provider
from app.llm.mock import MockLLMProvider
from app.llm.openai import OpenAILLMProvider
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


def test_factory_returns_openai_provider(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "openai")
    monkeypatch.setattr("app.llm.factory.settings.openai_api_key", "test-key")
    monkeypatch.setattr("app.llm.factory.settings.openai_model", "gpt-4.1-mini")
    monkeypatch.setattr(
        "app.llm.openai.OpenAILLMProvider._build_client",
        lambda self, api_key: object(),
    )

    provider = get_llm_provider()

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.model == "gpt-4.1-mini"


def test_factory_raises_for_unsupported_provider(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.llm_provider", "bogus")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_provider()


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------


def test_openai_provider_raises_value_error_when_api_key_missing():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        OpenAILLMProvider(api_key="")


def test_openai_provider_generate_text_returns_output_text(monkeypatch):
    class FakeResponses:
        def create(self, model: str, input: str):
            return type("FakeResponse", (), {"output_text": "hello from openai"})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(
        "app.llm.openai.OpenAILLMProvider._build_client",
        lambda self, api_key: FakeClient(),
    )
    provider = OpenAILLMProvider(api_key="test-key")

    assert provider.generate_text("test prompt") == "hello from openai"


def test_openai_provider_generate_text_uses_fallback_response_shape(monkeypatch):
    fake_part = type("FakePart", (), {"text": "fallback text"})()
    fake_item = type("FakeItem", (), {"content": [fake_part]})()
    fake_response = type("FakeResponse", (), {"output_text": None, "output": [fake_item]})()

    class FakeResponses:
        def create(self, model: str, input: str):
            return fake_response

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(
        "app.llm.openai.OpenAILLMProvider._build_client",
        lambda self, api_key: FakeClient(),
    )
    provider = OpenAILLMProvider(api_key="test-key")

    assert provider.generate_text("test prompt") == "fallback text"


def test_openai_provider_maps_client_errors(monkeypatch):
    class FakeResponses:
        def create(self, model: str, input: str):
            raise RuntimeError("boom")

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(
        "app.llm.openai.OpenAILLMProvider._build_client",
        lambda self, api_key: FakeClient(),
    )
    provider = OpenAILLMProvider(api_key="test-key")

    with pytest.raises(LLMProviderUnavailableError, match="OpenAI request failed"):
        provider.generate_text("test prompt")


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


def test_tailor_response_contains_llm_provider_output(db_session):
    """The completed run's tailored_summary should contain the [MOCK] provider marker."""
    post_response = client.post(
        "/api/v1/applications/tailor",
        json={
            "master_resume": "Software engineer with 5 years of Python experience.",
            "job_description": "Backend engineer role using FastAPI.",
        },
    )
    run_id = post_response.json()["run_id"]
    summary = client.get(f"/api/v1/applications/runs/{run_id}").json()["tailored_summary"]
    assert "[MOCK]" in summary
