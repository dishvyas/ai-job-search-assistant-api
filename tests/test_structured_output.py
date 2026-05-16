import json

import pytest
from fastapi.testclient import TestClient

from app.llm.exceptions import LLMOutputParsingError, LLMProviderUnavailableError
from app.llm.mock import MockLLMProvider
from app.llm.parsing import parse_tailoring_response
from app.main import app
from app.schemas.llm_output import TailoringLLMOutput

client = TestClient(app)

VALID_PAYLOAD = {
    "master_resume": "Software engineer with 5 years of Python experience.",
    "job_description": "Backend engineer role using FastAPI.",
}

# A minimal valid JSON string matching TailoringLLMOutput
VALID_LLM_JSON = json.dumps(
    {
        "tailored_summary": "Strong match for the role.",
        "tailored_bullets": ["Led API projects", "Improved system reliability"],
        "cover_letter_draft": "Dear Hiring Manager, I am a great fit.",
        "application_question_answers": ["Answer 1", "Answer 2"],
        "recruiter_message_draft": "Hi, I am interested in this role.",
        "fit_gap_analysis": "FIT: strong Python. GAP: no Kubernetes.",
        "interview_talking_points": ["Discuss API work", "Mention team projects"],
    }
)


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


def test_mock_provider_returns_valid_json():
    provider = MockLLMProvider()
    raw = provider.generate_text("test prompt")
    data = json.loads(raw)  # must not raise
    assert isinstance(data, dict)


def test_mock_provider_output_matches_schema():
    provider = MockLLMProvider()
    raw = provider.generate_text("test prompt")
    parsed = parse_tailoring_response(raw)  # must not raise
    assert isinstance(parsed, TailoringLLMOutput)


def test_mock_provider_output_has_non_empty_lists():
    provider = MockLLMProvider()
    raw = provider.generate_text("test prompt")
    parsed = parse_tailoring_response(raw)
    assert len(parsed.tailored_bullets) > 0
    assert len(parsed.interview_talking_points) > 0


# ---------------------------------------------------------------------------
# Parser — happy path
# ---------------------------------------------------------------------------


def test_parser_returns_tailoring_llm_output_on_valid_json():
    result = parse_tailoring_response(VALID_LLM_JSON)
    assert isinstance(result, TailoringLLMOutput)
    assert result.tailored_summary == "Strong match for the role."


def test_parser_preserves_list_fields():
    result = parse_tailoring_response(VALID_LLM_JSON)
    assert result.tailored_bullets == ["Led API projects", "Improved system reliability"]


# ---------------------------------------------------------------------------
# Parser — error cases
# ---------------------------------------------------------------------------


def test_parser_raises_on_malformed_json():
    with pytest.raises(LLMOutputParsingError, match="not valid JSON"):
        parse_tailoring_response("this is not json at all")


def test_parser_raises_on_json_with_missing_fields():
    incomplete = json.dumps({"tailored_summary": "Only this field"})
    with pytest.raises(LLMOutputParsingError, match="does not match expected schema"):
        parse_tailoring_response(incomplete)


def test_parser_raises_on_empty_string():
    with pytest.raises(LLMOutputParsingError):
        parse_tailoring_response("")


def test_parser_raises_on_json_array_instead_of_object():
    with pytest.raises(LLMOutputParsingError):
        parse_tailoring_response(json.dumps(["not", "an", "object"]))


# ---------------------------------------------------------------------------
# Service — parsing fallback
# ---------------------------------------------------------------------------


def test_service_falls_back_when_parsing_fails(monkeypatch):
    """If provider returns unparseable output, service falls back to mock."""

    class GarbageProvider:
        def generate_text(self, prompt: str) -> str:
            return "```json\nnot valid json\n```"

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider",
        lambda: GarbageProvider(),
    )

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "[Fallback mode used]" in body["tailored_summary"]


def test_service_falls_back_when_provider_unavailable(monkeypatch):
    """If provider raises LLMProviderError, service falls back to mock."""

    class FailingProvider:
        def generate_text(self, prompt: str) -> str:
            raise LLMProviderUnavailableError("503")

    monkeypatch.setattr(
        "app.services.application_tailoring.get_llm_provider",
        lambda: FailingProvider(),
    )

    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "[Fallback mode used]" in body["tailored_summary"]


# ---------------------------------------------------------------------------
# Endpoint — normal mock mode
# ---------------------------------------------------------------------------


def test_endpoint_returns_200_in_mock_mode():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_endpoint_response_contains_all_fields():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    for field in TailoringLLMOutput.model_fields:
        assert field in body, f"Missing field: {field}"


def test_endpoint_normal_mode_has_no_fallback_note():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    assert "[Fallback mode used]" not in body["tailored_summary"]


def test_endpoint_returns_non_empty_bullets_and_talking_points():
    response = client.post("/api/v1/applications/tailor", json=VALID_PAYLOAD)
    body = response.json()
    assert len(body["tailored_bullets"]) > 0
    assert len(body["interview_talking_points"]) > 0
