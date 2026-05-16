import json

from pydantic import ValidationError

from app.llm.exceptions import LLMOutputParsingError
from app.schemas.llm_output import TailoringLLMOutput


def parse_tailoring_response(raw_text: str) -> TailoringLLMOutput:
    """
    Parse and validate raw LLM output as a TailoringLLMOutput.

    Raises LLMOutputParsingError if:
    - the text is not valid JSON
    - the JSON does not match the expected schema
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise LLMOutputParsingError(
            f"LLM output is not valid JSON: {e}. Raw output preview: {raw_text[:200]!r}"
        ) from e

    try:
        return TailoringLLMOutput.model_validate(data)
    except ValidationError as e:
        raise LLMOutputParsingError(f"LLM JSON does not match expected schema: {e}") from e
