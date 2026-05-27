# OpenAI provider implementation — wraps the OpenAI SDK behind the
# LLMProvider interface so the rest of the codebase never imports OpenAI directly.
from app.llm.base import LLMProvider
from app.llm.exceptions import LLMProviderUnavailableError


class OpenAILLMProvider(LLMProvider):
    """
    LLM provider backed by OpenAI.
    Requires a valid OPENAI_API_KEY environment variable.
    Uses the OpenAI Python SDK.
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini") -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in your .env file or environment."
            )
        self.model = model
        self._client = self._build_client(api_key)

    def _build_client(self, api_key: str):  # type: ignore[return]
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            return OpenAI(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAI support. "
                "Install it with: pip install openai"
            ) from e

    def generate_text(self, prompt: str) -> str:
        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
            )

            text = getattr(response, "output_text", None)
            if text:
                return str(text)

            # Defensive fallback for SDK response-shape changes.
            output = getattr(response, "output", None)
            if output:
                chunks: list[str] = []
                for item in output:
                    content = getattr(item, "content", None) or []
                    for part in content:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            chunks.append(str(part_text))
                if chunks:
                    return "\n".join(chunks)

            raise LLMProviderUnavailableError("OpenAI response did not contain text output.")
        except LLMProviderUnavailableError:
            raise
        except Exception as e:
            raise LLMProviderUnavailableError(
                f"OpenAI request failed ({type(e).__name__}): {e}"
            ) from e
