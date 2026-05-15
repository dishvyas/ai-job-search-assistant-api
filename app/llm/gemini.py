from app.llm.base import LLMProvider


class GeminiLLMProvider(LLMProvider):
    """
    LLM provider backed by Google Gemini.
    Requires a valid GEMINI_API_KEY environment variable.
    Uses the google-genai SDK (pip install google-genai).
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini. "
                "Set it in your .env file or environment."
            )
        self.model = model
        self._client = self._build_client(api_key)

    def _build_client(self, api_key: str):  # type: ignore[return]
        try:
            from google import genai  # type: ignore[import-untyped]

            return genai.Client(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' package is required for Gemini support. "
                "Install it with: pip install google-genai"
            ) from e

    def generate_text(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text
