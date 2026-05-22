# Gemini provider implementation — wraps the google-genai SDK behind the
# LLMProvider interface so the rest of the codebase never imports google-genai directly.
from app.llm.base import LLMProvider
from app.llm.exceptions import LLMProviderUnavailableError


class GeminiLLMProvider(LLMProvider):
    """
    LLM provider backed by Google Gemini.
    Requires a valid GEMINI_API_KEY environment variable.
    Uses the google-genai SDK (pip install google-genai).
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        # Fail fast at construction time rather than on the first API call —
        # a missing key is a configuration error, not a transient runtime failure.
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini. "
                "Set it in your .env file or environment."
            )
        self.model = model
        self._client = self._build_client(api_key)

    def _build_client(self, api_key: str):  # type: ignore[return]
        # Lazy SDK import so that the class is importable even when google-genai
        # is not installed (e.g. CI environments that only run mock-mode tests).
        try:
            from google import genai  # type: ignore[import-untyped]

            return genai.Client(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' package is required for Gemini support. "
                "Install it with: pip install google-genai"
            ) from e

    def generate_text(self, prompt: str) -> str:
        # Broad except is intentional — the Gemini SDK raises several different
        # exception types (network, auth, quota). All are mapped to
        # LLMProviderUnavailableError so the fallback logic has one thing to catch.
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            raise LLMProviderUnavailableError(
                f"Gemini request failed ({type(e).__name__}): {e}"
            ) from e
