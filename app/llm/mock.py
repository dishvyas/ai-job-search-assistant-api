from app.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """
    Deterministic LLM provider for local development and testing.
    Returns a predictable response that includes a slice of the prompt,
    proving the provider was actually called with the right input.
    """

    def generate_text(self, prompt: str) -> str:
        prompt_preview = prompt[:80].strip().replace("\n", " ")
        return f'[MOCK LLM RESPONSE] Prompt received ({len(prompt)} chars): "{prompt_preview}..."'
