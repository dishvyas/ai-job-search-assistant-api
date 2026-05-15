from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for all LLM provider implementations."""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Send a prompt and return the generated text response."""
        ...
