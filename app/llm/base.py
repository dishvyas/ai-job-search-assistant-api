# Provider abstraction — defines the interface every LLM integration must satisfy.
# Using an ABC ensures new providers can't be added without implementing generate_text,
# and service code can depend on the interface without knowing which vendor is behind it.
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for all LLM provider implementations."""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Send a prompt and return the generated text response."""
        ...
