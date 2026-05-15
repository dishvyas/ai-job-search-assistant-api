class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""


class LLMProviderUnavailableError(LLMProviderError):
    """
    Raised when the provider is temporarily unavailable.
    Examples: 503 high demand, 429 rate limit, network timeout.
    """
