# Two distinct exception hierarchies — one for provider failures, one for output
# failures — because the recovery action differs:
#   LLMProviderError     → retry or fall back to mock (provider is broken)
#   LLMOutputParsingError → fall back to mock (provider worked, output was malformed)
# Keeping them separate means callers can catch each case independently if needed.
class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""


class LLMProviderUnavailableError(LLMProviderError):
    """
    Raised when the provider is temporarily unavailable.
    Examples: 503 high demand, 429 rate limit, network timeout.
    """


class LLMOutputParsingError(Exception):
    """
    Raised when the LLM response cannot be parsed or does not match
    the expected schema. Separate from LLMProviderError because the
    provider call itself succeeded — the problem is the output shape.
    """
