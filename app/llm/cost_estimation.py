"""
Lightweight cost estimation for LLM generation calls.

⚠️  EDUCATIONAL ESTIMATES ONLY
These figures are rough approximations based on publicly advertised pricing
at the time of writing. They are NOT production billing accuracy. Actual
costs depend on exact tokenization, model version, API tier, regional
pricing, and discounts. Do not use these values for financial reporting.

Purpose: surface a rough dollar signal in workflow metadata so engineers
can see relative cost of different providers and prompt sizes at a glance.
"""

# Approximate USD cost per 1 000 tokens (input / output).
# Sources: Google AI Studio pricing page (Gemini 2.x Flash class models).
_INPUT_COST_PER_1K: dict[str, float] = {
    "mock": 0.0,
    "fallback-mock": 0.0,
    # Gemini Flash: ~$0.075 per 1M input tokens → $0.000075 per 1K
    "gemini": 0.000075,
}

_OUTPUT_COST_PER_1K: dict[str, float] = {
    "mock": 0.0,
    "fallback-mock": 0.0,
    # Gemini Flash: ~$0.30 per 1M output tokens → $0.000300 per 1K
    "gemini": 0.000300,
}


def estimate_generation_cost(
    input_tokens: int,
    output_tokens: int,
    provider: str,
) -> float:
    """
    Return a rough USD cost estimate for a single generation call.

    provider should be one of: "mock", "gemini", "fallback-mock".
    Unknown providers default to 0.0 (safe no-op).
    """
    # Normalise to the base provider key (e.g. "gemini-2.5-flash" → "gemini")
    provider_key = _resolve_provider_key(provider)

    input_cost = (input_tokens / 1_000) * _INPUT_COST_PER_1K.get(provider_key, 0.0)
    output_cost = (output_tokens / 1_000) * _OUTPUT_COST_PER_1K.get(provider_key, 0.0)
    return round(input_cost + output_cost, 8)


def _resolve_provider_key(provider: str) -> str:
    """Map a provider_used value to the pricing table key."""
    if provider in _INPUT_COST_PER_1K:
        return provider
    # Handles variant model names like "gemini-2.5-flash" that aren't in the table directly.
    if provider.startswith("gemini"):
        return "gemini"
    # Unknown provider — treat as free to avoid crashing on new providers added in future.
    return "mock"
