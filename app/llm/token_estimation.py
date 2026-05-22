"""
Lightweight token estimation utilities.

These functions use a simple word-count approximation rather than a real
tokenizer. Real tokenizers (tiktoken, sentencepiece, etc.) are provider-
specific, heavy to import, and unnecessary for educational cost tracking.

Rule of thumb:  tokens ≈ words  (rough, but good enough for dashboards)

Accuracy note: Real LLMs tokenize sub-word units; a short word like "I"
may be 1 token while a compound identifier like "ApplicationTailoringRun"
could be 3–4. This approximation is intentionally coarse and should NOT
be used for production billing calculations.
"""


def estimate_input_tokens(prompt: str) -> int:
    """Estimate the number of input tokens in a prompt string."""
    # Word split is O(n) and dependency-free — appropriate for a rough signal.
    return len(prompt.split())


def estimate_output_tokens(output: str) -> int:
    """Estimate the number of output tokens in an LLM response string."""
    return len(output.split())
