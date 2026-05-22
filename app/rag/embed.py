# Embedding generation — stateless wrapper around the OpenAI embeddings API.
# Keeping this as a single function (rather than a class) makes it trivial to
# monkeypatch in tests: `monkeypatch.setattr("app.rag.embed.generate_embedding", mock_fn)`.
from app.core.config import settings


def generate_embedding(text: str) -> list[float]:
    """
    Generate a 1536-dimensional embedding vector for the given text.

    Uses the configured embedding model (default: text-embedding-3-small).
    Returns a list of floats suitable for storing in a pgvector column.

    This function makes a real OpenAI API call when called with a live key.
    In tests, monkeypatch this function to return a fixed vector.
    """
    # Lazy import — openai is only needed when RAG is enabled, and we don't
    # want to force callers to have it installed when llm_provider=mock.
    from openai import OpenAI  # type: ignore[import-untyped]

    client = OpenAI(api_key=settings.openai_api_key or "")
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


# Dimension constant exported for use in mock generation and validation.
EMBEDDING_DIM = 1536
