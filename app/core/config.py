# Centralised configuration — all runtime knobs live here and are read from environment
# variables (or .env). Using pydantic-settings means every setting is type-checked and
# validated at startup, not discovered at runtime when first used.
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Job Search Assistant API"
    environment: str = "local"
    debug: bool = True

    # LLM provider selection. Defaults to "mock" so the app runs with no API keys.
    llm_provider: str = "mock"
    gemini_api_key: str | None = None  # Only required when llm_provider="gemini"
    gemini_model: str = "gemini-2.5-flash"
    # OpenAI generation model; separate from embedding_model so generation and
    # vector search can evolve independently while sharing OPENAI_API_KEY.
    openai_model: str = "gpt-4.1-mini"

    # Database. Defaults to local SQLite — no PostgreSQL needed for local dev or tests.
    database_url: str = "sqlite:///./local.db"

    # Workflow mode. Controls whether generation uses the single-step path or the
    # multi-stage agentic LangGraph workflow.
    # Supported values: "single_step" | "agentic"
    workflow_mode: str = "single_step"

    # RAG pipeline settings. Disabled by default so existing behaviour is fully
    # preserved — no embedding calls happen unless RAG is explicitly turned on.
    rag_enabled: bool = False
    # text-embedding-3-small produces 1536-dimensional vectors; a good balance of
    # quality and cost for semantic similarity tasks at this scale.
    embedding_model: str = "text-embedding-3-small"
    # How many job descriptions to retrieve per query. 5 is a pragmatic default:
    # enough context to be useful, small enough to stay within prompt token budgets.
    retrieval_top_k: int = 5
    # Cosine similarity threshold below which matches are discarded. 0.75 filters
    # weak matches that add noise without adding relevant context to the prompt.
    similarity_threshold: float = 0.75
    # Tailored-artifact retrieval is a second optional RAG source. Disabled by
    # default so existing behaviour remains unchanged until explicitly enabled.
    artifact_retrieval_enabled: bool = False
    artifact_retrieval_top_k: int = 3
    artifact_similarity_threshold: float = 0.70
    # Shared OpenAI API key:
    # - generation when llm_provider="openai"
    # - embeddings when rag_enabled=True
    openai_api_key: str | None = None

    # extra="ignore" prevents startup failure when .env contains keys not defined above
    # (e.g. editor tooling or shell exports that bleed into the environment).
    model_config = {"env_file": ".env", "extra": "ignore"}


# Module-level singleton — import `settings` everywhere rather than calling Settings()
# repeatedly, so all code shares one validated instance.
settings = Settings()
