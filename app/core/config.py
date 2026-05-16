from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Job Search Assistant API"
    environment: str = "local"
    debug: bool = True

    # LLM provider selection. Defaults to "mock" so the app runs with no API keys.
    llm_provider: str = "mock"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # Database. Defaults to local SQLite — no PostgreSQL needed for local dev or tests.
    database_url: str = "sqlite:///./local.db"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
