from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Job Search Assistant API"
    environment: str = "local"
    debug: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
