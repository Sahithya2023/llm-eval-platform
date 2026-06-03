"""Application configuration.

All settings are environment-driven (loaded from a `.env` file or real
environment variables). Nothing secret or environment-specific is hard-coded.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "LLM Eval & Observability Platform"
    environment: str = "development"

    # --- Storage ---
    # SQLite for now. Because we go through SQLAlchemy, switching to Postgres
    # later is a URL change, not a code change.
    database_url: str = "sqlite:///./data/traces.db"
    data_dir: str = "./data"

    # --- Benchmark dataset (Stage 2) ---
    # Root of the extracted Spider release. Override with SPIDER_DIR in .env.
    spider_dir: str = "./data/benchmark/spider"

    # --- LLM connection (wired up from Stage 2 onward; defined here so all
    #     configuration lives in one place) ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # --- Evaluation defaults (logged with every trace for reproducibility) ---
    default_temperature: float = 0.0
    default_prompt_version: str = "v1"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once, reused everywhere)."""
    return Settings()
