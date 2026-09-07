"""
config.py

Centralizes every configurable value (data file path, default cluster
count, CORS origins) so nothing is hard-coded inside main.py. Values can
be overridden via environment variables or a local .env file, without
touching a single line of code. This is the standard pattern used before
deploying any real service — it's what lets the *same* code run on your
laptop, in CI, and on a production server with different settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Data ---
    data_path: str = "online_retail_II.xlsx"
    default_k: int = 4

    # --- API metadata ---
    api_title: str = "Customer Segmentation API"
    api_version: str = "1.0.0"

    # --- Security ---
    # Wildcard is fine for local development. Before deploying, set this to
    # the exact origin(s) that should be allowed to call the API, e.g.
    # CORS_ORIGINS=["https://my-dashboard.onrender.com"]
    cors_origins: List[str] = ["*"]

    # --- Paths (not overridable via env; derived from the project layout) ---
    base_dir: Path = Path(__file__).parent
    templates_dir: Path = base_dir / "templates"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance. lru_cache ensures the .env file and
    environment are only read once per process, and lets the rest of the
    app just call get_settings() wherever it needs a config value.
    """
    return Settings()
