"""Typed runtime configuration via pydantic-settings.

Values come from the environment (or a `.env` file). The justfile runs the server
with `uv run --env-file=.env`, so the same `.env` that holds `GOOGLE_API_KEY` for
pydantic-ai also populates these settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Agent server settings, overridable via environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agent_model: str = "google:gemini-2.5-flash"
    host: str = "127.0.0.1"
    port: int = 8000

    # Base URL the *browser* (OpenWebUI's client) uses to reach this server, so
    # generated images can be served back as plain `<img>`-able URLs.
    public_base_url: str = "http://localhost:8000"
    images_dir: Path = Path("work_dir/generated_images")
