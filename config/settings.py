"""
Centralised application settings loaded from environment variables / .env file.

Usage:
    from config.settings import settings

    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        base_url=settings.anthropic_base_url or None,
    )
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic / LLM
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_base_url: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"

    # Notification defaults
    notification_email_from: str = "noreply@loanapproval.local"
    notification_sms_from: str = "+10000000000"


settings = Settings()
