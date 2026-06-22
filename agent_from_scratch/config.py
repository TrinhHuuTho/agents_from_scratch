from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    model: str = "deepseek/deepseek-v4-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "agent-from-scratch"
    system_prompt: str = "You are a helpful assistant. Once you have enough information to answer, answer immediately; don't use any additional tools."
    max_iterations: int = 10


def load_settings() -> Settings:
    """Load application settings from the environment.

    Returns:
        Settings loaded from environment variables and .env files.
    """
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to your environment or .env file."
        )

    return Settings(
        openrouter_api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", Settings.model),
        base_url=os.getenv("OPENROUTER_BASE_URL", Settings.base_url).rstrip("/"),
        app_name=os.getenv("OPENROUTER_APP_NAME", Settings.app_name),
        system_prompt=os.getenv("SYSTEM_PROMPT", Settings.system_prompt),
        max_iterations=int(os.getenv("MAX_ITERATIONS", str(Settings.max_iterations))),
    )
