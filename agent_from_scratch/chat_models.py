from agent_from_scratch.config import load_settings
from dataclasses import dataclass
import requests

settings = load_settings()

@dataclass
class ChatModel:
    api_key: str = settings.openrouter_api_key
    model: str = "deepseek/deepseek-v4-flash"
    base_url: str = settings.base_url
    app_name: str = settings.app_name
    timeout_seconds: int = 90
    max_retries: int = 5

    def invoke(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, str]:
        payload: dict[str, str] = {
            "model": self.model,
            "messages": messages,
        }

        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Referer": self.app_name,
                "X-Title": self.app_name,
            },
            json=payload,
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            text = response.text or ""
            snippet = text[:500] + ("..." if len(text) > 500 else "")
            raise RuntimeError(
                f"OpenRouter request failed with status {response.status_code}: {snippet}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            text = response.text or ""
            snippet = text[:500] + ("..." if len(text) > 500 else "")
            raise RuntimeError(
                f"Failed to decode JSON from OpenRouter (status {response.status_code}): {snippet}"
            ) from exc

        message = data["choices"][0]["message"] or {}
        if not isinstance(message, dict):
            raise RuntimeError(f"Unexpected OpenRouter message format: {message!r}")

        return message
