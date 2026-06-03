from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import requests


@dataclass
class OpenRouterClient:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "agent-from-scratch"
    timeout_seconds: int = 90

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_enabled: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"enabled": reasoning_enabled},
        }

        if tools:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

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
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter response did not include choices: {data}")

        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError(f"Unexpected OpenRouter message format: {message!r}")

        return message
