from dataclasses import dataclass
from typing import Any

import requests

from agent_from_scratch.config import Settings, load_settings


@dataclass
class ChatModel:
    """OpenRouter chat client for both chatbot and agent workflows."""

    api_key: str = ""
    model: str = "deepseek/deepseek-v4-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "agent-from-scratch"
    timeout_seconds: int = 90
    max_retries: int = 5

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ChatModel":
        """Build a chat model from loaded application settings.

        Args:
            settings: Preloaded settings to reuse. When omitted, the values are
                loaded from the environment.

        Returns:
            A configured chat model instance.
        """
        resolved_settings = settings or load_settings()
        return cls(
            api_key=resolved_settings.openrouter_api_key,
            model=resolved_settings.model,
            base_url=resolved_settings.base_url,
            app_name=resolved_settings.app_name,
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers for an OpenRouter call.

        Returns:
            Headers ready to send with the API request.
        """
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Set it before calling OpenRouter."
            )

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Referer": self.app_name,
            "X-Title": self.app_name,
        }

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request and return the assistant message.

        Args:
            payload: JSON body passed to the OpenRouter chat completions API.

        Returns:
            The assistant message from the first choice in the response.
        """
        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
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

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_enabled: bool = True,
    ) -> dict[str, Any]:
        """Call the OpenRouter chat completion API.

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Optional tool schema payloads.
            tool_choice: Optional tool selection policy.
            reasoning_enabled: Whether to request reasoning metadata.

        Returns:
            The assistant message from OpenRouter.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"enabled": reasoning_enabled},
        }

        if tools:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        return self._post_chat_completion(payload)

    def invoke(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the assistant message for a plain chatbot conversation.

        Args:
            messages: Conversation messages in OpenAI format.

        Returns:
            The assistant message payload.
        """
        return self.chat_completion(messages=messages)


OpenRouterClient = ChatModel
