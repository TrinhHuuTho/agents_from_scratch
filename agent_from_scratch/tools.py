from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def translate(text: str, target_language: str) -> str:
    """Translate text to a target language."""
    return f"'{text}' in {target_language} is '...'"


def create_default_tool_registry() -> dict[str, ToolDefinition]:
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get weather for a given city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to look up.",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        handler=get_weather,
    )

    translate_tool = ToolDefinition(
        name="translate",
        description="Translate text to a target language.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to translate.",
                },
                "target_language": {
                    "type": "string",
                    "description": "The language to translate to.",
                },
            },
            "required": ["text", "target_language"],
            "additionalProperties": False,
        },
        handler=translate,
    )

    return {weather_tool.name: weather_tool, translate_tool.name: translate_tool}
