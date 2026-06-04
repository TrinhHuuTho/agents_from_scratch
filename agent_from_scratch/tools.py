from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from typing import Any, Callable
from googletrans import Translator
import python_weather


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


async def get_weather(city: str) -> str:
    """
        Get weather for a given city.
        Arguments: City: The city to look up.
        Returns: A string describing the current weather in the city.
    """
    async with python_weather.Client() as client:
        forecast = await client.get(city)

        temperature = None
        sky_text = None

        if hasattr(forecast, "current") and getattr(forecast, "current") is not None:
            cur = getattr(forecast, "current")
            temperature = getattr(cur, "temperature", None) or getattr(
                cur, "temp", None
            )
            sky_text = getattr(cur, "sky_text", None) or getattr(cur, "condition", None)

        if temperature is None and hasattr(forecast, "temperature"):
            temperature = getattr(forecast, "temperature", None)
            sky_text = getattr(forecast, "description", None) or getattr(
                forecast, "kind", None
            )

        if temperature is None and hasattr(forecast, "forecasts"):
            forecasts = getattr(forecast, "forecasts") or []
            if forecasts:
                first = forecasts[0]
                temperature = getattr(first, "temperature", None) or getattr(
                    first, "temp", None
                )
                sky_text = getattr(first, "sky_text", None) or getattr(
                    first, "condition", None
                )

        if temperature is None and hasattr(forecast, "as_dict"):
            try:
                d = forecast.as_dict()
                temperature = (
                    d.get("current", {}).get("temperature")
                    if isinstance(d, dict)
                    else None
                )
                sky_text = (
                    d.get("current", {}).get("sky_text")
                    if isinstance(d, dict)
                    else None
                )
            except Exception:
                pass

        if temperature is None and sky_text is None:
            return f"Could not get weather for {city}."

        weather_data = json.dumps({"temperature": temperature, "sky_text": sky_text})

    return weather_data


async def translate(text: str, target_language: str) -> str:
    """Translate text to a target language."""
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    if inspect.isawaitable(translated):
        translated = await translated

    translated_text = translated.text

    return f"'{text}' in {target_language} is '{translated_text}'"


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
