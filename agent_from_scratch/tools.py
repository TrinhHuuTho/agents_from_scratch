from __future__ import annotations

import asyncio
from datetime import datetime
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, get_origin

from googletrans import Translator
import python_weather


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a small JSON schema fragment.

    Args:
        annotation: The annotation to convert.

    Returns:
        A JSON-schema-like property definition.
    """
    if annotation in (inspect._empty, Any):
        return {}

    origin = get_origin(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    if annotation is list or origin is list:
        return {"type": "array"}

    return {}


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-style parameter schema from a function signature.

    Args:
        func: The function whose parameters should be described.

    Returns:
        A JSON schema object for the function arguments.
    """
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        schema = _annotation_to_schema(parameter.annotation)
        if parameter.default is not inspect._empty:
            schema["default"] = parameter.default
        else:
            required.append(parameter_name)

        properties[parameter_name] = schema

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required

    return parameters


def _resolve_workspace_path(file_path: str) -> Path:
    """Resolve a file path relative to the agent's workspace.

    Args:
        file_path: Absolute or workspace-relative path.

    Returns:
        A resolved path inside the workspace when relative.
    """
    path = Path(file_path)
    if path.is_absolute():
        return path

    return Path.cwd() / "workspace" / path


@dataclass(frozen=True)
class ToolDefinition:
    """Decorator-backed tool definition with sync and async execution helpers."""

    func: Callable[..., Any]
    name: str
    description: str
    parameters: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def invoke(self, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        call_kwargs = dict(arguments or {})
        call_kwargs.update(kwargs)
        result = self.func(**call_kwargs)
        if inspect.isawaitable(result):
            return _run_coroutine_sync(result)
        return result

    async def ainvoke(
        self, arguments: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        call_kwargs = dict(arguments or {})
        call_kwargs.update(kwargs)
        result = self.func(**call_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _run_coroutine_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _target() -> None:
        result["value"] = asyncio.run(coro)

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    return result.get("value")


def tool(func: Callable[..., Any]) -> ToolDefinition:
    """Wrap a function as a workspace tool definition.

    Args:
        func: The function to expose as a tool.

    Returns:
        A tool object with metadata and invoke helpers.
    """
    description = inspect.getdoc(func) or ""
    return ToolDefinition(
        func=func,
        name=func.__name__,
        description=description,
        parameters=_build_parameters_schema(func),
    )


@tool
async def get_weather(city: str) -> str:
    """Get weather for a given city.

    Args:
        city: The city to look up.

    Returns:
        A JSON string describing the current weather.
    """
    try:
        async with python_weather.Client() as client:
            forecast = await client.get(city)
    except Exception as exc:
        return f"Could not get weather for {city}: {exc}"

    temperature = None
    sky_text = None

    if hasattr(forecast, "current") and getattr(forecast, "current") is not None:
        cur = getattr(forecast, "current")
        temperature = getattr(cur, "temperature", None) or getattr(cur, "temp", None)
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
                d.get("current", {}).get("temperature") if isinstance(d, dict) else None
            )
            sky_text = (
                d.get("current", {}).get("sky_text") if isinstance(d, dict) else None
            )
        except Exception:
            pass

    if temperature is None and sky_text is None:
        return f"Could not get weather for {city}."

    return json.dumps({"temperature": temperature, "sky_text": sky_text})


@tool
async def translate(text: str, target_language: str) -> str:
    """Translate text to a target language.

    Args:
        text: The text to translate.
        target_language: The language code to translate to.

    Returns:
        A human-readable translation summary.
    """
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    if inspect.isawaitable(translated):
        translated = await translated

    translated_text = translated.text

    return f"'{text}' in {target_language} is '{translated_text}'"


@tool
def ls_tool(directory: str = ".") -> dict:
    """List the contents of a directory.

    Args:
        directory: The directory to list. Defaults to the current directory.

    Returns:
        A dictionary containing the directory path and a list of its contents,
        or an error message if the directory does not exist.
    """
    try:
        if not os.path.exists(directory):
            return {"error": f"Directory '{directory}' does not exist."}

        items = os.listdir(directory)
        results = []

        for item in items:
            item_path = os.path.join(directory, item)
            stat = os.stat(item_path)

            # Classify and retrieve metadata
            is_dir = os.path.isdir(item_path)
            modified_time = datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            results.append(
                {
                    "name": item,
                    "type": "directory" if is_dir else "file",
                    "size_bytes": stat.st_size if not is_dir else 0,
                    "last_modified": modified_time,
                }
            )

        return {"directory": os.path.abspath(directory), "contents": results}
    except Exception as e:
        return {"error": str(e)}


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: The path to the file to read.

    Returns:
        The contents of the file as a string.
    """
    try:
        resolved_path = _resolve_workspace_path(file_path)
        with open(resolved_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.

    Args:
        file_path: The path to the file to write.
        content: The content to write to the file.

    Returns:
        A success message or an error message if the write operation fails.
    """
    try:
        resolved_path = _resolve_workspace_path(file_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}."
    except Exception as e:
        return f"Error writing to file: {str(e)}"


@tool
def edit_file(file_path: str, new_content: str) -> str:
    """Edit the contents of a file by overwriting it with new content.

    Args:
        file_path: The path to the file to edit.
        new_content: The new content to write to the file.

    Returns:
        A success message or an error message if the edit operation fails.
    """
    return write_file(file_path, new_content)


def create_default_tool_registry() -> dict[str, Any]:
    """Build the default tool registry.

    Returns:
        A mapping from tool names to decorated tool objects.
    """
    return {
        get_weather.name: get_weather,
        translate.name: translate,
        ls_tool.name: ls_tool,
        read_file.name: read_file,
        write_file.name: write_file,
        edit_file.name: edit_file,
    }
