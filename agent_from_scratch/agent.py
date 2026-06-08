from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from agent_from_scratch.chat_models import ChatModel
from agent_from_scratch.tools import ToolDefinition


def _tool_to_openai_schema(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a custom tool object to an OpenAI tool schema.

    Args:
        tool: The custom tool to convert.

    Returns:
        An OpenAI-compatible tool payload.
    """
    return tool.to_openai_tool()


def _run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine from synchronous code.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine result.
    """
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


class SimpleAgent:
    """Drive a chat model until it produces a final answer."""

    def __init__(
        self,
        client: ChatModel,
        system_prompt: str,
        tools: dict[str, ToolDefinition] | None = None,
        max_iterations: int = 6,
    ) -> None:
        """Initialize the agent.

        Args:
            client: Chat client used for completions.
            system_prompt: Default system instruction to inject.
            tools: Tool registry keyed by tool name.
            max_iterations: Maximum tool-use iterations before failing.
        """
        self.client = client
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.max_iterations = max_iterations

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the agent until it returns a final assistant message.

        Args:
            inputs: Dictionary containing a `messages` conversation list.

        Returns:
            The accumulated message list and the last assistant message.
        """
        messages = self._build_messages(inputs)
        tool_schemas = [_tool_to_openai_schema(tool) for tool in self.tools.values()]

        for _ in range(self.max_iterations):
            assistant_message = self.client.chat_completion(
                messages=messages,
                tools=tool_schemas or None,
            )
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return {"messages": messages, "final_message": assistant_message}

            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                messages.append(tool_result)

        raise RuntimeError(
            "Agent exceeded max_iterations before producing a final answer."
        )

    def _build_messages(self, inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate and normalize the incoming message list.

        Args:
            inputs: Raw agent inputs.

        Returns:
            A message list with a system prompt prepended when needed.
        """
        raw_messages = list(inputs.get("messages", []))
        if not raw_messages:
            raise ValueError("invoke() requires a non-empty messages list.")

        if not any(message.get("role") == "system" for message in raw_messages):
            raw_messages.insert(
                0,
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
            )

        return raw_messages

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call returned by the model.

        Args:
            tool_call: Tool call payload from the assistant message.

        Returns:
            A tool role message containing the tool output.
        """
        function_data = tool_call.get("function") or {}
        tool_name = function_data.get("name")
        if tool_name not in self.tools:
            raise KeyError(f"Unknown tool requested: {tool_name}")

        raw_arguments = function_data.get("arguments") or "{}"
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Tool arguments for {tool_name} are not valid JSON: {raw_arguments}"
            ) from exc

        tool_definition = self.tools[tool_name]
        tool_output = tool_definition.invoke(parsed_arguments)

        if isinstance(tool_output, (dict, list)):
            content = json.dumps(tool_output)
        else:
            content = "" if tool_output is None else str(tool_output)

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": content,
        }
