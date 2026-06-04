from __future__ import annotations

import json
import inspect
import asyncio
import threading
from typing import Any

from agent_from_scratch.openrouter_client import OpenRouterClient
from agent_from_scratch.tools import ToolDefinition


class SimpleAgent:
    def __init__(
        self,
        client: OpenRouterClient,
        system_prompt: str,
        tools: dict[str, ToolDefinition] | None = None,
        max_iterations: int = 6,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.max_iterations = max_iterations

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        messages = self._build_messages(inputs)
        tool_schemas = [tool.to_openai_tool() for tool in self.tools.values()]

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

        handler = tool_definition.handler

        def _run_coroutine_sync(coro):
            try:
                # If no running loop, run directly
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            else:
                # Running loop exists (e.g. in async servers). Run the coroutine in a new thread.
                result: dict[str, Any] = {}

                def _target() -> None:
                    result["value"] = asyncio.run(coro)

                t = threading.Thread(target=_target)
                t.start()
                t.join()
                return result.get("value")

        if inspect.iscoroutinefunction(handler):
            tool_output = _run_coroutine_sync(handler(**parsed_arguments))
        else:
            maybe = handler(**parsed_arguments)
            if inspect.isawaitable(maybe):
                tool_output = _run_coroutine_sync(maybe)
            else:
                tool_output = maybe

        # Normalize output to a string (JSON-serialize structured results)
        if isinstance(tool_output, (dict, list)):
            content = json.dumps(tool_output)
        else:
            content = "" if tool_output is None else str(tool_output)

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": content,
        }
