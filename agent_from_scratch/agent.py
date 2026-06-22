from __future__ import annotations

import asyncio
import inspect
import json
import threading
from typing import Any

from agent_from_scratch.chat_models import ChatModel
from agent_from_scratch.tools import ToolDefinition
from agent_from_scratch.runtime import Runtime, ToolRuntime

from agent_from_scratch.memory import SQLiteMemoryManager


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
        skills: list[str] | None = None,
        tools: list[str] | dict[str, ToolDefinition] | None = None,
        max_iterations: int = 10,
        sensitive_tools: list[str] | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            client: Chat client used for completions.
            system_prompt: Default system instruction to inject.
            skills: List of skill directories to load.
            tools: Tool registry keyed by tool name, or a list of tool names.
            max_iterations: Maximum tool-use iterations before failing.
        """
        self.client = client
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.memory = SQLiteMemoryManager(max_messages=10)
        self.sensitive_tools = sensitive_tools or []

        if isinstance(tools, list):
            from agent_from_scratch.tools import TOOL_REGISTRY

            self.tools = {
                name: TOOL_REGISTRY[name] for name in tools if name in TOOL_REGISTRY
            }
        else:
            self.tools = tools or {}

        if skills:
            for skill_dir in skills:
                self._load_skills_from_directory(skill_dir)

        #  Add stopping instruction to system prompt
        self.system_prompt += (
            "\n\nIMPORTANT RULE: Once you have gathered enough information "
            "to answer the user's question, stop calling tools immediately "
            "and provide your final answer. Do not fetch additional documentation "
            "if you already have what you need from the context or previous tool calls. "
            "Aim to answer concisely with minimal tool usage."
        )

    def _load_skills_from_directory(self, skill_dir: str) -> None:
        """Load skill instructions from a directory and append them to the system prompt.

        Args:
            skill_dir: Path to the skill directory containing a SKILL.md file.
        """
        import os
        import logging

        logger = logging.getLogger(__name__)

        skill_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_path):
            logger.warning("No SKILL.md found in %s", skill_dir)
            return

        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.system_prompt += (
                f"\n\n<activated_skill>\n{content}\n</activated_skill>\n"
            )
            logger.info("Loaded skill from %s", skill_path)
        except Exception as e:
            logger.error("Failed to load skill from %s: %s", skill_path, e)

    def _summarize_thread(self, thread_id: str) -> None:
        """Summarize the thread if it's too long."""
        raw_msgs = self.memory.get_raw_messages_for_summarization(thread_id)
        if not raw_msgs:
            return

        summary_prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please summarize the following conversation concisely. Include key facts like user names, preferences, or decisions made.",
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{json.dumps(raw_msgs, ensure_ascii=False, indent=2)}",
            },
        ]

        summary_response = self.client.chat_completion(messages=summary_prompt)
        new_summary = summary_response.get("content", "")
        if new_summary:
            self.memory.save_summary_and_clear(thread_id, new_summary)

    def invoke(
        self,
        inputs: dict[str, Any],
        user_response: dict[str, Any] | None = None,
        runtime: Runtime | None = None,
        current_iteration: int = 0,  #  Add parameter for tracking iteration
    ) -> dict[str, Any]:
        """Run the agent, pausing for HITL if a sensitive tool is called.

        Args:
            inputs: Dictionary containing a `messages` conversation list and a `thread_id`.
            user_response: Dict containing human resolution if
                resuming (e.g., {approve, edit, reject, respond})
            current_iteration: Current iteration count when resuming from HITL.
        Returns:
            The accumulated message list and the last assistant message.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Create a default Runtime if none provided
        if runtime is None:
            runtime = Runtime()

        thread_id = inputs.get("thread_id")
        if not thread_id:
            raise ValueError("A 'thread_id' is required in inputs to manage memory.")

        new_messages = inputs.get("messages", [])
        if new_messages and not user_response:
            self.memory.add_messages(thread_id, new_messages)

        # Check if we need to summarize
        if self.memory.needs_summarization(thread_id):
            logger.info("Triggering thread summarization...")
            self._summarize_thread(thread_id)
            logger.info("Summarization complete.")

        # Get full context
        current_context = self.memory.get_messages(thread_id, self.system_prompt)
        tool_schemas = [_tool_to_openai_schema(tool) for tool in self.tools.values()]

        # Handle HITL resume
        if user_response:
            logger.info("Resuming from human response: %s", user_response)
            pending_tool_call = user_response["tool_call"]
            desicion = user_response["decision"]

            if desicion == "approve":
                tool_result = self._execute_tool_call(pending_tool_call, runtime)
            elif desicion == "edit":
                pending_tool_call["function"]["arguments"] = json.dumps(
                    user_response["edited_args"]
                )
                tool_result = self._execute_tool_call(pending_tool_call, runtime)
            elif desicion == "reject":
                tool_result = {
                    "role": "tool",
                    "tool_call_id": pending_tool_call.get("id"),
                    "content": "Tool execution was explicitly denied/rejected by the human supervisor.",
                }
            else:
                tool_result = {
                    "role": "tool",
                    "tool_call_id": pending_tool_call.get("id"),
                    "content": user_response["human_message"],
                }

            # Save the tool result
            self.memory.add_messages(thread_id, [tool_result])
            current_context.append(tool_result)

        #  Continue loop from current_iteration
        for i in range(current_iteration, self.max_iterations):
            logger.info(f"Agent loop iteration {i+1}/{self.max_iterations}...")

            assistant_message = self.client.chat_completion(
                messages=current_context,
                tools=tool_schemas or None,
            )

            # Save assistant message
            self.memory.add_messages(thread_id, [assistant_message])
            current_context.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                logger.info("No tool calls, returning final message.")
                return {
                    "status": "completed",
                    "messages": current_context,
                    "final_message": assistant_message,
                }

            logger.info(f"Executing {len(tool_calls)} tool calls...")
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")

                if tool_name in self.sensitive_tools:
                    logger.warning(f"Interrupting loop for sensitive tool: {tool_name}")
                    return {
                        "status": "requires_action",
                        "tool_call": tool_call,
                        "message": f"Tool '{tool_name}' requires supervisor approval.",
                        "current_iteration": i + 1,  # Save the next iteration
                    }

                tool_result = self._execute_tool_call(tool_call, runtime)
                # Save tool result
                self.memory.add_messages(thread_id, [tool_result])
                current_context.append(tool_result)

        raise RuntimeError(
            "Agent exceeded max_iterations before producing a final answer."
        )

    def _execute_tool_call(
        self, tool_call: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any]:
        """Execute one tool call returned by the model.

        Args:
            tool_call: Tool call payload from the assistant message.
            runtime: Runtime object to pass to the tool for dependency injection.

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

        # Retrieve the tool function
        tool = self.tools[tool_name]

        # If the tool expects a 'runtime' argument, inject it via ToolRuntime wrapper
        sig = inspect.signature(tool)
        if "runtime" in sig.parameters:
            parsed_arguments["runtime"] = ToolRuntime(runtime)

        # Execute the tool safely
        try:
            tool_output = tool(**parsed_arguments)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {str(e)}"

        # Prepare content for the tool result message
        if isinstance(tool_output, (dict, list)):
            content = json.dumps(tool_output)
        else:
            content = (
                tool_output
                if isinstance(tool_output, str)
                else ("" if tool_output is None else str(tool_output))
            )

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": content,
        }