from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from agent_from_scratch.chat_models import ChatModel
from agent_from_scratch.tools import ToolDefinition


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
        max_iterations: int = 6,
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
        self.memory = SQLiteMemoryManager(max_messages=4)  # Kept small to easily trigger summarization for testing

        if isinstance(tools, list):
            from agent_from_scratch.tools import TOOL_REGISTRY
            self.tools = {
                name: TOOL_REGISTRY[name] 
                for name in tools 
                if name in TOOL_REGISTRY
            }
        else:
            self.tools = tools or {}

        if skills:
            for skill_dir in skills:
                self._load_skills_from_directory(skill_dir)

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
            self.system_prompt += f"\n\n<activated_skill>\n{content}\n</activated_skill>\n"
            logger.info("Loaded skill from %s", skill_path)
        except Exception as e:
            logger.error("Failed to load skill from %s: %s", skill_path, e)

    def _summarize_thread(self, thread_id: str) -> None:
        """Summarize the thread if it's too long."""
        raw_msgs = self.memory.get_raw_messages_for_summarization(thread_id)
        if not raw_msgs:
            return
            
        summary_prompt = [
            {"role": "system", "content": "You are a helpful assistant. Please summarize the following conversation concisely. Include key facts like user names, preferences, or decisions made."},
            {"role": "user", "content": f"Conversation history:\n{json.dumps(raw_msgs, ensure_ascii=False, indent=2)}"}
        ]
        
        summary_response = self.client.chat_completion(messages=summary_prompt)
        new_summary = summary_response.get("content", "")
        if new_summary:
            self.memory.save_summary_and_clear(thread_id, new_summary)


    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the agent until it returns a final assistant message.

        Args:
            inputs: Dictionary containing a `messages` conversation list and a `thread_id`.

        Returns:
            The accumulated message list and the last assistant message.
        """
        import logging
        logger = logging.getLogger(__name__)

        thread_id = inputs.get("thread_id")
        if not thread_id:
            raise ValueError("A 'thread_id' is required in inputs to manage memory.")

        new_messages = inputs.get("messages", [])
        if new_messages:
            self.memory.add_messages(thread_id, new_messages)
            
        # Check if we need to summarize before getting the context window
        if self.memory.needs_summarization(thread_id):
            logger.info("Triggering thread summarization...")
            self._summarize_thread(thread_id)
            logger.info("Summarization complete.")

        # Get full context (System prompt + optional summary + recent messages)
        current_context = self.memory.get_messages(thread_id, self.system_prompt)
        tool_schemas = [_tool_to_openai_schema(tool) for tool in self.tools.values()]

        for i in range(self.max_iterations):
            logger.info(f"Agent loop iteration {i+1}...")
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
                return {"messages": current_context, "final_message": assistant_message}

            logger.info(f"Executing {len(tool_calls)} tool calls...")
            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                # Save tool result
                self.memory.add_messages(thread_id, [tool_result])
                current_context.append(tool_result)

        raise RuntimeError(
            "Agent exceeded max_iterations before producing a final answer."
        )


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
