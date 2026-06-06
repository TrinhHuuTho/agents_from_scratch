import logging

from agent_from_scratch.agent import SimpleAgent
from agent_from_scratch.chat_models import ChatModel
from agent_from_scratch.config import load_settings
from agent_from_scratch.tools import create_default_tool_registry


def main() -> None:
    """Run the demo agent from the command line.

    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return

    client = ChatModel(
        api_key=settings.openrouter_api_key,
        model=settings.model,
        base_url=settings.base_url,
        app_name=settings.app_name,
    )

    agent = SimpleAgent(
        client=client,
        system_prompt=settings.system_prompt,
        tools=create_default_tool_registry(),
        max_iterations=settings.max_iterations,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Thời tiết ở Hanoi hôm nay thế nào? Và làm ơn dịch câu trả lời sang tiếng Anh nhé.",
                }
            ]
        }
    )

    final_message = result.get("messages", [])[-1] if result.get("messages") else None
    content = final_message.get("content") if isinstance(final_message, dict) else None
    if content:
        logger.info(content)
    else:
        logger.info("No final content available; final_message=%s", final_message)


if __name__ == "__main__":
    main()
